#!/usr/bin/env python3
"""
evaluate_phase3_puente.py — Inferencia + Evaluación granular del modelo
================================================================================
de FASE 3 (PUENTE) sobre el Test Set 50% con el PROMPT DISEÑADO.

Pipeline (paralelo a evaluate_final.py de Fase 2):

FASE 1 — Inferencia visual + generación
    • Carga el modelo Phase 3 PUENTE (4 pasos):
        1) Base BLIP-2 bf16 (Salesforce/blip2-opt-2.7b)
        2) ViT  ← PeftModel(LoRA Fase 2) + merge_and_unload()    [permanente]
        3) LLM  ← PeftModel(LoRA Fase 2) + merge_and_unload()    [permanente]
        4) load_state_dict(puente_state.pt, strict=False)        [Phase 3]
           → reemplaza query_tokens + qformer + language_projection
    • Itera el Test Set (50%) con MimicCXRDataset
    • Genera reportes con num_beams=5, do_sample=False, rep_pen=1.1,
      no_repeat_ngram=3, max_new_tokens=400, use_cache=True
    • Exporta predicciones_Phase3_Puente.csv  (dicom_id, study_id,
      reference_report, generated_report)
    • Reanudable: si el CSV ya existe, salta los dicom_id ya generados.

FASE 2 — Evaluación granular fila por fila (Gold Standard)
    • CheXpert GT      → mimic-cxr-2.0.0-chexpert.csv (merge por study_id)
    • CheXpert Pred    → CheXbert oficial Stanford AIMI sobre generated_report
    • BLEU-4, ROUGE-L  → sobre (reference_report, generated_report)
    • F1-RadGraph      → RG_ERbar (radgraph-xl, reward_level='all', idx 2)
    • Guarda resultados_granulares_Phase3_Puente.csv  (todas las cols + métricas)
    • Calcula promedios globales → metricas_globales_Phase3_Puente.csv

Uso:
    python evaluate_phase3_puente.py
    python evaluate_phase3_puente.py --num-samples 200       # smoke test
    python evaluate_phase3_puente.py --skip-inference        # solo re-evaluar
    python evaluate_phase3_puente.py --resume                # reanudar inferencia

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Patch de compatibilidad (antes de cualquier import de radgraph)
# ─────────────────────────────────────────────────────────────────────────────
from transformers import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "encode_plus"):
    def _ep(self, text, text_pair=None, add_special_tokens=True,
            padding=False, truncation=False, max_length=None,
            stride=0, return_tensors=None, **kwargs):
        return self.__call__(text, text_pair=text_pair,
                             add_special_tokens=add_special_tokens,
                             padding=padding, truncation=truncation,
                             max_length=max_length, stride=stride,
                             return_tensors=return_tensors, **kwargs)
    PreTrainedTokenizerBase.encode_plus = _ep

if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
    def _bi(self, t0, t1=None):
        if t1 is None:
            return [self.cls_token_id] + t0 + [self.sep_token_id]
        return [self.cls_token_id] + t0 + [self.sep_token_id] + t1 + [self.sep_token_id]
    PreTrainedTokenizerBase.build_inputs_with_special_tokens = _bi


# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# cv2 sin hilos para no competir con DataLoader/torch
import cv2
cv2.setNumThreads(0)

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_scorer_module
from radgraph import F1RadGraph

# ── Módulos del proyecto ──────────────────────────────────────────────────────
from config import config
from model_utils import cargar_procesador
from data_loader import MimicCXRDataset
from preprocessing import estandarizar_imagen
from chexbert_official import CheXbertLabeler, CHEXBERT_LABELS

# ── Helpers de Clinical BERTScore (Bio_ClinicalBERT) reutilizados de Fase 2 ──
# Importar este módulo aplica el patch del tokenizer (no romper RadGraph) y
# garantiza la MISMA configuración que en reevaluar_sucio_vs_limpio.py:
#   model_type='emilyalsentzer/Bio_ClinicalBERT', num_layers=12,
#   rescale_with_baseline=False (no hay baseline oficial para modelos clínicos).
from reevaluar_sucio_vs_limpio import (
    bertscore_clinico_por_fila,
    cargar_bertscorer,
    CLINICAL_BERT_MODEL,
    CLINICAL_BERT_LAYERS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Constantes — rutas, modelo, generación
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR: Path = config.paths.base_dir

# Test set 50% (mismo split que Fase 2)
TEST_CSV_50PCT: Path   = BASE_DIR / "test_split_50pct.csv"
IMAGES_DIR_50PCT: Path = BASE_DIR / "imagenes_50_porciento"

# Pesos de Fase 2 (LoRAs LLM y ViT) que se fusionarán permanentemente
PHASE2_BEST_DIR: Path = BASE_DIR / "phase2_results-50pct" / "checkpoints" / "best_model_Config_B1"
PHASE2_LLM_DIR:  Path = PHASE2_BEST_DIR / "llm"
PHASE2_VIT_DIR:  Path = PHASE2_BEST_DIR / "vit"

# Pesos del Phase 3 PUENTE (qformer + language_projection + query_tokens)
PHASE3_PUENTE_DIR:    Path = BASE_DIR / "phase3_puente_results-50pct"
PHASE3_PUENTE_BEST:   Path = PHASE3_PUENTE_DIR / "checkpoints" / "best_model_Phase3_Puente"
PHASE3_PUENTE_STATE:  Path = PHASE3_PUENTE_BEST / "puente_state.pt"
PHASE3_PUENTE_META:   Path = PHASE3_PUENTE_BEST / "puente_meta.json"

# Outputs (paralelo a phase2_results-50pct/evaluation/)
EVAL_DIR:        Path = PHASE3_PUENTE_DIR / "evaluation"
PREDICTIONS_CSV: Path = EVAL_DIR / "predicciones_Phase3_Puente.csv"
GRANULAR_CSV:    Path = EVAL_DIR / "resultados_granulares_Phase3_Puente.csv"
GLOBAL_CSV:      Path = EVAL_DIR / "metricas_globales_Phase3_Puente.csv"

# Nombre canónico de la config (usado en el CSV de salida)
CONFIG_NAME: str = "Phase3_Puente"

# Parámetros de generación — IDÉNTICOS a evaluate_final.py para que las métricas
# sean DIRECTAMENTE comparables con Fase 2.
GEN_PARAMS: Dict = dict(
    num_beams            = 5,
    do_sample            = False,
    repetition_penalty   = 1.1,
    no_repeat_ngram_size = 3,
    max_new_tokens       = 400,
    use_cache            = True,
)

# Columnas CheXpert oficiales de MIMIC-CXR (orden = orden del CSV)
_CHEX_COLS    = CHEXBERT_LABELS
_PRED_PREFIX  = "pred_chexpert_"

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

SEP = "─" * 70


# ─────────────────────────────────────────────────────────────────────────────
# 4. Helpers de métricas (BLEU-4, ROUGE-L, RadGraph) — IGUAL que evaluate_final
# ─────────────────────────────────────────────────────────────────────────────

def _bleu4(reference: str, hypothesis: str) -> float:
    ref_tok = nltk.word_tokenize(reference.lower())
    hyp_tok = nltk.word_tokenize(hypothesis.lower())
    return sentence_bleu(
        [ref_tok], hyp_tok,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    )


_rouge_scorer = rouge_scorer_module.RougeScorer(["rougeL"], use_stemmer=True)


def _rouge_l(reference: str, hypothesis: str) -> float:
    return _rouge_scorer.score(reference, hypothesis)["rougeL"].fmeasure


def _radgraph_f1(f1rg: F1RadGraph, reference: str, hypothesis: str) -> float:
    try:
        mean_reward, _, _, _ = f1rg(refs=[reference], hyps=[hypothesis])
        if isinstance(mean_reward, (tuple, list)):
            return float(mean_reward[2])      # RG_ERbar
        return float(mean_reward)
    except Exception as exc:
        logger.warning(f"      RadGraph error: {exc}")
        return float("nan")


def _truncar_para_bert(textos: List[str], tokenizer, max_tokens: int = 510) -> List[str]:
    """
    Trunca cada texto a <= max_tokens TOKENS del tokenizador BERT clínico.

    Bio_ClinicalBERT (BERT-base) tiene un máximo de 512 tokens de posición;
    los reportes pueden superarlo. La truncación se aplica SOLO a la entrada
    de BERTScore; BLEU/ROUGE/RadGraph siguen usando el texto completo.

    Réplica EXACTA de exp1_metrics_comparison._truncar_para_bert.
    """
    out = []
    for t in textos:
        t = t if t else "."
        ids = tokenizer.encode(
            t, add_special_tokens=False,
            truncation=True, max_length=max_tokens,
        )
        dec = tokenizer.decode(ids, skip_special_tokens=True)
        out.append(dec if dec.strip() else ".")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Helpers CheXpert — comparación GT (CSV oficial) vs Pred (CheXbert)
# ─────────────────────────────────────────────────────────────────────────────

def _chexpert_pred_cols(labeler: CheXbertLabeler, report: str) -> Dict[str, float]:
    """Ejecuta CheXbert sobre el reporte y devuelve dict con prefijo pred_."""
    raw = labeler.label(report)
    return {
        f"{_PRED_PREFIX}{lbl.lower().replace(' ', '_')}": v
        for lbl, v in raw.items()
    }


def _chexpert_comparison_metrics(
    gt_row: pd.Series,
    pred_dict: Dict[str, float],
) -> Dict[str, float]:
    """
    Compara GT (CSV oficial) vs Pred (CheXbert) clase por clase y calcula
    micro Precision/Recall/F1 por reporte. Réplica EXACTA de la lógica de
    evaluate_final._chexpert_comparison_metrics.

        1.0  → positivo
        0.0  → negativo
       -1.0  → positivo (incierto = mencionado)
        NaN  → 0 (no mencionado = negativo)
    """
    tp = fp = fn = 0
    metrics: Dict[str, float] = {}

    for lbl in _CHEX_COLS:
        col_gt   = lbl
        col_pred = f"{_PRED_PREFIX}{lbl.lower().replace(' ', '_')}"

        gt_val   = gt_row.get(col_gt,   float("nan"))
        pred_val = pred_dict.get(col_pred, float("nan"))

        def _to_bin(v):
            if isinstance(v, float) and math.isnan(v):
                return None
            return 1 if float(v) != 0.0 else 0

        gt_bin   = _to_bin(gt_val)
        pred_bin = _to_bin(pred_val)

        if gt_bin is None and pred_bin is None:
            metrics[f"chexpert_match_{lbl.lower().replace(' ', '_')}"] = float("nan")
            continue
        if gt_bin   is None: gt_bin   = 0
        if pred_bin is None: pred_bin = 0

        match = 1.0 if gt_bin == pred_bin else 0.0
        metrics[f"chexpert_match_{lbl.lower().replace(' ', '_')}"] = match

        if   gt_bin == 1 and pred_bin == 1: tp += 1
        elif gt_bin == 0 and pred_bin == 1: fp += 1
        elif gt_bin == 1 and pred_bin == 0: fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if not (math.isnan(precision) or math.isnan(recall)) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    metrics["chexpert_precision"] = precision
    metrics["chexpert_recall"]    = recall
    metrics["chexpert_f1"]        = f1
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 6. CARGA DEL MODELO Phase 3 PUENTE  ← función CLAVE de este script
# ─────────────────────────────────────────────────────────────────────────────

def cargar_modelo_phase3_puente(
    base_model_name:   str  = config.model.model_name,
    ruta_lora_vit:     Path = PHASE2_VIT_DIR,
    ruta_lora_llm:     Path = PHASE2_LLM_DIR,
    ruta_puente_state: Path = PHASE3_PUENTE_STATE,
):
    """
    Carga el modelo BLIP-2 entrenado en Fase 3 PUENTE para inferencia.

    Es el INVERSO LÓGICO de preparar_modelo_fase_puente() (que prepara para
    entrenar): aquí el modelo se construye congelado y listo para `generate()`.

    Pasos (idénticos en orden a la preparación de entrenamiento, pero sin
    cambiar `requires_grad` — el modelo sale en modo eval):
        1) Carga base BLIP-2 (bfloat16, device_map='auto')
        2) Carga LoRA ViT (Fase 2) → merge_and_unload()  [fusión PERMANENTE]
        3) Carga LoRA LLM (Fase 2) → merge_and_unload()  [fusión PERMANENTE]
        4) load_state_dict(puente_state.pt, strict=False)
           → reemplaza ÚNICAMENTE las keys del puente:
                 query_tokens         (1, 32, 768)
                 qformer.*            (~ 105M params)
                 language_projection.*(~ 1.97M params)
             el resto del modelo (ViT y LLM fusionados) NO se toca.

    Returns:
        (model, device)
    """
    from transformers import Blip2ForConditionalGeneration
    from peft import PeftModel

    # Validación de archivos
    for path, label in [
        (Path(ruta_lora_vit) / "adapter_config.json", "ViT LoRA Fase 2"),
        (Path(ruta_lora_llm) / "adapter_config.json", "LLM LoRA Fase 2"),
        (Path(ruta_puente_state),                     "puente_state.pt Fase 3"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"  ✗ Falta {label}: {path}")

    logger.info(f"\n{SEP}")
    logger.info(f"  CARGA MODELO FASE 3 (PUENTE)  ·  {CONFIG_NAME}")
    logger.info(f"  base_model    : {base_model_name}")
    logger.info(f"  ViT LoRA F2   : {ruta_lora_vit}")
    logger.info(f"  LLM LoRA F2   : {ruta_lora_llm}")
    logger.info(f"  Puente F3     : {ruta_puente_state}")
    logger.info(f"{SEP}")

    # ── 1. Base BLIP-2 ───────────────────────────────────────────────────────
    logger.info("  [1/4] Cargando modelo base BLIP-2 (bfloat16, device_map='auto')...")
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype = torch.bfloat16,
        device_map  = "auto",
        max_memory  = {0: "30GiB"},
    )
    target_device = next(base_model.parameters()).device
    target_dtype  = next(base_model.parameters()).dtype
    logger.info(f"        device={target_device}  dtype={target_dtype}")

    # ── 2. ViT LoRA Fase 2 → merge_and_unload ────────────────────────────────
    logger.info(f"  [2/4] Fusionando ViT LoRA (Fase 2) permanentemente...")
    vit_peft = PeftModel.from_pretrained(
        base_model.vision_model, str(ruta_lora_vit), is_trainable=False
    )
    base_model.vision_model = vit_peft.merge_and_unload().to(
        device=target_device, dtype=target_dtype
    )
    del vit_peft
    logger.info(f"        ✓ ViT fusionado (sin wrapper PeftModel)")

    # ── 3. LLM LoRA Fase 2 → merge_and_unload ────────────────────────────────
    logger.info(f"  [3/4] Fusionando LLM LoRA (Fase 2) permanentemente...")
    llm_peft = PeftModel.from_pretrained(
        base_model.language_model, str(ruta_lora_llm), is_trainable=False
    )
    base_model.language_model = llm_peft.merge_and_unload().to(
        device=target_device, dtype=target_dtype
    )
    del llm_peft
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"        ✓ LLM fusionado (sin wrapper PeftModel)")

    # ── 4. Cargar pesos del PUENTE entrenados en Fase 3 ──────────────────────
    logger.info(f"  [4/4] Cargando puente_state.pt (Phase 3) en strict=False...")
    state = torch.load(str(ruta_puente_state), map_location="cpu", weights_only=True)

    # Mover a device + dtype del modelo antes de load_state_dict
    state = {k: v.to(device=target_device, dtype=target_dtype) for k, v in state.items()}

    # `missing` será LARGO (todas las keys de ViT+LLM fusionados no están en el .pt
    # del puente; eso es correcto y no hay que reportarlo).  `unexpected` debe ser
    # vacío.
    missing, unexpected = base_model.load_state_dict(state, strict=False)
    n_loaded = sum(t.numel() for t in state.values())
    logger.info(f"        ✓ Puente cargado: {len(state)} keys, {n_loaded:,} params")
    if unexpected:
        logger.warning(f"        ⚠ {len(unexpected)} keys INESPERADAS en puente_state.pt:")
        for k in unexpected[:5]:
            logger.warning(f"            {k}")

    # Verificación: los 3 bloques del puente fueron sobreescritos
    expected_prefixes = ("query_tokens", "qformer.", "language_projection.")
    loaded_prefixes = {k.split(".", 1)[0] for k in state.keys()}
    logger.info(f"        Bloques cargados: {sorted(loaded_prefixes)}")
    for pref_root in {"query_tokens", "qformer", "language_projection"}:
        if pref_root not in loaded_prefixes:
            logger.warning(f"        ⚠ FALTA bloque del puente: {pref_root}")

    # Modo eval (no entrenamos nada en inferencia)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    del state
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"  ✓ Modelo Phase 3 PUENTE listo en {target_device} | {target_dtype}\n")
    return base_model, target_device


# ─────────────────────────────────────────────────────────────────────────────
# 7. FASE 1 — Inferencia visual + generación (batched, reanudable)
# ─────────────────────────────────────────────────────────────────────────────

def fase1_inferencia(
    model,
    processor,
    device,
    output_csv:  Path,
    num_samples: Optional[int] = None,
    batch_size:  int           = 4,
    resume:      bool          = False,
) -> Path:
    """
    Itera el Test Set 50% y genera reportes con el prompt diseñado, exportando
    un CSV incremental (escritura tras cada lote).

    Si resume=True y el CSV ya existe, se cargan los dicom_id ya guardados y se
    saltan en el bucle (reanudación robusta tras interrupciones).
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Cargando Test Set 50%...")
    test_dataset = MimicCXRDataset(
        csv_path             = TEST_CSV_50PCT,
        images_dir           = IMAGES_DIR_50PCT,
        processor            = processor,
        image_size           = config.data.image_size,
        clahe_clip_limit     = config.data.clahe_clip_limit,
        clahe_tile_grid_size = config.data.clahe_tile_grid_size,
        max_length           = config.data.max_length,
        padding              = config.data.padding,
        prompt               = config.inference.default_prompt,
        augment              = False,
    )
    n_total = len(test_dataset)
    n_eval  = n_total if num_samples is None else min(num_samples, n_total)
    logger.info(f"  Test Set: {n_total:,} muestras | A evaluar: {n_eval:,}")
    logger.info(f"  Prompt base: {config.inference.default_prompt!r}")
    logger.info(f"  GEN_PARAMS : {GEN_PARAMS}")

    COLS = ["dicom_id", "study_id", "reference_report", "generated_report"]

    # ── Modo resume: detectar dicom_id ya generados ──────────────────────────
    procesados: set = set()
    if resume and output_csv.exists():
        try:
            _prev = pd.read_csv(output_csv, usecols=["dicom_id"])
            procesados = set(_prev["dicom_id"].astype(str).tolist())
        except Exception as exc:
            logger.warning(
                f"  [resume] no se pudo leer {output_csv.name}: {exc}; "
                f"se empieza desde cero."
            )
            procesados = set()

    if procesados:
        logger.info(
            f"  [RESUME] {len(procesados):,} reportes ya guardados → "
            f"se reanuda en modo append (faltan ~{max(0, n_eval - len(procesados)):,})."
        )
    else:
        # CSV nuevo con header
        pd.DataFrame(columns=COLS).to_csv(output_csv, index=False)
        logger.info(f"  CSV incremental iniciado: {output_csv}")

    batch_images:  List = []
    batch_prompts: List[str] = []
    batch_meta:    List[Dict] = []
    errores       = 0
    n_guardadas   = 0
    t0            = time.time()

    for idx in tqdm(range(n_eval), desc=f"  Inferencia {CONFIG_NAME}", unit="img"):
        row      = test_dataset.data.iloc[idx]
        dicom_id = str(row["dicom_id"])

        # Resume: saltar muestras ya generadas
        if procesados and dicom_id in procesados:
            continue

        study_id = str(row.get("study_id", ""))
        ref_text = str(row["report_text"]).strip()

        # Prompt con vista mapeada (igual que Fase 2)
        view_raw = row.get("ViewPosition", None)
        if view_raw is not None and not (
            isinstance(view_raw, float) and math.isnan(view_raw)
        ):
            vista = test_dataset._mapear_vista(str(view_raw))
        else:
            vista = "Unknown"
        prompt = f"[Context: {vista} view] {config.inference.default_prompt}"

        # Imagen
        imagen_path = test_dataset._buscar_imagen(dicom_id)
        if imagen_path is None:
            logger.warning(f"  Imagen no encontrada: {dicom_id}")
            errores += 1
            continue
        imagen_pil = estandarizar_imagen(
            imagen_path,
            target_size          = config.data.image_size,
            clahe_clip_limit     = config.data.clahe_clip_limit,
            clahe_tile_grid_size = config.data.clahe_tile_grid_size,
        )
        if imagen_pil is None:
            errores += 1
            continue

        batch_images.append(imagen_pil)
        batch_prompts.append(prompt)
        batch_meta.append({
            "dicom_id": dicom_id,
            "study_id": study_id,
            "ref_text": ref_text,
        })

        # Vaciar lote cuando esté lleno o sea la última iteración
        if len(batch_images) == batch_size or idx == n_eval - 1:
            if not batch_images:
                continue
            try:
                inputs = processor(
                    images        = batch_images,
                    text          = batch_prompts,
                    return_tensors= "pt",
                    padding       = True,
                ).to(device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, **GEN_PARAMS)

                decoded = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                rows = [
                    {
                        "dicom_id":         m["dicom_id"],
                        "study_id":         m["study_id"],
                        "reference_report": m["ref_text"],
                        "generated_report": rep.strip(),
                    }
                    for m, rep in zip(batch_meta, decoded)
                ]
                pd.DataFrame(rows, columns=COLS).to_csv(
                    output_csv, mode="a", header=False, index=False
                )
                n_guardadas += len(rows)

            except Exception as exc:
                logger.exception(f"  Error inferencia lote idx={idx}: {exc}")
                errores += len(batch_images)

            batch_images.clear()
            batch_prompts.clear()
            batch_meta.clear()

    dt = time.time() - t0
    logger.info(
        f"  Inferencia {CONFIG_NAME} OK: {n_guardadas:,} reportes nuevos | "
        f"{errores} errores | {dt/60:.1f} min"
    )
    logger.info(f"  → {output_csv}")

    del test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("  VRAM liberada\n")
    return output_csv


# ─────────────────────────────────────────────────────────────────────────────
# 8. FASE 2 — Evaluación granular fila por fila (Gold Standard)
# ─────────────────────────────────────────────────────────────────────────────

def fase2_evaluacion(
    preds_path:     Path,
    f1rg:           F1RadGraph,
    chexbert:       CheXbertLabeler,
    chexpert_csv:   Optional[Path],
    bert_scorer            = None,
    bert_tokenizer         = None,
    bert_batch:    int     = 64,
) -> pd.DataFrame:
    """
    Evaluación granular (BLEU-4 + ROUGE-L + RadGraph + CheXpert + Clinical
    BERTScore) sobre el CSV de predicciones. Réplica EXACTA de
    evaluate_final.fase2_evaluacion ampliada con BERTScore clínico (idem
    reevaluar_sucio_vs_limpio.py de Fase 2).

    BERTScore se calcula al FINAL del bucle, en una sola pasada batched
    (más eficiente que fila-por-fila en GPU).
    """
    logger.info(f"\n{SEP}")
    logger.info(f"  FASE 2 — Evaluación granular: {CONFIG_NAME}")
    logger.info(f"{SEP}")

    df = pd.read_csv(preds_path)
    logger.info(f"  Filas: {len(df):,}")

    # ── Merge GT oficial CheXpert ────────────────────────────────────────────
    has_chexpert_gt = False
    if chexpert_csv is not None:
        chexpert_csv = Path(chexpert_csv)
        if not chexpert_csv.exists():
            gz_alt = (
                Path(str(chexpert_csv) + ".gz")
                if not str(chexpert_csv).endswith(".gz")
                else Path(str(chexpert_csv)[:-3])
            )
            if gz_alt.exists():
                logger.info(f"  Usando variante: {gz_alt}")
                chexpert_csv = gz_alt

    if chexpert_csv is not None and Path(chexpert_csv).exists():
        logger.info(f"  Cargando CSV oficial CheXpert: {chexpert_csv}")
        df_chex = pd.read_csv(chexpert_csv, compression="infer")

        def _norm_sid(s):
            s = str(s).strip()
            return s.lstrip("s") if s.startswith("s") else s

        df["_sid"]      = df["study_id"].astype(str).apply(_norm_sid)
        df_chex["_sid"] = df_chex["study_id"].astype(str).apply(_norm_sid)

        rename_map = {lbl: lbl for lbl in _CHEX_COLS if lbl in df_chex.columns}
        df = df.merge(
            df_chex[["_sid"] + list(rename_map.keys())].drop_duplicates(subset=["_sid"]),
            on="_sid", how="left",
        )
        df.drop(columns=["_sid"], inplace=True)

        has_chexpert_gt = True
        n_matched = (
            df[_CHEX_COLS[0]].notna().sum() if _CHEX_COLS[0] in df.columns else 0
        )
        logger.info(f"  Filas con GT CheXpert tras merge: {n_matched:,}/{len(df):,}")
    else:
        if chexpert_csv is None:
            logger.warning("  --chexpert-csv no proporcionado: se omite GT CheXpert.")
        else:
            logger.warning(f"  CSV no encontrado: {chexpert_csv} — se omite GT CheXpert.")

    # ── Bucle de evaluación fila por fila ────────────────────────────────────
    rows_out: List[Dict] = []
    for idx, row in tqdm(df.iterrows(), total=len(df),
                         desc=f"  Métricas {CONFIG_NAME}", unit="row"):
        ref = str(row["reference_report"]).strip() or "no findings"
        hyp = str(row["generated_report"]).strip() or "no findings"

        out: Dict = {
            "dicom_id":         row.get("dicom_id", ""),
            "study_id":         row.get("study_id", ""),
            "reference_report": ref,
            "generated_report": hyp,
        }

        # BLEU-4
        try:
            out["bleu_4"] = _bleu4(ref, hyp)
        except Exception as exc:
            logger.warning(f"  [row {idx}] BLEU-4: {exc}")
            out["bleu_4"] = float("nan")

        # ROUGE-L
        try:
            out["rouge_l"] = _rouge_l(ref, hyp)
        except Exception as exc:
            logger.warning(f"  [row {idx}] ROUGE-L: {exc}")
            out["rouge_l"] = float("nan")

        # F1-RadGraph
        out["radgraph_f1"] = _radgraph_f1(f1rg, ref, hyp)

        # CheXpert Predicted (CheXbert sobre generated_report)
        try:
            pred_chex = _chexpert_pred_cols(chexbert, hyp)
        except Exception as exc:
            logger.warning(f"  [row {idx}] CheXbert pred: {exc}")
            pred_chex = {
                f"{_PRED_PREFIX}{lbl.lower().replace(' ', '_')}": float("nan")
                for lbl in _CHEX_COLS
            }
        out.update(pred_chex)

        # CheXpert: comparación GT vs Pred
        if has_chexpert_gt:
            try:
                cmp = _chexpert_comparison_metrics(row, pred_chex)
            except Exception as exc:
                logger.warning(f"  [row {idx}] CheXpert comparison: {exc}")
                cmp = {}
            out.update(cmp)

        rows_out.append(out)

    df_out = pd.DataFrame(rows_out)

    # ── Clinical BERTScore (Bio_ClinicalBERT, batched) ───────────────────────
    # Se calcula en una sola pasada GPU al final, sobre TODAS las filas. Esto
    # es mucho más eficiente que iterar fila-por-fila (BLEU/ROUGE/RadGraph sí
    # van fila-por-fila por flexibilidad y porque RadGraph batched ya se hace
    # internamente). Las entradas se truncan a 510 tokens (Bio_ClinicalBERT
    # = BERT-base, tope 512 tokens de posición).
    if bert_scorer is not None:
        logger.info(f"\n  Calculando Clinical BERTScore sobre {len(df_out):,} reportes...")
        refs_full = df_out["reference_report"].astype(str).fillna(".").tolist()
        hyps_full = df_out["generated_report"].astype(str).fillna(".").tolist()
        if bert_tokenizer is not None:
            refs_t = _truncar_para_bert(refs_full, bert_tokenizer)
            hyps_t = _truncar_para_bert(hyps_full, bert_tokenizer)
        else:
            refs_t, hyps_t = refs_full, hyps_full
        bs = bertscore_clinico_por_fila(
            bert_scorer, refs_t, hyps_t,
            batch_size=bert_batch, etiqueta=f"[{CONFIG_NAME}]",
        )
        df_out["bertscore_p"]  = bs["p"]
        df_out["bertscore_r"]  = bs["r"]
        df_out["bertscore_f1"] = bs["f1"]
        logger.info(
            f"  ClinBERT → P={bs['p'].mean():.4f}  "
            f"R={bs['r'].mean():.4f}  F1={bs['f1'].mean():.4f}"
        )

    df_out.to_csv(GRANULAR_CSV, index=False)
    logger.info(f"  CSV granular: {GRANULAR_CSV}  ({len(df_out.columns)} cols)")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# 9. Resumen global
# ─────────────────────────────────────────────────────────────────────────────

def calcular_resumen_global(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula promedios globales y guarda metricas_globales_Phase3_Puente.csv."""
    skip_cols = {"dicom_id", "study_id", "reference_report", "generated_report"}
    numeric_cols = [
        c for c in df.columns
        if c not in skip_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    row = {"config": CONFIG_NAME, "n_samples": len(df)}
    for col in numeric_cols:
        row[col] = df[col].mean(skipna=True)
    df_summary = pd.DataFrame([row])
    df_summary.to_csv(GLOBAL_CSV, index=False)
    logger.info(f"  Resumen global: {GLOBAL_CSV}")
    return df_summary


# ─────────────────────────────────────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inferencia + evaluación granular del modelo Phase 3 PUENTE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limitar a N muestras del Test Set (None = todas).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Tamaño de lote de inferencia (beam=5, max_new=400).")
    p.add_argument("--num-beams", type=int, default=GEN_PARAMS["num_beams"],
                   help="Número de beams para generación (default: 5; 3 es ~30%% más rápido).")
    p.add_argument("--max-new-tokens", type=int, default=GEN_PARAMS["max_new_tokens"],
                   help="Tope de tokens generados (default: 400).")
    p.add_argument("--chexpert-csv", type=str,
                   default=str(BASE_DIR / "mimic-cxr-2.0.0-chexpert.csv"),
                   help="CSV oficial de etiquetas CheXpert (.csv o .csv.gz).")
    p.add_argument("--skip-inference", action="store_true",
                   help="Reutilizar predicciones_Phase3_Puente.csv y solo (re)evaluar.")
    p.add_argument("--resume", action="store_true",
                   help="Reanudar inferencia: conserva el CSV existente y solo "
                        "genera los reportes faltantes (robusto tras pausa/reinicio).")
    p.add_argument("--puente-state", type=str,
                   default=str(PHASE3_PUENTE_STATE),
                   help="Ruta al puente_state.pt entrenado en Fase 3.")
    p.add_argument("--bert-batch", type=int, default=64,
                   help="Tamaño de lote de Bio_ClinicalBERT (BERTScore).")
    p.add_argument("--no-bertscore", action="store_true",
                   help="Saltar Clinical BERTScore (útil para smoke tests rápidos).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 11. Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    print("\n" + "=" * 80)
    print(f"  EVALUACIÓN — Modelo FASE 3 PUENTE  ·  prompt diseñado")
    print("=" * 80)
    print(f"  Config        : {CONFIG_NAME}")
    print(f"  Test set      : {TEST_CSV_50PCT}")
    print(f"  Imágenes      : {IMAGES_DIR_50PCT}")
    print(f"  ViT LoRA F2   : {PHASE2_VIT_DIR}")
    print(f"  LLM LoRA F2   : {PHASE2_LLM_DIR}")
    print(f"  Puente F3     : {args.puente_state}")
    print(f"  Outputs       : {EVAL_DIR}")
    print(f"  GEN_PARAMS    : beam={GEN_PARAMS['num_beams']} max_new={GEN_PARAMS['max_new_tokens']}")
    print(f"  GPU           : {'disponible ✓' if torch.cuda.is_available() else 'NO disponible ✗'}")
    print("=" * 80 + "\n")

    if not torch.cuda.is_available():
        logger.warning("CUDA no disponible: la inferencia será MUY lenta en CPU.")

    # Override de parámetros de generación desde CLI (no muta el dict global,
    # crea una copia local que se pasará a fase1).
    if args.num_beams != GEN_PARAMS["num_beams"] or args.max_new_tokens != GEN_PARAMS["max_new_tokens"]:
        GEN_PARAMS["num_beams"]      = args.num_beams
        GEN_PARAMS["max_new_tokens"] = args.max_new_tokens
        logger.info(
            f"  GEN_PARAMS override CLI: beam={args.num_beams} max_new={args.max_new_tokens}"
        )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    chexpert_csv = Path(args.chexpert_csv) if args.chexpert_csv else None
    if chexpert_csv is not None and not chexpert_csv.exists():
        gz_alt = Path(str(chexpert_csv) + ".gz")
        if gz_alt.exists():
            chexpert_csv = gz_alt

    # ── Procesador BLIP-2 ────────────────────────────────────────────────────
    logger.info("Cargando Blip2Processor...")
    processor = cargar_procesador(model_name=config.model.model_name)

    # ── FASE 1 — Inferencia ──────────────────────────────────────────────────
    if args.skip_inference and PREDICTIONS_CSV.exists():
        logger.info(f"--skip-inference: reutilizando {PREDICTIONS_CSV}")
    else:
        model, device = cargar_modelo_phase3_puente(
            base_model_name   = config.model.model_name,
            ruta_lora_vit     = PHASE2_VIT_DIR,
            ruta_lora_llm     = PHASE2_LLM_DIR,
            ruta_puente_state = Path(args.puente_state),
        )
        fase1_inferencia(
            model       = model,
            processor   = processor,
            device      = device,
            output_csv  = PREDICTIONS_CSV,
            num_samples = args.num_samples,
            batch_size  = args.batch_size,
            resume      = args.resume,
        )
        # Liberar el modelo BLIP-2 antes de cargar F1RadGraph + CheXbert
        del model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("  VRAM del modelo BLIP-2 liberada.\n")

    # ── FASE 2 — Métricas (RadGraph + CheXbert + Clinical BERTScore) ─────────────
    logger.info("Cargando F1RadGraph (reward_level='all', model_type='radgraph-xl')...")
    f1rg = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    logger.info("  F1RadGraph listo")

    logger.info("Cargando CheXbert oficial (Stanford AIMI)...")
    chexbert = CheXbertLabeler()
    logger.info("  CheXbert listo")

    bert_scorer    = None
    bert_tokenizer = None
    if not args.no_bertscore:
        logger.info(
            f"Cargando Clinical BERTScore (model='{CLINICAL_BERT_MODEL}', "
            f"num_layers={CLINICAL_BERT_LAYERS})..."
        )
        bert_scorer = cargar_bertscorer(
            model=CLINICAL_BERT_MODEL,
            num_layers=CLINICAL_BERT_LAYERS,
            device=None,                                # autodetecta CUDA
        )
        # Tokenizer dedicado para truncar a 510 tokens antes de scorear
        from transformers import AutoTokenizer
        bert_tokenizer = AutoTokenizer.from_pretrained(CLINICAL_BERT_MODEL)
        logger.info("  Clinical BERTScore listo\n")
    else:
        logger.info("  Clinical BERTScore: OMITIDO (--no-bertscore)\n")

    df_gran = fase2_evaluacion(
        preds_path     = PREDICTIONS_CSV,
        f1rg           = f1rg,
        chexbert       = chexbert,
        chexpert_csv   = chexpert_csv,
        bert_scorer    = bert_scorer,
        bert_tokenizer = bert_tokenizer,
        bert_batch     = args.bert_batch,
    )

    # ── Resumen global ───────────────────────────────────────────────────────
    logger.info(f"\n{SEP}")
    logger.info("  Calculando resumen global de métricas...")
    df_summary = calcular_resumen_global(df_gran)

    show_cols = [
        "config", "n_samples", "bleu_4", "rouge_l", "radgraph_f1",
        "bertscore_p", "bertscore_r", "bertscore_f1",
        "chexpert_precision", "chexpert_recall", "chexpert_f1",
    ]
    show_cols = [c for c in show_cols if c in df_summary.columns]
    print(f"\n{'=' * 70}")
    print(f"  MÉTRICAS GLOBALES — {CONFIG_NAME}")
    print(f"{'=' * 70}")
    print(df_summary[show_cols].to_string(index=False))
    print(f"{'=' * 70}\n")

    print("=" * 80)
    print("  EVALUACIÓN PHASE 3 PUENTE COMPLETADA")
    print("=" * 80)
    print(f"  Predicciones    : {PREDICTIONS_CSV}")
    print(f"  Granular        : {GRANULAR_CSV}")
    print(f"  Resumen global  : {GLOBAL_CSV}")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
