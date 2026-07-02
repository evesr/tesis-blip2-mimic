#!/usr/bin/env python3
"""
evaluate_exp2_raddino.py — Inferencia + Evaluación granular del modelo
================================================================================
del EXPERIMENTO 2 (RadDINO + Puente Full-FT) sobre el Test Set 50% con el
PROMPT DISEÑADO.

Es el ESPEJO EXACTO de evaluate_phase3_puente.py, pero cambiando ÚNICAMENTE la
forma de construir el modelo: en lugar de fusionar el ViT LoRA de Fase 2 +
puente_state.pt, aquí se reemplaza el codificador visual EVA-CLIP por RadDINO
(+ LoRA + proyector 768→1408) mediante build_experiment_2_model() y se cargan
los pesos entrenados en `exp2_state.pt` con strict=False.

  ► Toda la lógica de inferencia, los GEN_PARAMS y las 5 métricas
    (BLEU-4 / ROUGE-L / F1-RadGraph / CheXpert / Clinical BERTScore) se
    REUTILIZAN sin cambios desde evaluate_phase3_puente.py, garantizando que
    los resultados sean DIRECTAMENTE comparables con Fase 2 y Fase 3 PUENTE.

Pipeline (paralelo a evaluate_phase3_puente.py):

FASE 1 — Inferencia visual + generación
    • Carga el modelo del Experimento 2 (cargar_modelo_exp2_raddino):
        1) build_experiment_2_model():
             a) Base BLIP-2 bf16 (Salesforce/blip2-opt-2.7b)
             b) LLM ← LoRA Fase 2 + merge_and_unload() → CONGELADO
             c) ViT ← RadDINO (microsoft/rad-dino) + LoRA + proyector 768→1408
             d) Puente (qformer + language_projection + query_tokens) en base FT
        2) load_state_dict(exp2_state.pt, strict=False)
             → reemplaza vision_model(LoRA+proyector) + qformer +
               language_projection + query_tokens con los pesos ENTRENADOS.
    • Itera el Test Set (50%) con MimicCXRDataset (mismo procesador 224×224 +
      CLAHE usado en el entrenamiento del Experimento 2).
    • Genera reportes con num_beams=5, do_sample=False, rep_pen=1.1,
      no_repeat_ngram=3, max_new_tokens=400, use_cache=True  (idéntico a Fase 2/3)
    • Exporta predicciones_Exp2_RadDINO.csv  (dicom_id, study_id,
      reference_report, generated_report)
    • Reanudable: si el CSV ya existe, salta los dicom_id ya generados.

FASE 2 — Evaluación granular fila por fila (Gold Standard)
    • CheXpert GT      → mimic-cxr-2.0.0-chexpert.csv (merge por study_id)
    • CheXpert Pred    → CheXbert oficial Stanford AIMI sobre generated_report
    • BLEU-4, ROUGE-L  → sobre (reference_report, generated_report)
    • F1-RadGraph      → RG_ERbar (radgraph-xl, reward_level='all', idx 2)
    • Clinical BERTScore → Bio_ClinicalBERT (12 capas, sin baseline) batched
    • Guarda resultados_granulares_Exp2_RadDINO.csv  (todas las cols + métricas)
    • Calcula promedios globales → metricas_globales_Exp2_RadDINO.csv

Uso:
    python evaluate_exp2_raddino.py
    python evaluate_exp2_raddino.py --num-samples 200       # smoke test
    python evaluate_exp2_raddino.py --skip-inference        # solo re-evaluar
    python evaluate_exp2_raddino.py --resume                # reanudar inferencia

NOTA: NO usa entornos virtuales — corre sobre el Python del sistema.

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Patch de compatibilidad (antes de cualquier import de radgraph)
#    Idéntico a evaluate_phase3_puente.py; es idempotente (solo aplica si falta).
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
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

# cv2 sin hilos para no competir con DataLoader/torch
import cv2
cv2.setNumThreads(0)

from radgraph import F1RadGraph

# ── Módulos del proyecto ──────────────────────────────────────────────────────
from config import config
from model_utils import cargar_procesador
from data_loader import MimicCXRDataset
from preprocessing import estandarizar_imagen
from chexbert_official import CheXbertLabeler, CHEXBERT_LABELS

# ── Helpers de MÉTRICAS reutilizados SIN CAMBIOS desde la evaluación PUENTE ──
# Importar evaluate_phase3_puente aplica el patch del tokenizer (idempotente) y
# trae EXACTAMENTE los mismos cálculos de BLEU-4, ROUGE-L, F1-RadGraph y
# CheXpert (GT vs Pred) — esto garantiza que las métricas del Experimento 2 sean
# DIRECTAMENTE comparables con Fase 2 y Fase 3 PUENTE. NO se redefine ninguna
# lógica de métrica; solo cambia cómo se construye el modelo.
from evaluate_phase3_puente import (
    _bleu4,
    _rouge_l,
    _radgraph_f1,
    _chexpert_pred_cols,
    _chexpert_comparison_metrics,
    _truncar_para_bert,
    _CHEX_COLS,
    _PRED_PREFIX,
    GEN_PARAMS as _PUENTE_GEN_PARAMS,
)

# ── Helpers de Clinical BERTScore (Bio_ClinicalBERT) — los mismos de Fase 2/3 ──
from reevaluar_sucio_vs_limpio import (
    bertscore_clinico_por_fila,
    cargar_bertscorer,
    CLINICAL_BERT_MODEL,
    CLINICAL_BERT_LAYERS,
)

# ── Constructor del Experimento 2 (RadDINO + Puente) y sus rutas ──────────────
from trainer_utils_exp2_raddino import (
    build_experiment_2_model,
    BASE_MODEL_NAME,
    RADDINO_MODEL_ID,
    PHASE2_LLM_DIR,
    OUTPUT_DIR,
    CHECKPOINTS_DIR,
    TEST_CSV as TEST_CSV_50PCT,
    IMAGES_DIR as IMAGES_DIR_50PCT,
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

# Nombre canónico de la corrida (define los nombres de los CSV de salida)
CONFIG_NAME: str = "Exp2_RadDINO"

# Pesos entrenados del Experimento 2 (best checkpoint, solo params entrenables)
EXP2_BEST_DIR: Path = CHECKPOINTS_DIR / f"best_model_{CONFIG_NAME}"
EXP2_STATE:    Path = EXP2_BEST_DIR / "exp2_state.pt"
EXP2_META:     Path = EXP2_BEST_DIR / "exp2_meta.json"

# Outputs (paralelo a phase3_puente_results-50pct/evaluation/)
EVAL_DIR:        Path = OUTPUT_DIR / "evaluation"
PREDICTIONS_CSV: Path = EVAL_DIR / f"predicciones_{CONFIG_NAME}.csv"
GRANULAR_CSV:    Path = EVAL_DIR / f"resultados_granulares_{CONFIG_NAME}.csv"
GLOBAL_CSV:      Path = EVAL_DIR / f"metricas_globales_{CONFIG_NAME}.csv"

# Parámetros de generación — COPIA local IDÉNTICA a evaluate_phase3_puente.GEN_PARAMS
# (se copia para poder sobre-escribir desde la CLI sin mutar el dict del otro
# módulo). Mismos valores → métricas directamente comparables con Fase 2/3.
GEN_PARAMS: Dict = dict(_PUENTE_GEN_PARAMS)

SEP = "─" * 70


# ─────────────────────────────────────────────────────────────────────────────
# 4. CARGA DEL MODELO Experimento 2  ← única función realmente NUEVA del script
# ─────────────────────────────────────────────────────────────────────────────

def cargar_modelo_exp2_raddino(
    ruta_exp2_state:  Path = EXP2_STATE,
    base_model_name:  str  = BASE_MODEL_NAME,
    raddino_model_id: str  = RADDINO_MODEL_ID,
    ruta_lora_llm:    Path = PHASE2_LLM_DIR,
    max_memory_gib:   int  = 30,
):
    """
    Reconstruye el modelo del Experimento 2 (RadDINO + Puente) para inferencia.

    Es el INVERSO LÓGICO de la preparación de entrenamiento: la arquitectura se
    reconstruye EXACTAMENTE igual con build_experiment_2_model() y luego se
    cargan los pesos entrenados (`exp2_state.pt`) — que contienen SOLO los
    parámetros entrenables — con strict=False.

    Pasos:
        1) build_experiment_2_model():
             • Base BLIP-2 bf16 (device_map='auto')
             • LLM  ← LoRA Fase 2 + merge_and_unload() → CONGELADO
             • ViT  ← RadDINO + LoRA (query, value) + proyector 768→1408
             • Puente (qformer + language_projection + query_tokens) liberado
           (esta llamada ejecuta su propia auditoría de parámetros)
        2) load_state_dict(exp2_state.pt, strict=False):
             • `missing`    será LARGO (todos los pesos congelados: backbone
               RadDINO + LLM fusionado + base no entrenable) → es CORRECTO.
             • `unexpected` debe ser vacío (los nombres coinciden porque la
               arquitectura se reconstruye igual).
        3) eval() + requires_grad=False en TODO el modelo (inferencia pura).

    Returns:
        (model, device)
    """
    ruta_exp2_state = Path(ruta_exp2_state)
    if not ruta_exp2_state.exists():
        raise FileNotFoundError(
            f"  ✗ Falta exp2_state.pt del Experimento 2: {ruta_exp2_state}\n"
            f"      Verifica {EXP2_BEST_DIR}/ (se genera al terminar el "
            f"entrenamiento con train_experiment2_raddino.py)."
        )

    logger.info(f"\n{SEP}")
    logger.info(f"  CARGA MODELO EXPERIMENTO 2 (RadDINO + Puente)  ·  {CONFIG_NAME}")
    logger.info(f"  base_model    : {base_model_name}")
    logger.info(f"  RadDINO       : {raddino_model_id}")
    logger.info(f"  LLM LoRA F2   : {ruta_lora_llm}")
    logger.info(f"  exp2_state.pt : {ruta_exp2_state}")
    logger.info(f"{SEP}")

    # ── 1. Reconstruir la arquitectura EXACTA del Experimento 2 ──────────────
    # build_experiment_2_model imprime su propia auditoría y deja el modelo con
    # los requires_grad de ENTRENAMIENTO (LLM=0, Q-Former=100%, ViT=solo LoRA).
    model = build_experiment_2_model(
        base_model_name    = base_model_name,
        raddino_model_id   = raddino_model_id,
        ruta_lora_llm      = ruta_lora_llm,
        train_query_tokens = True,            # precedente Fase 3 / Exp2
        max_memory_gib     = max_memory_gib,
    )

    # Referencia estable de device/dtype (qformer nunca cambia de device)
    _ref = next(model.qformer.parameters())
    target_device, target_dtype = _ref.device, _ref.dtype
    logger.info(f"  device={target_device}  dtype={target_dtype}")

    # ── 2. Cargar los pesos ENTRENADOS del Experimento 2 (strict=False) ──────
    logger.info(f"  Cargando exp2_state.pt (solo params entrenables) strict=False...")
    state = torch.load(str(ruta_exp2_state), map_location="cpu", weights_only=True)
    # Mover a device + dtype del modelo antes de load_state_dict
    state = {k: v.to(device=target_device, dtype=target_dtype) for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    n_loaded = sum(t.numel() for t in state.values())
    logger.info(f"  ✓ exp2_state cargado: {len(state)} keys, {n_loaded:,} params")
    if unexpected:
        logger.warning(f"  ⚠ {len(unexpected)} keys INESPERADAS en exp2_state.pt:")
        for k in unexpected[:5]:
            logger.warning(f"      {k}")

    # Verificación: los 4 bloques entrenados fueron sobreescritos
    #   vision_model (LoRA RadDINO + proyector) · qformer · language_projection ·
    #   query_tokens
    loaded_prefixes = {k.split(".", 1)[0] for k in state.keys()}
    logger.info(f"  Bloques entrenados cargados: {sorted(loaded_prefixes)}")
    for pref_root in ("vision_model", "qformer", "language_projection", "query_tokens"):
        if pref_root not in loaded_prefixes:
            logger.warning(f"  ⚠ FALTA bloque entrenado: {pref_root}")

    # ── 3. Modo eval + congelar TODO (inferencia pura) ───────────────────────
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    del state
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"  ✓ Modelo Experimento 2 listo en {target_device} | {target_dtype}\n")
    return model, target_device


# ─────────────────────────────────────────────────────────────────────────────
# 5. FASE 1 — Inferencia visual + generación (batched, reanudable)
#    Réplica EXACTA de evaluate_phase3_puente.fase1_inferencia (solo cambian
#    CONFIG_NAME y los paths del experimento; la lógica de generación es idéntica).
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
    import math

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

        # Prompt con vista mapeada (igual que Fase 2/3)
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
# 6. FASE 2 — Evaluación granular fila por fila (Gold Standard)
#    Réplica EXACTA de evaluate_phase3_puente.fase2_evaluacion.
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
    BERTScore) sobre el CSV de predicciones. Misma lógica fila-por-fila que
    evaluate_phase3_puente.fase2_evaluacion (CheXpert por enfermedad + micro
    P/R/F1 por reporte); BERTScore batched al final.
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

        # CheXpert: comparación GT vs Pred (P/R/F1 micro + match por enfermedad)
        if has_chexpert_gt:
            try:
                cmp = _chexpert_comparison_metrics(row, pred_chex)
            except Exception as exc:
                logger.warning(f"  [row {idx}] CheXpert comparison: {exc}")
                cmp = {}
            out.update(cmp)

        rows_out.append(out)

    df_out = pd.DataFrame(rows_out)

    # ── Clinical BERTScore (Bio_ClinicalBERT, batched al final) ──────────────
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
# 7. Resumen global
# ─────────────────────────────────────────────────────────────────────────────

def calcular_resumen_global(df: pd.DataFrame) -> pd.DataFrame:
    """Promedios globales y guarda metricas_globales_Exp2_RadDINO.csv."""
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
# 8. CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inferencia + evaluación granular del modelo Experimento 2 "
                    "(RadDINO + Puente) con prompt diseñado.",
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
                   help="Reutilizar predicciones_Exp2_RadDINO.csv y solo (re)evaluar.")
    p.add_argument("--resume", action="store_true",
                   help="Reanudar inferencia: conserva el CSV existente y solo "
                        "genera los reportes faltantes (robusto tras pausa/reinicio).")
    p.add_argument("--exp2-state", type=str,
                   default=str(EXP2_STATE),
                   help="Ruta al exp2_state.pt entrenado en el Experimento 2.")
    p.add_argument("--bert-batch", type=int, default=64,
                   help="Tamaño de lote de Bio_ClinicalBERT (BERTScore).")
    p.add_argument("--no-bertscore", action="store_true",
                   help="Saltar Clinical BERTScore (útil para smoke tests rápidos).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    print("\n" + "=" * 80)
    print(f"  EVALUACIÓN — Modelo EXPERIMENTO 2 (RadDINO + Puente)  ·  prompt diseñado")
    print("=" * 80)
    print(f"  Config        : {CONFIG_NAME}")
    print(f"  Test set      : {TEST_CSV_50PCT}")
    print(f"  Imágenes      : {IMAGES_DIR_50PCT}")
    print(f"  RadDINO       : {RADDINO_MODEL_ID}")
    print(f"  LLM LoRA F2   : {PHASE2_LLM_DIR}")
    print(f"  exp2_state.pt : {args.exp2_state}")
    print(f"  Outputs       : {EVAL_DIR}")
    print(f"  GEN_PARAMS    : beam={GEN_PARAMS['num_beams']} max_new={GEN_PARAMS['max_new_tokens']}")
    print(f"  GPU           : {'disponible ✓' if torch.cuda.is_available() else 'NO disponible ✗'}")
    print("=" * 80 + "\n")

    if not torch.cuda.is_available():
        logger.warning("CUDA no disponible: la inferencia será MUY lenta en CPU.")

    # Override de parámetros de generación desde CLI (sobre la copia LOCAL).
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

    # ── Procesador BLIP-2 (mismo que en el entrenamiento del Experimento 2) ──
    logger.info("Cargando Blip2Processor...")
    processor = cargar_procesador(model_name=config.model.model_name)

    # ── FASE 1 — Inferencia ──────────────────────────────────────────────────
    if args.skip_inference and PREDICTIONS_CSV.exists():
        logger.info(f"--skip-inference: reutilizando {PREDICTIONS_CSV}")
    else:
        model, device = cargar_modelo_exp2_raddino(
            ruta_exp2_state  = Path(args.exp2_state),
            base_model_name  = config.model.model_name,
            raddino_model_id = RADDINO_MODEL_ID,
            ruta_lora_llm    = PHASE2_LLM_DIR,
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
        logger.info("  VRAM del modelo Experimento 2 liberada.\n")

    # ── FASE 2 — Métricas (RadGraph + CheXbert + Clinical BERTScore) ─────────
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
    print("  EVALUACIÓN EXPERIMENTO 2 (RadDINO) COMPLETADA")
    print("=" * 80)
    print(f"  Predicciones    : {PREDICTIONS_CSV}")
    print(f"  Granular        : {GRANULAR_CSV}")
    print(f"  Resumen global  : {GLOBAL_CSV}")
    print("\n  Siguiente   : bash run_vllm_formatter_tmux.sh --tasks=disenado_exp2raddino")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
