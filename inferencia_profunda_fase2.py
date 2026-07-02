#!/usr/bin/env python3
"""
inferencia_profunda_fase2.py — Evaluación Cualitativa + Métricas (BLIP-2 dual-LoRA · 50 %)
==========================================================================================
Versión ejecutable (script) de la "Sección 1" del notebook de evaluación profunda.

Pipeline
--------
1.  Carga el modelo dual-LoRA de Fase 2 (Config_B1, 50 %):
        · LLM LoRA  → PeftModel (se mantiene el wrapper).
        · ViT LoRA  → merge_and_unload() (fusión OBLIGATORIA: generate() llama al
          ViT con pixel_values POSICIONAL, incompatible con el wrapper PeftModel).
2.  Selecciona N estudios aleatorios del test_split_50pct.csv.
3.  Genera reportes con hiperparámetros ESTRICTOS:
        max_new_tokens=512, repetition_penalty=1.2, num_beams=5, do_sample=False.
4.  Visualización comparativa (imagen + GT vs Generado) → PNG por estudio.
5.  Métricas de TEXTO (mismo pipeline que evaluate_final.py):
        BLEU-4, ROUGE-L, F1-RadGraph (radgraph-xl, reward_level='all' → RG_ERbar).
6.  Evaluación CLÍNICA CheXpert:
        GT  → mimic-cxr-2.0.0-chexpert.csv (merge por study_id).
        Pred→ CheXbert oficial sobre el reporte generado.
        Micro Precision/Recall/F1 por estudio + agreement por clase.

Salidas (en --output-dir, por defecto evaluacion_profunda_results/):
    predicciones_inferencia.csv      (dicom_id, study_id, reference, generated)
    metricas_texto.csv               (BLEU-4, ROUGE-L, F1-RadGraph por estudio)
    metricas_chexpert.csv            (precision, recall, f1 por estudio)
    agreement_chexpert_por_clase.csv (agreement por hallazgo clínico)
    comparativa_estudio_XX.png       (una figura por estudio)

Uso
---
    python inferencia_profunda_fase2.py
    python inferencia_profunda_fase2.py --num-samples 10 --seed 42
    python inferencia_profunda_fase2.py --skip-radgraph        # sin F1-RadGraph
    python inferencia_profunda_fase2.py --skip-chexpert        # sin CheXpert/CheXbert
    python inferencia_profunda_fase2.py --no-plots             # sin PNGs

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Patch de compatibilidad (ANTES de cualquier import de radgraph)
#    Idéntico a evaluate_final.py: radgraph espera la API antigua del tokenizer.
# ─────────────────────────────────────────────────────────────────────────────
from transformers import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "encode_plus"):
    def _ep(self, text, text_pair=None, add_special_tokens=True, padding=False,
            truncation=False, max_length=None, stride=0, return_tensors=None, **kwargs):
        return self.__call__(text, text_pair=text_pair, add_special_tokens=add_special_tokens,
                             padding=padding, truncation=truncation, max_length=max_length,
                             stride=stride, return_tensors=return_tensors, **kwargs)
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
import argparse
import gc
import logging
import math
import random
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")                 # backend sin display (guardar PNGs)
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Módulos del proyecto (arquitectura modular) ───────────────────────────────
from config import config
from model_utils import cargar_procesador
from data_loader import MimicCXRDataset
from preprocessing import estandarizar_imagen

# ─────────────────────────────────────────────────────────────────────────────
# 2. Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
SEP = "=" * 70

# ─────────────────────────────────────────────────────────────────────────────
# 3. Constantes y rutas (Fase 2 · 50 %)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = config.paths.base_dir
BEST_MODEL_DIR = BASE_DIR / "phase2_results-50pct" / "checkpoints" / "best_model_Config_B1"
LLM_DIR        = BEST_MODEL_DIR / "llm"
VIT_DIR        = BEST_MODEL_DIR / "vit"

TEST_CSV       = BASE_DIR / "test_split_50pct.csv"
IMAGES_DIR     = BASE_DIR / "imagenes_50_porciento"
CHEXPERT_CSV   = BASE_DIR / "mimic-cxr-2.0.0-chexpert.csv"

# Hiperparámetros de generación ESTRICTOS (solicitados)
GEN_PARAMS = dict(
    max_new_tokens     = 512,
    repetition_penalty = 1.2,
    num_beams          = 5,
    do_sample          = False,        # beam search determinista
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Carga del modelo dual-LoRA
# ─────────────────────────────────────────────────────────────────────────────
def cargar_modelo_dual_lora(best_model_dir: Path, device_map: str = "auto"):
    """
    Carga BLIP-2 base en bfloat16 e inyecta AMBOS adaptadores LoRA de Fase 2.

    - LLM  → PeftModel (se mantiene el wrapper; generate() lo usa internamente).
    - ViT  → merge_and_unload() (fusión OBLIGATORIA: generate() llama al ViT con
             pixel_values POSICIONAL, incompatible con el wrapper PeftModel; ver
             la explicación detallada en run_tournament.py).

    Returns
    -------
    (model, device) — modelo en eval() listo para generate().
    """
    from transformers import Blip2ForConditionalGeneration
    from peft import PeftModel

    llm_dir = Path(best_model_dir) / "llm"
    vit_dir = Path(best_model_dir) / "vit"
    for sub, lbl in [(llm_dir, "LLM"), (vit_dir, "ViT")]:
        if not sub.exists():
            raise FileNotFoundError(f"Adaptador {lbl} no encontrado: {sub}")

    logger.info("Cargando modelo base BLIP-2 (bfloat16)…")
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        config.model.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    logger.info("Inyectando LLM LoRA (is_trainable=False)…")
    base_model.language_model = PeftModel.from_pretrained(
        base_model.language_model, str(llm_dir), is_trainable=False
    )

    logger.info("Inyectando ViT LoRA y fusionando (merge_and_unload)…")
    vit_peft = PeftModel.from_pretrained(
        base_model.vision_model, str(vit_dir), is_trainable=False
    )
    # qformer nunca se toca → refleja el device/dtype reales del modelo base
    _ref = next(base_model.qformer.parameters())
    base_model.vision_model = vit_peft.merge_and_unload().to(
        device=_ref.device, dtype=_ref.dtype
    )

    base_model.eval()
    logger.info(f"✅ Modelo dual-LoRA en {_ref.device} | dtype={_ref.dtype}")
    return base_model, _ref.device


# ─────────────────────────────────────────────────────────────────────────────
# 5. Generación de reportes
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def generar_reporte(model, processor, test_dataset, idx: int, device) -> dict:
    """
    Genera el reporte para una fila del test set usando PREFIX FORCING.

    Se inyecta un encabezado administrativo FIJO como prefijo de decodificación:
    el modelo deja de alucinar cabeceras desde cero y continúa el texto a partir
    de "...FINDINGS: ", concentrándose en describir los hallazgos.

    Nota técnica (importante): en BLIP-2 (OPT) model.generate() devuelve SOLO los
    tokens NUEVOS (no incluye el prefijo). Por eso el prefijo se antepone a mano
    al texto decodificado para reconstruir el reporte completo y limpio.
    """
    row      = test_dataset.data.iloc[idx]
    dicom_id = str(row["dicom_id"])
    study_id = str(row.get("study_id", ""))
    ref_text = str(row["report_text"]).strip()

    # Vista: solo para metadato/etiquetado (la generación usa el prefijo fijo)
    view_raw = row.get("ViewPosition", None)
    if view_raw is not None and not (isinstance(view_raw, float) and math.isnan(view_raw)):
        vista = test_dataset._mapear_vista(str(view_raw))
    else:
        vista = "Unknown"

    # Preprocesado idéntico al de entrenamiento (CLAHE + resize + normalización).
    # imagen_pil ES la imagen ya estandarizada → equivalente a tu `raw_image`,
    # pero con CLAHE para no degradar el rendimiento respecto al entrenamiento.
    imagen_path = test_dataset._buscar_imagen(dicom_id)
    imagen_pil  = estandarizar_imagen(
        imagen_path,
        target_size          = config.data.image_size,
        clahe_clip_limit     = config.data.clahe_clip_limit,
        clahe_tile_grid_size = config.data.clahe_tile_grid_size,
    )

    # ── PREFIX FORCING ────────────────────────────────────────────────────────
    prefijo_forzado = "EXAMINATION: CHEST (PA AND LAT). TECHNIQUE: PA and lateral. FINDINGS: "

    inputs = processor(
        images=imagen_pil, text=prefijo_forzado, return_tensors="pt"
    ).to(device, torch.bfloat16)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens     = 300,
        repetition_penalty = 1.5,
        num_beams          = 3,
        early_stopping     = True,
    )

    # BLIP-2/OPT devuelve solo la continuación → anteponer el prefijo forzado
    continuacion = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    generado     = (prefijo_forzado + continuacion).strip()

    # ── Impresión limpia en terminal ──────────────────────────────────────────
    print("\n" + "─" * 80)
    print(f"Estudio idx={idx}  |  dicom_id={dicom_id}  |  vista={vista}")
    print("─" * 80)
    print(generado)
    print("─" * 80, flush=True)

    return {
        "dicom_id":   dicom_id,
        "study_id":   study_id,
        "vista":      vista,
        "imagen":     imagen_pil,
        "referencia": ref_text,
        "generado":   generado,
    }
# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualización comparativa (imagen + GT vs Generado) → PNG
# ─────────────────────────────────────────────────────────────────────────────
def _wrap(texto: str, w: int = 88) -> str:
    """Envuelve el texto respetando los saltos de línea originales."""
    lineas = texto.splitlines() or [texto]
    return "\n".join(textwrap.fill(l, w) if l.strip() else "" for l in lineas)


def guardar_comparativa(r: dict, idx_global: int, output_dir: Path) -> Path:
    """Una figura por estudio: radiografía (izq) + GT vs Generado (der)."""
    fig, (ax_img, ax_txt) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.7]}
    )

    ax_img.imshow(r["imagen"], cmap="gray")
    ax_img.set_title(f"Estudio {idx_global} · {r['vista']} view\n{r['dicom_id'][:18]}…",
                     fontsize=11, fontweight="bold")
    ax_img.axis("off")

    ax_txt.axis("off")
    bloque = (
        f"📋 GROUND TRUTH (Original)\n{'─' * 72}\n{_wrap(r['referencia'])}"
        f"\n\n\n🤖 GENERADO (beams=5 · rep_pen=1.2)\n{'─' * 72}\n{_wrap(r['generado'])}"
    )
    ax_txt.text(0.0, 0.99, bloque, va="top", ha="left",
                fontsize=8.5, family="monospace", transform=ax_txt.transAxes)

    plt.tight_layout()
    out_path = output_dir / f"comparativa_estudio_{idx_global:02d}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 7. Métricas de TEXTO (idénticas a evaluate_final.py)
# ─────────────────────────────────────────────────────────────────────────────
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer as _rouge_module

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

_rouge = _rouge_module.RougeScorer(["rougeL"], use_stemmer=True)


def bleu4(reference: str, hypothesis: str) -> float:
    ref = nltk.word_tokenize(reference.lower())
    hyp = nltk.word_tokenize(hypothesis.lower())
    return sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=SmoothingFunction().method1)


def rouge_l(reference: str, hypothesis: str) -> float:
    return _rouge.score(reference, hypothesis)["rougeL"].fmeasure


def cargar_radgraph():
    """Devuelve (f1rg, fn_score) o (None, None) si radgraph no está disponible."""
    try:
        from radgraph import F1RadGraph
        logger.info("Cargando F1RadGraph (reward_level='all', model_type='radgraph-xl')…")
        f1rg = F1RadGraph(reward_level="all", model_type="radgraph-xl")

        def _score(reference: str, hypothesis: str) -> float:
            try:
                mean_reward, _, _, _ = f1rg(refs=[reference], hyps=[hypothesis])
                if isinstance(mean_reward, (tuple, list)):
                    return float(mean_reward[2])     # RG_ERbar
                return float(mean_reward)
            except Exception as exc:
                logger.warning(f"  RadGraph error: {exc}")
                return float("nan")

        logger.info("  F1RadGraph listo.")
        return f1rg, _score
    except Exception as exc:
        logger.warning(f"  RadGraph no disponible ({exc}). Se omite F1-RadGraph.")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# 8. Evaluación CLÍNICA CheXpert (idéntica a evaluate_final.py · Fase 2)
# ─────────────────────────────────────────────────────────────────────────────
def _norm_sid(s):
    """Normaliza study_id ('s5069…' o 5069…) → '5069…'."""
    s = str(s).strip()
    return s.lstrip("s") if s.startswith("s") else s


def _to_bin(v):
    """1.0/-1.0 (incierto)=mencionado→1 ; 0.0→0 ; NaN→None (no mencionado)."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return 1 if float(v) != 0.0 else 0


def _chexpert_comparison_metrics(gt_row, pred_dict, chex_cols, pred_prefix) -> Dict[str, float]:
    """Compara GT (CSV oficial) vs Pred (CheXbert) → micro P/R/F1 + match por clase."""
    tp = fp = fn = 0
    m: Dict[str, float] = {}
    for lbl in chex_cols:
        gt_val   = gt_row.get(lbl, float("nan")) if gt_row is not None else float("nan")
        pred_val = pred_dict.get(f"{pred_prefix}{lbl.lower().replace(' ', '_')}", float("nan"))
        gt_bin, pred_bin = _to_bin(gt_val), _to_bin(pred_val)

        key = f"match_{lbl.lower().replace(' ', '_')}"
        if gt_bin is None and pred_bin is None:
            m[key] = float("nan")
            continue
        if gt_bin   is None: gt_bin   = 0      # imputar no-mencionado = negativo
        if pred_bin is None: pred_bin = 0
        m[key] = 1.0 if gt_bin == pred_bin else 0.0

        if   gt_bin == 1 and pred_bin == 1: tp += 1
        elif gt_bin == 0 and pred_bin == 1: fp += 1
        elif gt_bin == 1 and pred_bin == 0: fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if not (math.isnan(precision) or math.isnan(recall)) and (precision + recall) > 0
          else float("nan"))
    m["chexpert_precision"] = precision
    m["chexpert_recall"]    = recall
    m["chexpert_f1"]        = f1
    return m


def evaluar_chexpert(resultados: List[dict], output_dir: Path) -> None:
    """GT oficial vs CheXbert sobre reportes generados → CSVs de métricas clínicas."""
    from chexbert_official import CheXbertLabeler, CHEXBERT_LABELS

    chex_cols   = CHEXBERT_LABELS
    pred_prefix = "pred_chexpert_"

    logger.info("Cargando CheXbert oficial (Stanford AIMI)…")
    chexbert   = CheXbertLabeler()
    df_chex_gt = pd.read_csv(CHEXPERT_CSV)
    df_chex_gt["_sid"] = df_chex_gt["study_id"].astype(str).apply(_norm_sid)
    gt_by_sid = df_chex_gt.set_index("_sid")

    filas, n_sin_gt = [], 0
    match_accum = {l: [] for l in chex_cols}

    for i, r in enumerate(tqdm(resultados, desc="  CheXpert", unit="estudio"), 1):
        sid = _norm_sid(r["study_id"])
        if sid in gt_by_sid.index:
            gt_row = gt_by_sid.loc[sid]
            if isinstance(gt_row, pd.DataFrame):     # study_id duplicado → 1ª fila
                gt_row = gt_row.iloc[0]
        else:
            gt_row = None
            n_sin_gt += 1

        raw  = chexbert.label(r["generado"] or "no findings")
        pred = {f"{pred_prefix}{k.lower().replace(' ', '_')}": v for k, v in raw.items()}
        cmp  = _chexpert_comparison_metrics(gt_row, pred, chex_cols, pred_prefix)

        filas.append({
            "estudio":   i,
            "study_id":  r["study_id"],
            "precision": cmp["chexpert_precision"],
            "recall":    cmp["chexpert_recall"],
            "f1":        cmp["chexpert_f1"],
        })
        for l in chex_cols:
            v = cmp.get(f"match_{l.lower().replace(' ', '_')}", float("nan"))
            if not (isinstance(v, float) and math.isnan(v)):
                match_accum[l].append(v)

    df_chexpert = pd.DataFrame(filas)
    df_chexpert.to_csv(output_dir / "metricas_chexpert.csv", index=False)

    df_agreement = (pd.DataFrame({
        "clase":     list(match_accum.keys()),
        "agreement": [np.mean(v) if v else float("nan") for v in match_accum.values()],
        "n_eval":    [len(v) for v in match_accum.values()],
    }).sort_values("agreement", ascending=False).reset_index(drop=True))
    df_agreement.to_csv(output_dir / "agreement_chexpert_por_clase.csv", index=False)

    logger.info(f"\n{SEP}")
    logger.info("  MÉTRICAS CHEXPERT (micro, por estudio)")
    logger.info(f"{SEP}")
    logger.info(f"  Estudios sin GT (sin match por study_id): {n_sin_gt}/{len(resultados)}")
    for col in ("precision", "recall", "f1"):
        logger.info(f"  {col:<10}: {df_chexpert[col].mean(skipna=True):.4f}")
    logger.info("  Agreement por clase → agreement_chexpert_por_clase.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Métricas de texto sobre todos los reportes
# ─────────────────────────────────────────────────────────────────────────────
def evaluar_texto(resultados: List[dict], output_dir: Path,
                  radgraph_score) -> pd.DataFrame:
    filas = []
    for i, r in enumerate(tqdm(resultados, desc="  Métricas texto", unit="estudio"), 1):
        ref = r["referencia"] or "no findings"
        hyp = r["generado"]   or "no findings"
        fila = {
            "estudio":  i,
            "dicom_id": r["dicom_id"],
            "study_id": r["study_id"],
            "BLEU-4":   bleu4(ref, hyp),
            "ROUGE-L":  rouge_l(ref, hyp),
        }
        if radgraph_score is not None:
            fila["F1-RadGraph"] = radgraph_score(ref, hyp)
        filas.append(fila)

    df = pd.DataFrame(filas)
    df.to_csv(output_dir / "metricas_texto.csv", index=False)

    logger.info(f"\n{SEP}")
    logger.info("  MÉTRICAS DE TEXTO (promedios)")
    logger.info(f"{SEP}")
    for col in ("BLEU-4", "ROUGE-L", "F1-RadGraph"):
        if col in df.columns:
            logger.info(f"  {col:<12}: {df[col].mean(skipna=True):.4f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inferencia profunda BLIP-2 dual-LoRA (Fase 2 · 50 %).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-samples", type=int, default=20, help="Estudios aleatorios a evaluar.")
    p.add_argument("--seed",        type=int, default=42, help="Semilla de selección aleatoria.")
    p.add_argument("--output-dir",  type=str,
                   default=str(BASE_DIR / "evaluacion_profunda_results"),
                   help="Directorio de salida (CSVs + PNGs).")
    p.add_argument("--no-plots",        action="store_true", help="No generar PNGs comparativos.")
    p.add_argument("--skip-radgraph",   action="store_true", help="Omitir F1-RadGraph.")
    p.add_argument("--skip-chexpert",   action="store_true", help="Omitir evaluación CheXpert.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"\n{SEP}")
    logger.info("  INFERENCIA PROFUNDA — BLIP-2 dual-LoRA (Config_B1 · 50 %)")
    logger.info(f"  best_model : {BEST_MODEL_DIR}")
    logger.info(f"  test_csv   : {TEST_CSV.name}")
    logger.info(f"  device     : {device_str}  | num_samples={args.num_samples}  seed={args.seed}")
    logger.info(f"  output_dir : {output_dir}")
    logger.info(f"{SEP}\n")

    # ── Modelo + procesador ───────────────────────────────────────────────────
    processor = cargar_procesador(model_name=config.model.model_name)
    model, device = cargar_modelo_dual_lora(BEST_MODEL_DIR, device_map="auto")

    # ── Test dataset ──────────────────────────────────────────────────────────
    logger.info("Cargando Test Set (50 %)…")
    test_dataset = MimicCXRDataset(
        csv_path             = TEST_CSV,
        images_dir           = IMAGES_DIR,
        processor            = processor,
        image_size           = config.data.image_size,
        clahe_clip_limit     = config.data.clahe_clip_limit,
        clahe_tile_grid_size = config.data.clahe_tile_grid_size,
        max_length           = config.data.max_length,
        padding              = config.data.padding,
    )

    n = min(args.num_samples, len(test_dataset))
    indices = random.sample(range(len(test_dataset)), n)
    logger.info(f"Test set: {len(test_dataset):,} estudios | seleccionados {n}: {indices}\n")

    # ── Generación ────────────────────────────────────────────────────────────
    resultados = []
    for idx in tqdm(indices, desc="  Generando reportes", unit="estudio"):
        resultados.append(generar_reporte(model, processor, test_dataset, idx, device))

    # Guardar predicciones crudas
    pd.DataFrame([{
        "dicom_id":         r["dicom_id"],
        "study_id":         r["study_id"],
        "vista":            r["vista"],
        "reference_report": r["referencia"],
        "generated_report": r["generado"],
    } for r in resultados]).to_csv(output_dir / "predicciones_inferencia.csv", index=False)
    logger.info(f"✅ {len(resultados)} reportes generados → predicciones_inferencia.csv")

    # ── Visualización ─────────────────────────────────────────────────────────
    if not args.no_plots:
        logger.info("Guardando comparativas visuales (PNG)…")
        for i, r in enumerate(resultados, 1):
            guardar_comparativa(r, i, output_dir)
        logger.info(f"  {len(resultados)} figuras → comparativa_estudio_XX.png")

    # ── Métricas de texto ─────────────────────────────────────────────────────
    radgraph_score = None
    if not args.skip_radgraph:
        _, radgraph_score = cargar_radgraph()
    evaluar_texto(resultados, output_dir, radgraph_score)

    # Liberar el F1RadGraph antes de CheXbert (ambos cargan modelos en VRAM)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Evaluación CheXpert ───────────────────────────────────────────────────
    if not args.skip_chexpert:
        if CHEXPERT_CSV.exists():
            evaluar_chexpert(resultados, output_dir)
        else:
            logger.warning(f"  CSV CheXpert no encontrado: {CHEXPERT_CSV} — se omite.")

    logger.info(f"\n{SEP}")
    logger.info(f"  FINALIZADO. Resultados en: {output_dir}")
    logger.info(f"{SEP}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
