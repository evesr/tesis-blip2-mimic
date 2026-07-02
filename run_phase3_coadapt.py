#!/usr/bin/env python3
"""
run_phase3_coadapt.py — Fase 3 Co-adaptación: Q-Former LoRA r=128 + ViT layers 37-38
======================================================================================
Estudio de Ablación — Paso 2: Co-adaptación de Espacios Latentes (10% MIMIC-CXR).

Configuración Única
--------------------
  Config_3B_Coadapt : Base B (LLM r=32 + ViT r=8) + Q-Former LoRA (r=128, α=256)
                      + ViT encoder.layers.37 y encoder.layers.38 DESCONGELADOS

Motivación: En el experimento Gigante (r=128 con ViT totalmente congelado) se
detectó una "brecha modal" entre el extractor visual y el Q-Former. Al descongelar
quirúrgicamente las dos últimas capas del ViT, el extractor puede co-adaptarse
al espacio de representación del Q-Former, reduciendo el gap de distribución.

Por cada corrida este script:
  1. Llama a trainer_utils_phase3_coadapt.train_single_config_phase3() → entrenamiento.
  2. Guarda adaptador Q-Former en phase3_results-coadapt/checkpoints/best_model_{name}/qformer/.
  3. Guarda pesos ViT co-adaptados en best_model_{name}/vit_coadapted.pt.
  4. Llama a fase1_inferencia_phase3() → genera predicciones_{name}.csv en el
     Test Set del 10% cargando LLM + ViT (con pesos co-adaptados) + Q-Former.
  5. Llama a evaluate_final.fase2_evaluacion() → métricas granulares fila por fila
     (BLEU-4, ROUGE-L, F1-RadGraph, CheXpert Precision/Recall/F1).
  6. Persiste resultados incrementales por si el proceso se interrumpe.

Al final:
  - Ranking final ordenado por best_val_loss.
  - Tabla de métricas globales comparativas.
  - 3 PNGs de curvas comparativas (Val Loss / BLEU-4 / ROUGE-L).

Uso
---
    python run_phase3_coadapt.py
    python run_phase3_coadapt.py --skip-train              # solo inferencia + eval
    python run_phase3_coadapt.py --skip-inference          # reutiliza predicciones existentes
    python run_phase3_coadapt.py --configs Config_3B_Coadapt
    python run_phase3_coadapt.py --num-samples 500         # evalúa solo N muestras del test set

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Patch de compatibilidad tokenizer (debe ir ANTES de radgraph)
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
# 1.  Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import gc
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

# cv2 se desactiva en trainer_utils ya al importarlo; re-confirmamos aquí
import cv2
cv2.setNumThreads(0)

# ── Módulos del proyecto ──────────────────────────────────────────────────────
from config import config
from model_utils import cargar_procesador
from data_loader import MimicCXRDataset
from preprocessing import estandarizar_imagen

# ── Utilidades de Fase 3 Co-adaptación ───────────────────────────────────────
from trainer_utils_phase3_coadapt import (
    OUTPUT_DIR,
    CHECKPOINTS_DIR,
    TEST_CSV_10PCT,
    IMAGES_DIR_10PCT,
    DUAL_WEIGHTS,
    TRAINING_CONFIG,
    load_phase3_datasets,
    plot_comparative_curves,
    train_single_config_phase3,
)

# ── Pipeline de evaluación reutilizado de evaluate_final.py ──────────────────
from evaluate_final import (
    fase2_evaluacion,
    calcular_resumen_global,
    GEN_PARAMS,
)
from radgraph import F1RadGraph
from chexbert_official import CheXbertLabeler

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DEFINICIÓN DE LA CONFIGURACIÓN DE CO-ADAPTACIÓN
# ─────────────────────────────────────────────────────────────────────────────
#
# base_weights_key: "B" → best_model_Config_B1 (LLM r=32 + ViT r=8 de Fase 2)
#
# Configuración única: Q-Former LoRA r=128 + ViT layers 37-38 co-adaptados
#
TOURNAMENT_CONFIGS: List[Dict] = [
    {
        "name":             "Config_3B_Coadapt",
        "qformer_r":        128,
        "qformer_alpha":    256,
        "base_weights_key": "B",
    },
]

# Directorio donde se guardan las predicciones y los resultados de evaluación
EVAL_OUTPUT_DIR = OUTPUT_DIR / "evaluation"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  INFERENCIA CO-ADAPTACIÓN: LLM + ViT (fusionado + pesos co-adaptados) + Q-Former
# ─────────────────────────────────────────────────────────────────────────────

def fase1_inferencia_phase3(
    config_name:  str,
    model_dir:    Path,
    base_key:     str,
    output_dir:   Path,
    processor,
    num_samples:  Optional[int] = None,
) -> Path:
    """
    Genera reportes para el Test Set del 10% y los exporta a
    predicciones_{config_name}.csv.

    Diferencia respecto a Gigante
    ------------------------------
    Además de los tres adaptadores estándar (LLM LoRA, ViT LoRA, Q-Former LoRA),
    carga los pesos co-adaptados del ViT guardados en vit_coadapted.pt y los
    aplica sobre el vision_model fusionado antes de la inferencia.

    Estructura del checkpoint esperada:
        DUAL_WEIGHTS[base_key]["llm"]   ← LLM LoRA (Fase 2)
        DUAL_WEIGHTS[base_key]["vit"]   ← ViT LoRA (Fase 2, base para co-adapt)
        model_dir/qformer/              ← Q-Former LoRA (Co-adaptación)
        model_dir/vit_coadapted.pt      ← Pesos ViT layers 37-38 co-adaptados

    Args:
        config_name  : Nombre de la config (p. ej. "Config_3B_Coadapt").
        model_dir    : Ruta a phase3_results-coadapt/checkpoints/best_model_{name}/.
        base_key     : "A" o "B" (indica qué pesos duales de Fase 2 usar).
        output_dir   : Directorio destino del CSV de predicciones.
        processor    : Blip2Processor ya instanciado.
        num_samples  : Limitar a N muestras del test set (None = todo).

    Returns:
        Path al CSV  predicciones_{config_name}.csv.

    Raises:
        FileNotFoundError : Si model_dir o subdirectorios no existen.
    """
    llm_dir          = DUAL_WEIGHTS[base_key]["llm"]
    vit_dir          = DUAL_WEIGHTS[base_key]["vit"]
    qformer_dir      = model_dir / "qformer"
    vit_coadapted_pt = model_dir / "vit_coadapted.pt"

    for subdir, label in [
        (llm_dir,     "LLM"),
        (vit_dir,     "ViT (base Fase 2)"),
        (qformer_dir, "Q-Former"),
    ]:
        if not subdir.exists():
            raise FileNotFoundError(
                f"Adaptador {label} no encontrado: {subdir}\n"
                "El entrenamiento debe haber concluido antes de llamar a la inferencia."
            )

    if not vit_coadapted_pt.exists():
        raise FileNotFoundError(
            f"Pesos co-adaptados del ViT no encontrados: {vit_coadapted_pt}\n"
            "El entrenamiento de co-adaptación debe haber guardado vit_coadapted.pt."
        )

    logger.info(f"\n{'='*65}")
    logger.info(f"  INFERENCIA FASE CO-ADAPTACIÓN (10%): {config_name}")
    logger.info(f"  Base: {base_key}")
    logger.info(f"  LLM          ← {llm_dir}")
    logger.info(f"  ViT base     ← {vit_dir}")
    logger.info(f"  Q-Former     ← {qformer_dir}")
    logger.info(f"  ViT coadapt  ← {vit_coadapted_pt}")
    logger.info(f"{'='*65}")

    # ── Cargar modelo base + inyectar LLM + ViT ──────────────────────────────
    logger.info("  Cargando modelo base BLIP-2 (bfloat16)...")
    from transformers import Blip2ForConditionalGeneration
    from peft import PeftModel

    base_model = Blip2ForConditionalGeneration.from_pretrained(
        config.model.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info("  Cargando LLM LoRA y fusionando pesos (merge_and_unload)...")
    llm_peft = PeftModel.from_pretrained(
        base_model.language_model,
        str(llm_dir),
        is_trainable=False,
    )
    base_model.language_model = llm_peft.merge_and_unload()

    logger.info("  Cargando ViT LoRA (Fase 2) y fusionando pesos (merge_and_unload)...")
    vit_peft = PeftModel.from_pretrained(
        base_model.vision_model,
        str(vit_dir),
        is_trainable=False,
    )
    _ref = next(base_model.qformer.parameters())   # referencia de device/dtype
    base_model.vision_model = vit_peft.merge_and_unload().to(
        device=_ref.device, dtype=_ref.dtype
    )

    # ── Aplicar pesos co-adaptados del ViT (layers 37-38) ────────────────────
    logger.info("  Aplicando pesos co-adaptados del ViT (layers 37-38)...")
    vit_coadapted_state = torch.load(str(vit_coadapted_pt), map_location=_ref.device)
    # Cargar en el vision_model fusionado (solo las capas 37-38)
    missing, unexpected = base_model.vision_model.load_state_dict(
        vit_coadapted_state, strict=False
    )
    if unexpected:
        logger.warning(f"  ViT co-adapt — tensores inesperados: {unexpected}")
    logger.info(
        f"  ViT co-adapt cargado: {len(vit_coadapted_state)} tensores aplicados "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    # Asegurar dtype/device correcto tras la carga
    base_model.vision_model = base_model.vision_model.to(
        device=_ref.device, dtype=_ref.dtype
    )

    # ── Cargar Q-Former LoRA y fusionar ──────────────────────────────────────
    logger.info("  Cargando Q-Former LoRA y fusionando pesos (merge_and_unload)...")
    qformer_peft = PeftModel.from_pretrained(
        base_model.qformer,
        str(qformer_dir),
        is_trainable=False,
    )
    base_model.qformer = qformer_peft.merge_and_unload().to(
        device=_ref.device, dtype=_ref.dtype
    )
    logger.info(
        f"  Todos los adaptadores fusionados → device={_ref.device}, dtype={_ref.dtype}"
    )

    model = base_model
    model.eval()
    device = _ref.device
    logger.info(f"  Modelo en {device} | dtype={_ref.dtype}")

    # ── Test Dataset ──────────────────────────────────────────────────────────
    logger.info("  Cargando Test Set (10%)...")
    test_dataset = MimicCXRDataset(
        csv_path             = TEST_CSV_10PCT,
        images_dir           = IMAGES_DIR_10PCT,
        processor            = processor,
        image_size           = config.data.image_size,
        clahe_clip_limit     = config.data.clahe_clip_limit,
        clahe_tile_grid_size = config.data.clahe_tile_grid_size,
        max_length           = config.data.max_length,
        padding              = config.data.padding,
    )

    n_total = len(test_dataset)
    n_eval  = n_total if num_samples is None else min(num_samples, n_total)
    logger.info(f"  Test Set: {n_total:,} muestras | A evaluar: {n_eval:,}")

    # ── CSV incremental ───────────────────────────────────────────────────────
    COLS       = ["dicom_id", "study_id", "reference_report", "generated_report"]
    preds_path = output_dir / f"predicciones_{config_name}.csv"
    pd.DataFrame(columns=COLS).to_csv(preds_path, index=False)

    BATCH_SIZE    = 4
    batch_images  = []
    batch_prompts = []
    batch_meta    = []
    errores       = 0
    n_guardadas   = 0

    for idx in tqdm(range(n_eval), desc=f"  Inferencia {config_name}", unit="img"):
        row      = test_dataset.data.iloc[idx]
        dicom_id = str(row["dicom_id"])
        study_id = str(row.get("study_id", ""))
        ref_text = str(row["report_text"]).strip()

        # Prompt con metadato de vista
        view_raw = row.get("ViewPosition", None)
        if view_raw is not None and not (
            isinstance(view_raw, float) and math.isnan(view_raw)
        ):
            vista = test_dataset._mapear_vista(str(view_raw))
        else:
            vista = "Unknown"
        prompt = f"[Context: {vista} view] {config.inference.default_prompt}"

        # Buscar y preprocesar imagen
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
        batch_meta.append({"dicom_id": dicom_id, "study_id": study_id,
                            "ref_text": ref_text})

        # Vaciar lote cuando esté lleno o sea la última iteración válida
        if len(batch_images) == BATCH_SIZE or idx == n_eval - 1:
            if not batch_images:
                continue
            try:
                inputs = processor(
                    images=batch_images,
                    text=batch_prompts,
                    return_tensors="pt",
                    padding=True,
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
                    preds_path, mode="a", header=False, index=False
                )
                n_guardadas += len(rows)

            except Exception as exc:
                logger.exception(f"  Error en lote idx={idx}: {exc}")
                errores += len(batch_images)

            batch_images.clear()
            batch_prompts.clear()
            batch_meta.clear()

    logger.info(
        f"  Inferencia OK: {n_guardadas} reportes guardados | {errores} errores"
    )
    logger.info(f"  → {preds_path}")

    # Liberar VRAM
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("  VRAM liberada tras inferencia")

    return preds_path


# ─────────────────────────────────────────────────────────────────────────────
# 5.  BUCLE PRINCIPAL DEL TORNEO
# ─────────────────────────────────────────────────────────────────────────────

def run_tournament(
    configs_to_run:  List[Dict],
    skip_train:      bool,
    skip_inference:  bool,
    num_samples:     Optional[int],
    chexpert_csv:    Optional[Path],
) -> None:
    """
    Itera sobre las configuraciones de co-adaptación en secuencia.

    Por cada config:
        [Fase Train]  train_single_config_phase3()
        [Fase Inf]    fase1_inferencia_phase3()   (salvo --skip-inference)
        [Fase Eval]   evaluate_final.fase2_evaluacion()

    Al finalizar todas las configs:
        - Ranking final (CSV)
        - Resumen global de métricas (CSV)
        - 3 PNGs comparativos de curvas
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pre-cargar modelos de evaluación UNA sola vez ────────────────────────
    logger.info("\nCargando F1RadGraph (reward_level='all', model_type='radgraph-xl')...")
    f1rg = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    logger.info("  F1RadGraph listo")

    logger.info("Cargando CheXbert oficial (Stanford AIMI)...")
    chexbert = CheXbertLabeler()
    logger.info("  CheXbert listo\n")

    # ── Procesador BLIP-2 ─────────────────────────────────────────────────────
    logger.info("Cargando Blip2Processor...")
    processor = cargar_procesador(model_name=config.model.model_name)
    logger.info("  Processor listo\n")

    # ── Datasets de entrenamiento / validación ───────────────────────────────
    logger.info("Cargando datasets del 10% (train + val)...")
    train_dataset, val_dataset = load_phase3_datasets(processor)

    # ── Contenedores de resultados ────────────────────────────────────────────
    train_results:  List[Dict]              = []
    granular_dfs:   Dict[str, pd.DataFrame] = {}

    total = len(configs_to_run)

    for i, cfg in enumerate(configs_to_run, 1):
        name = cfg["name"]
        logger.info(f"\n{'#'*70}")
        logger.info(f"  TORNEO CO-ADAPTACIÓN (10%) — Config {i}/{total}: {name}")
        logger.info(f"{'#'*70}")

        # ── FASE TRAIN ────────────────────────────────────────────────────────
        if not skip_train:
            try:
                result = train_single_config_phase3(
                    config_dict   = cfg,
                    config_num    = i,
                    total_configs = total,
                    train_dataset = train_dataset,
                    val_dataset   = val_dataset,
                    processor     = processor,
                )
                train_results.append(result)
            except Exception as exc:
                import traceback
                logger.error(f"\nERROR durante entrenamiento de {name}: {exc}")
                traceback.print_exc()
                logger.warning(f"  Saltando {name} y continuando...")
                continue

            # Guardar resultados parciales de entrenamiento
            pd.DataFrame(train_results).to_csv(
                OUTPUT_DIR / "intermediate_train_results.csv", index=False
            )
        else:
            # Modo --skip-train: verificar que el checkpoint existe
            model_dir_check = CHECKPOINTS_DIR / f"best_model_{name}"
            if not model_dir_check.exists():
                logger.warning(
                    f"  [skip-train] Checkpoint no encontrado: {model_dir_check}\n"
                    f"  Saltando {name} (entrenamiento aún no ha concluido)."
                )
                continue
            logger.info(f"  [skip-train] Checkpoint encontrado: {model_dir_check}")

        # ── FASE INFERENCIA ───────────────────────────────────────────────────
        preds_path = EVAL_OUTPUT_DIR / f"predicciones_{name}.csv"
        model_dir  = CHECKPOINTS_DIR / f"best_model_{name}"

        if skip_inference and preds_path.exists():
            logger.info(f"  [Inferencia] --skip-inference: reutilizando {preds_path.name}")
        else:
            if not model_dir.exists():
                logger.warning(
                    f"  [Inferencia] Directorio de checkpoint no encontrado: {model_dir}\n"
                    f"  Saltando inferencia y evaluación para {name}."
                )
                continue
            try:
                preds_path = fase1_inferencia_phase3(
                    config_name = name,
                    model_dir   = model_dir,
                    base_key    = cfg["base_weights_key"],
                    output_dir  = EVAL_OUTPUT_DIR,
                    processor   = processor,
                    num_samples = num_samples,
                )
            except Exception as exc:
                import traceback
                logger.error(f"\nERROR durante inferencia de {name}: {exc}")
                traceback.print_exc()
                logger.warning(f"  Saltando evaluación de {name}...")
                continue

        # ── FASE EVALUACIÓN ───────────────────────────────────────────────────
        try:
            df_gran = fase2_evaluacion(
                preds_path   = preds_path,
                cfg_name     = name,
                output_dir   = EVAL_OUTPUT_DIR,
                f1rg         = f1rg,
                chexbert     = chexbert,
                chexpert_csv = chexpert_csv,
            )
            granular_dfs[name] = df_gran
        except Exception as exc:
            import traceback
            logger.error(f"\nERROR durante evaluación de {name}: {exc}")
            traceback.print_exc()

        # Guardar resumen parcial de métricas tras cada config
        if granular_dfs:
            _df_partial = calcular_resumen_global(granular_dfs, EVAL_OUTPUT_DIR)
            logger.info(f"  Resumen parcial de métricas guardado.")

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTADOS FINALES
    # ─────────────────────────────────────────────────────────────────────────
    # En modo --skip-train, intentar recuperar ranking del CSV incremental
    if not train_results:
        csv_incremental = OUTPUT_DIR / "intermediate_train_results.csv"
        if csv_incremental.exists():
            logger.info(f"  Recuperando resultados de entrenamiento desde {csv_incremental.name}")
            train_results = pd.read_csv(csv_incremental).to_dict("records")
        else:
            logger.warning("No hay resultados de entrenamiento disponibles (ranking omitido).")

    # Ranking final por best_val_loss
    df_train = pd.DataFrame(train_results).sort_values("best_val_loss") if train_results else pd.DataFrame()
    if not df_train.empty:
        df_train.to_csv(OUTPUT_DIR / "tournament_train_ranked.csv", index=False)
        print(f"\n{'='*70}")
        print("  RANKING FINAL — Val Loss (menor es mejor)")
        print(f"{'='*70}")
        show_cols = ["config_name", "qformer_r", "qformer_alpha", "base", "best_val_loss",
                     "best_step", "total_steps", "final_train_loss"]
        print(df_train[[c for c in show_cols if c in df_train.columns]].to_string(index=False))
        print(f"{'='*70}\n")

    # Tabla de métricas globales
    if granular_dfs:
        df_summary = calcular_resumen_global(granular_dfs, EVAL_OUTPUT_DIR)
        show_metric_cols = [
            "config", "n_samples",
            "bleu_4", "rouge_l", "radgraph_f1",
            "chexpert_precision", "chexpert_recall", "chexpert_f1",
        ]
        show_metric_cols = [c for c in show_metric_cols if c in df_summary.columns]
        print(f"{'='*70}")
        print("  MÉTRICAS GLOBALES DE EVALUACIÓN — Co-adaptación")
        print(f"{'='*70}")
        print(df_summary[show_metric_cols].to_string(index=False))
        print(f"{'='*70}\n")

    # Gráficos comparativos de curvas (Val Loss / BLEU-4 / ROUGE-L)
    plot_comparative_curves(
        config_names = [c["name"] for c in configs_to_run],
        output_dir   = OUTPUT_DIR,
        plots_dir    = OUTPUT_DIR / "plots",
    )

    logger.info(f"\nFase 3 Co-adaptación completada. Resultados en: {OUTPUT_DIR}")
    logger.info(f"  tournament_train_ranked.csv")
    logger.info(f"  evaluation/metricas_globales_comparacion.csv")
    logger.info(f"  evaluation/resultados_granulares_{{name}}.csv  (x{len(granular_dfs)})")
    logger.info(f"  plots/01_val_loss_phase3_coadapt.png  /  02_bleu4_phase3_coadapt.png  /  03_rougeL_phase3_coadapt.png\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    valid_names = [c["name"] for c in TOURNAMENT_CONFIGS]
    parser = argparse.ArgumentParser(
        description=(
            "Fase 3 Co-adaptación — Q-Former LoRA r=128 + ViT layers 37-38 "
            "descongelados (Estudio de Ablación: Co-adaptación de Espacios Latentes)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=valid_names,
        default=valid_names,
        metavar="CONFIG",
        help=f"Configs a ejecutar. Opciones: {valid_names}.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help=(
            "Saltar entrenamiento y pasar directamente a inferencia + evaluación "
            "usando los checkpoints ya guardados en "
            "phase3_results-coadapt/checkpoints/best_model_{name}/. "
            "Úsalo cuando el entrenamiento ya concluyó."
        ),
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help=(
            "Saltar Fase 1 (inferencia) y reutilizar predicciones_{name}.csv "
            "existentes en phase3_results-coadapt/evaluation/."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        metavar="N",
        help="Limitar evaluación a N muestras del Test Set (None = todo).",
    )
    parser.add_argument(
        "--chexpert-csv",
        type=str,
        default=str(config.paths.base_dir / "mimic-cxr-2.0.0-chexpert.csv.gz"),
        metavar="PATH",
        help=(
            "Ruta al CSV oficial de etiquetas CheXpert de MIMIC-CXR "
            "(.csv o .csv.gz). Si no existe se omite la comparación GT."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    # Filtrar solo las configs seleccionadas, preservando el orden del torneo
    configs_to_run = [c for c in TOURNAMENT_CONFIGS if c["name"] in args.configs]

    chexpert_csv = Path(args.chexpert_csv) if args.chexpert_csv else None

    print("\n" + "="*80)
    print("  FASE 3 CO-ADAPTACIÓN — ESTUDIO DE ABLACIÓN: PASO 2")
    print("  Q-Former LoRA r=128, α=256 + ViT layers 37-38 co-adaptados")
    print("="*80)
    print(f"  Configuraciones: {len(configs_to_run)}")
    print(f"  Dataset: 10% MIMIC-CXR")
    print(f"  Scheduler: cosine | warmup_steps: {TRAINING_CONFIG['warmup_steps']}")
    print(f"  ViT co-adaptación: encoder.layers.37 + encoder.layers.38 DESCONGELADOS")
    print(f"  skip_text_generation=True (ahorro ~70% en eval)")
    print(f"  Resultados: {OUTPUT_DIR}")
    print("="*80 + "\n")

    try:
        run_tournament(
            configs_to_run = configs_to_run,
            skip_train     = args.skip_train,
            skip_inference = args.skip_inference,
            num_samples    = args.num_samples,
            chexpert_csv   = chexpert_csv,
        )
        return 0
    except KeyboardInterrupt:
        logger.warning("\n\nInterrupción manual detectada. Guardando progreso...")
        return 130
    except Exception as exc:
        import traceback
        logger.error(f"\n\nError fatal en el torneo: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
