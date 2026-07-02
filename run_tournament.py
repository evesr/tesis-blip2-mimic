#!/usr/bin/env python3
"""
run_tournament.py — Ejecutable principal del Torneo Fase 2.2
==================================================================
Escalamiento de finalistas: Config_A2 y Config_B1 al 10% de MIMIC-CXR (L40S, 48 GB).

Finalistas seleccionados (2 corridas secuenciales)
---------------------------------------------------
  Config_A2 : LLM r=16 (pesos Fase 1)  +  ViT r=16 (nuevo)
  Config_B1 : LLM r=32 (pesos Fase 1)  +  ViT r=8  (nuevo)

Por cada corrida este script:
  1. Llama a trainer_utils.train_single_config_phase2() → entrenamiento aislado.
  2. Guarda los pesos duales (llm/ y vit/) en phase2_results/checkpoints/best_model_{name}/.
  3. Llama a fase1_inferencia_phase2() → genera predicciones_{name}.csv en el
     Test Set del 10% cargando AMBOS adaptadores.
  4. Llama a evaluate_final.fase2_evaluacion() → métricas granulares fila por fila
     (BLEU-4, ROUGE-L, F1-RadGraph, CheXpert Precision/Recall/F1).
  5. Persiste resultados incrementales por si el proceso se interrumpe.

Al final:
  - Ranking final ordenado por best_val_loss.
  - Tabla de métricas globales comparativas.
  - 3 PNGs de curvas comparativas (Val Loss / BLEU-4 / ROUGE-L).

Optimizaciones Fase 2.2:
  - skip_text_generation=True en Phase2AuditCallback (ahorra ~70% tiempo en eval)
  - Dataset 10x más grande que Fase 2.1 (entrenamiento más robusto)

Uso
---
    python run_tournament.py
    python run_tournament.py --skip-train              # solo inferencia + eval (checkpoints ya listos)
    python run_tournament.py --skip-inference          # reutiliza predicciones existentes
    python run_tournament.py --configs Config_A2       # procesa solo Config_A2
    python run_tournament.py --num-samples 500         # evalúa sólo N muestras del test set

Autor : Evelyn Silva Rozas
Fecha : Mayo 2026 (actualizado para Fase 2.2)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Patch de compatibilidad tokenizer (debe ir ANTES de cualquier import
#     de radgraph, igual que en evaluate_final.py)
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

# ── Utilidades de Fase 2.2 (trainer_utils.py) ────────────────────────────────
from trainer_utils import (
    OUTPUT_DIR,
    CHECKPOINTS_DIR,
    TEST_CSV_10PCT,
    IMAGES_DIR_10PCT,
    TRAINING_CONFIG,
    load_phase2_datasets,
    plot_comparative_curves,
    train_single_config_phase2,
)

# ── Pipeline de evaluación reutilizado de evaluate_final.py ──────────────────
# Importamos SOLO las funciones puras de evaluación (Fase 2 del pipeline
# original).  La Fase 1 (inferencia) la reimplementamos aquí porque los
# checkpoints de Fase 2 tienen una estructura dual (llm/ + vit/) distinta a la
# de Fase 1, y evaluate_final.fase1_inferencia() solo carga LLM LoRA.
from evaluate_final import (
    fase2_evaluacion,
    calcular_resumen_global,
    GEN_PARAMS,
)
from radgraph import F1RadGraph
from chexbert_official import CheXbertLabeler

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Logging — consola + fichero (phase2_results-10pct/tournament_Config_B1.log)
# ─────────────────────────────────────────────────────────────────────────────
_LOG_FMT  = "%(asctime)s  %(levelname)-8s  %(message)s"
_LOG_DATE = "%H:%M:%S"

# Handler de consola (visible en tmux en tiempo real)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATE))

# Handler de fichero — se crea en cuanto OUTPUT_DIR esté disponible (post-import)
# Se inicializa en _setup_file_logger() llamado desde main().
_file_handler: Optional[logging.FileHandler] = None

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FMT,
    datefmt=_LOG_DATE,
    handlers=[_console_handler],
)
logger = logging.getLogger(__name__)


def _setup_file_logger(log_path: Path) -> None:
    """Añade un FileHandler al root logger para duplicar todo al fichero."""
    global _file_handler
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATE))
    logging.getLogger().addHandler(_file_handler)
    logger.info(f"Log guardado en: {log_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DEFINICIÓN DE LAS 4 CONFIGURACIONES DEL TORNEO
# ─────────────────────────────────────────────────────────────────────────────
#
# llm_weights_key: "r16" → best_model_Config_A (ganador Fase 1)
#                  "r32" → best_model_Config_B
#
# FASE 2.2 — FINALISTAS: Solo Config_A2 y Config_B1 (escalamiento al 10%)
TOURNAMENT_CONFIGS: List[Dict] = [
    
    {
        "name":            "Config_B1",
        "llm_r":           32,
        "vit_r":           8,
        "llm_weights_key": "r32",
    },
]

# Directorio donde se guardan las predicciones y los resultados de evaluación
EVAL_OUTPUT_DIR = OUTPUT_DIR / "evaluation"

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FASE 1 PHASE-2-AWARE: Inferencia con adaptadores duales (LLM + ViT)
# ─────────────────────────────────────────────────────────────────────────────

def fase1_inferencia_phase2(
    config_name:  str,
    model_dir:    Path,
    output_dir:   Path,
    processor,
    num_samples:  Optional[int] = None,
) -> Path:
    """
    Genera reportes para el Test Set del 10 % y los exporta a
    predicciones_{config_name}.csv.

    Diferencia respecto a evaluate_final.fase1_inferencia()
    --------------------------------------------------------
    Los checkpoints de Fase 2.1 guardan adaptadores en dos subdirectorios:
        model_dir/llm/   ← LLM LoRA (OPT)
        model_dir/vit/   ← ViT LoRA (Blip2VisionModel)

    Esta función carga AMBOS antes de lanzar la inferencia.

    Args:
        config_name  : Nombre de la config (p. ej. "Config_A1").
        model_dir    : Ruta a phase2_results/checkpoints/best_model_{name}/.
        output_dir   : Directorio destino del CSV de predicciones.
        processor    : Blip2Processor ya instanciado.
        num_samples  : Limitar a N muestras del test set (None = todo).

    Returns:
        Path al CSV  predicciones_{config_name}.csv.

    Raises:
        FileNotFoundError : Si model_dir o sus subdirectorios no existen.
    """
    llm_dir = model_dir / "llm"
    vit_dir = model_dir / "vit"

    for subdir, label in [(llm_dir, "LLM"), (vit_dir, "ViT")]:
        if not subdir.exists():
            raise FileNotFoundError(
                f"Adaptador {label} no encontrado: {subdir}\n"
                "El entrenamiento debe haber concluido antes de llamar a la inferencia."
            )

    logger.info(f"\n{'='*65}")
    logger.info(f"  INFERENCIA PHASE-2: {config_name}")
    logger.info(f"  LLM ← {llm_dir}")
    logger.info(f"  ViT ← {vit_dir}")
    logger.info(f"{'='*65}")

    # ── Cargar modelo base + inyectar AMBOS adaptadores ──────────────────────
    logger.info("  Cargando modelo base BLIP-2 (bfloat16)...")
    from transformers import Blip2ForConditionalGeneration
    from peft import PeftModel

    base_model = Blip2ForConditionalGeneration.from_pretrained(
        config.model.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info("  Cargando LLM LoRA (inferencia, is_trainable=False)...")
    base_model.language_model = PeftModel.from_pretrained(
        base_model.language_model,
        str(llm_dir),
        is_trainable=False,
    )

    logger.info("  Cargando ViT LoRA y fusionando pesos (merge_and_unload)...")
    # ── FIX DEFINITIVO: merge_and_unload() en el ViT ─────────────────────────
    #
    # ROOT CAUSE del error "You have to specify pixel_values":
    #   generate() llama  self.vision_model(pixel_values, ...)  POSICIONAL.
    #   PeftModelForFeatureExtraction.forward(self, input_ids=None, ...) mapea
    #   ese arg posicional a su primer parámetro "input_ids" (nombre distinto).
    #   El whitelist del monkey-patch elimina "input_ids" como kwarg.
    #   El original Blip2VisionModel.forward recibe pixel_values=None → error.
    #   (Nota: durante entrenamiento forward() llama vision_model(pixel_values=..)
    #    como KEYWORD, que cae en **kwargs y el whitelist lo conserva → OK)
    #
    # merge_and_unload() fusiona los deltas LoRA (W += B@A×scaling) en los
    # pesos base y devuelve un Blip2VisionModel estándar sin wrapper PeftModel.
    # generate() llama el forward nativo: pixel_values posicional → args[0] →
    # pasa directamente al forward original sin pasar por PeftModel → OK.
    #
    # FIX ADICIONAL — dtype:
    #   Las matrices LoRA A/B se almacenan en float32 por defecto.
    #   Tras el merge (W_bf16 + delta_f32), PyTorch puede upcastear W a float32.
    #   Forzamos bfloat16 + device usando qformer como referencia (submodulo
    #   que nunca se toca y siempre refleja el dtype/device del modelo base).
    # ─────────────────────────────────────────────────────────────────────────
    vit_peft = PeftModel.from_pretrained(
        base_model.vision_model,
        str(vit_dir),
        is_trainable=False,
    )
    _ref = next(base_model.qformer.parameters())   # referencia de device/dtype
    base_model.vision_model = vit_peft.merge_and_unload().to(
        device=_ref.device, dtype=_ref.dtype
    )
    logger.info(
        f"  ViT LoRA fusionado → device={_ref.device}, dtype={_ref.dtype}"
    )

    model = base_model
    model.eval()
    device = _ref.device                           # garantizado CUDA
    logger.info(f"  Modelo en {device} | dtype={_ref.dtype}")

    # ── Test Dataset ──────────────────────────────────────────────────────────
    logger.info("  Cargando Test Set (10 %)...")
    test_dataset = MimicCXRDataset(
        csv_path            = TEST_CSV_10PCT,
        images_dir          = IMAGES_DIR_10PCT,
        processor           = processor,
        image_size          = config.data.image_size,
        clahe_clip_limit    = config.data.clahe_clip_limit,
        clahe_tile_grid_size= config.data.clahe_tile_grid_size,
        max_length          = config.data.max_length,
        padding             = config.data.padding,
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

        # Prompt con metadato de vista (igual que evaluate_final.py)
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
    Itera sobre las 4 configuraciones del torneo en secuencia.

    Por cada config:
        [Fase Train]  train_single_config_phase2()
        [Fase Inf]    fase1_inferencia_phase2()   (salvo --skip-inference)
        [Fase Eval]   evaluate_final.fase2_evaluacion()

    Al finalizar todas las configs:
        - Ranking final (CSV)
        - Resumen global de métricas (CSV)
        - 3 PNGs comparativos de curvas
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Pre-cargar modelos de evaluación UNA sola vez (muy costoso en RAM) ──
    logger.info("\nCargando F1RadGraph (reward_level='all', model_type='radgraph-xl')...")
    f1rg = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    logger.info("  F1RadGraph listo")

    logger.info("Cargando CheXbert oficial (Stanford AIMI)...")
    chexbert = CheXbertLabeler()
    logger.info("  CheXbert listo\n")

    # ── Procesador BLIP-2 (compartido; no ocupa VRAM significativa) ──────────
    logger.info("Cargando Blip2Processor...")
    processor = cargar_procesador(model_name=config.model.model_name)
    logger.info("  Processor listo\n")

    # ── Datasets de entrenamiento / validación (compartidos entre configs) ───
    logger.info("Cargando datasets del 10 % (train + val)...")
    train_dataset, val_dataset = load_phase2_datasets(processor)

    # ── Contenedores de resultados ────────────────────────────────────────────
    train_results:  List[Dict]            = []
    granular_dfs:   Dict[str, pd.DataFrame] = {}

    total = len(configs_to_run)

    for i, cfg in enumerate(configs_to_run, 1):
        name = cfg["name"]
        logger.info(f"\n{'#'*70}")
        logger.info(f"  TORNEO — Config {i}/{total}: {name}")
        logger.info(f"{'#'*70}")

        # ── FASE TRAIN ────────────────────────────────────────────────────────
        if not skip_train:
            try:
                result = train_single_config_phase2(
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
                logger.warning(f"  Saltando {name} y continuando con la siguiente config...")
                continue

            # Guardar resultados parciales de entrenamiento
            pd.DataFrame(train_results).to_csv(
                OUTPUT_DIR / "intermediate_train_results.csv", index=False
            )
        else:
            # Modo --skip-train: verificar que el checkpoint existe antes de continuar
            model_dir_check = CHECKPOINTS_DIR / f"best_model_{name}"
            if not model_dir_check.exists():
                logger.warning(
                    f"  [skip-train] Checkpoint no encontrado: {model_dir_check}\n"
                    f"  Saltando {name} (entrenamiento aún no ha concluido para esta config)."
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
                preds_path = fase1_inferencia_phase2(
                    config_name = name,
                    model_dir   = model_dir,
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

        # ── FASE EVALUACIÓN (reutiliza evaluate_final.fase2_evaluacion) ───────
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
        show_cols = ["config_name", "llm_r", "vit_r", "best_val_loss", "best_step",
                     "total_steps", "final_train_loss"]
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
        print("  MÉTRICAS GLOBALES DE EVALUACIÓN")
        print(f"{'='*70}")
        print(df_summary[show_metric_cols].to_string(index=False))
        print(f"{'='*70}\n")

    # Gráficos comparativos de curvas (Val Loss / BLEU-4 / ROUGE-L)
    plot_comparative_curves(
        config_names = [c["name"] for c in configs_to_run],
        output_dir   = OUTPUT_DIR,
        plots_dir    = OUTPUT_DIR / "plots",
    )

    logger.info(f"\nTorneo Fase 2.1 completado. Resultados en: {OUTPUT_DIR}")
    logger.info(f"  tournament_train_ranked.csv")
    logger.info(f"  evaluation/metricas_globales_comparacion.csv")
    logger.info(f"  evaluation/resultados_granulares_{{name}}.csv  (x{len(granular_dfs)})")
    logger.info(f"  plots/01_val_loss_phase2.png  /  02_bleu4_phase2.png  /  03_rougeL_phase2.png\n")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    valid_names = [c["name"] for c in TOURNAMENT_CONFIGS]
    parser = argparse.ArgumentParser(
        description=(
            "Mini-Torneo Fase 2.1 — Ablación Visual LoRA (ViT+LLM) al 1%% de MIMIC-CXR."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=valid_names,
        default=valid_names,
        metavar="CONFIG",
        help=(
            f"Configs a ejecutar. Opciones: {valid_names}. "
            "Por defecto corre las 4 en secuencia."
        ),
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help=(
            "Saltar entrenamiento y pasar directamente a inferencia + evaluación "
            "usando los checkpoints ya guardados en "
            "phase2_results/checkpoints/best_model_{name}/. "
            "Úsalo cuando el entrenamiento ya concluyó."
        ),
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help=(
            "Saltar Fase 1 (inferencia) y reutilizar predicciones_{{name}}.csv "
            "existentes en phase2_results/evaluation/."
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

    # ── Logging a fichero (desde este punto todo queda guardado) ─────────────
    from trainer_utils import OUTPUT_DIR as _OUT
    _log_file = _OUT / "tournament_Config_B1.log"
    _setup_file_logger(_log_file)

    chexpert_csv = Path(args.chexpert_csv) if args.chexpert_csv else None
    
    # Auto-fallback: si no existe, probar la variante .csv/.csv.gz opuesta
    if chexpert_csv is not None and not chexpert_csv.exists():
        if str(chexpert_csv).endswith(".gz"):
            fallback = Path(str(chexpert_csv)[:-3])  # quitar .gz
        else:
            fallback = Path(str(chexpert_csv) + ".gz")  # añadir .gz
        
        if fallback.exists():
            logger.info(f"CSV CheXpert: usando variante {fallback.name}")
            chexpert_csv = fallback
        else:
            logger.warning(
                f"CSV CheXpert no encontrado: {chexpert_csv}\n"
                f"  Tampoco existe: {fallback}\n"
                f"  Se omitirán métricas CheXpert (precision/recall/F1)."
            )
            chexpert_csv = None

    print("\n" + "=" * 70)
    print("  FASE 2.2 — ESCALAMIENTO DE FINALISTAS: Config_A2 vs Config_B1")
    print("=" * 70)
    print(f"  GPU target  : L40S (48 GB VRAM)")
    print(f"  Dataset     : 10 % MIMIC-CXR  (10x más datos que Fase 2.1)")
    print(f"  Configs     : {[c['name'] for c in configs_to_run]}")
    print(f"  LR LLM      : {TRAINING_CONFIG['lr_llm']:.0e}   (anclado a Fase 1)")
    print(f"  LR ViT      : {TRAINING_CONFIG['lr_vit']:.0e}   (domain adaptation)")
    print(f"  Batch ef.   : {TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['gradient_accumulation']}")
    print(f"  Max epochs  : {TRAINING_CONFIG['max_epochs']}")
    print(f"  Early stop  : patience={TRAINING_CONFIG['early_stopping_patience']} evals")
    print(f"  Optimización: skip_text_generation=True (ahorro ~70% en eval)")

    print(f"  skip-train  : {args.skip_train}")
    print(f"  skip-inf.   : {args.skip_inference}")
    print(f"  num-samples : {args.num_samples or 'ALL'}")
    print(f"  Output      : {OUTPUT_DIR}")
    print("=" * 70 + "\n")

    run_tournament(
        configs_to_run = configs_to_run,
        skip_train     = args.skip_train,
        skip_inference = args.skip_inference,
        num_samples    = args.num_samples,
        chexpert_csv   = chexpert_csv,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
