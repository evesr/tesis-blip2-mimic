#!/usr/bin/env python3
"""
train_phase3_puente.py — Entry point para Fase 3 (Re-entrenamiento del Puente)
================================================================================
Lanza un único entrenamiento Full-FT del puente BLIP-2:
    qformer + language_projection
sobre el modelo Fase 2 (LoRAs LLM+ViT) FUSIONADO permanentemente
(merge_and_unload).

Uso (desde tmux):
    python train_phase3_puente.py
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import gc
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from config import config
from model_utils import cargar_procesador

from trainer_utils_phase3_puente import (
    BASE_DIR,
    OUTPUT_DIR,
    CHECKPOINTS_DIR,
    PHASE2_BEST_DIR,
    PHASE2_LLM_DIR,
    PHASE2_VIT_DIR,
    PUENTE_CONFIG,
    TRAINING_CONFIG,
    load_phase3_datasets,
    plot_phase3_curves,
    train_puente_config,
)


def _audit_environment() -> None:
    """Auditoría de entorno y verificación de pre-requisitos."""
    print("\n" + "=" * 80)
    print("  ENTORNO — Fase 3 (Puente)")
    print("=" * 80)
    print(f"  Hora           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PyTorch        : {torch.__version__}")
    print(f"  CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA           : {torch.version.cuda}")
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  BASE_DIR       : {BASE_DIR}")
    print(f"  OUTPUT_DIR     : {OUTPUT_DIR}")
    print(f"  CHECKPOINTS    : {CHECKPOINTS_DIR}")
    print("=" * 80 + "\n")

    # Verificar que existen los adaptadores LoRA de Fase 2
    missing = []
    for path, label in [
        (PHASE2_BEST_DIR, "PHASE2_BEST_DIR"),
        (PHASE2_LLM_DIR,  "PHASE2_LLM_DIR (LLM LoRA)"),
        (PHASE2_VIT_DIR,  "PHASE2_VIT_DIR (ViT LoRA)"),
        (PHASE2_LLM_DIR / "adapter_config.json", "LLM adapter_config.json"),
        (PHASE2_VIT_DIR / "adapter_config.json", "ViT adapter_config.json"),
    ]:
        if not path.exists():
            missing.append((label, path))
    if missing:
        print("✗ ARCHIVOS FALTANTES (no se puede iniciar Fase 3):")
        for label, path in missing:
            print(f"    · {label}: {path}")
        sys.exit(1)
    print("✓ Pre-requisitos OK: adaptadores LoRA de Fase 2 detectados.\n")


def main() -> None:
    _audit_environment()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Logger global de progreso (se va apendizando entre corridas si se reanuda)
    intermediate_csv = OUTPUT_DIR / "intermediate_results_phase3.csv"

    # Procesador (compartido entre datasets)
    print("→ Cargando procesador BLIP-2...")
    processor = cargar_procesador(config.model.model_name)

    # Datasets (50 % de MIMIC-CXR, mismo split que Fase 2)
    print("\n→ Cargando datasets (50 % de MIMIC-CXR)...")
    train_ds, val_ds = load_phase3_datasets(processor)

    # Entrenamiento
    print("\n→ Iniciando entrenamiento del puente...")
    result = train_puente_config(
        config_dict   = PUENTE_CONFIG,
        train_dataset = train_ds,
        val_dataset   = val_ds,
        processor     = processor,
    )

    # Persistir resumen
    df_result = pd.DataFrame([result])
    if intermediate_csv.exists():
        prev = pd.read_csv(intermediate_csv)
        df_result = pd.concat([prev, df_result], ignore_index=True)
    df_result.to_csv(intermediate_csv, index=False)
    print(f"\n✓ Resultado guardado: {intermediate_csv}")

    # Graficar curvas
    print("\n→ Generando gráficos de curvas Train vs Val...")
    plot_phase3_curves(
        config_name = PUENTE_CONFIG["name"],
        output_dir  = OUTPUT_DIR,
        plots_dir   = plots_dir,
    )

    # CSV final ordenado (mismo formato que Fase 2)
    final_csv = OUTPUT_DIR / "final_results_phase3.csv"
    df_result.to_csv(final_csv, index=False)
    print(f"✓ Resultado final: {final_csv}")

    gc.collect()
    torch.cuda.empty_cache()

    name = PUENTE_CONFIG["name"]
    print("\n" + "=" * 80)
    print("  FASE 3 — RE-ENTRENAMIENTO DEL PUENTE COMPLETADO")
    print("=" * 80)
    print(f"  best_val_loss = {result['best_val_loss']}  @  step {result['best_step']}")
    print(f"  Pesos finales : {CHECKPOINTS_DIR / f'best_model_{name}'}")
    print(f"  History CSV   : {OUTPUT_DIR / f'history_phase3_{name}.csv'}")
    print(f"  Hora          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
