#!/usr/bin/env python3
"""
verify_phase3.py — Verificación de pre-requisitos para Fase 3
==============================================================
Valida que todos los componentes necesarios estén disponibles antes
de lanzar el torneo de Q-Former LoRA.

Autor: Evelyn Silva Rozas
Fecha: Mayo 2026
"""

import sys
from pathlib import Path
import pandas as pd

# Colores ANSI para terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

def check(condition: bool, description: str, details: str = "") -> bool:
    """Imprime resultado de una verificación."""
    status = f"{GREEN}✓{RESET}" if condition else f"{RED}✗{RESET}"
    print(f"  {status} {description}")
    if details and not condition:
        print(f"    {YELLOW}→ {details}{RESET}")
    return condition


def main():
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}  VERIFICACIÓN DE PRE-REQUISITOS — FASE 3{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")

    base_dir = Path("/workspace/tesis-blip2-mimic")
    all_ok = True

    # ── 1. Verificar scripts de Fase 3 ────────────────────────────────────
    print(f"{BOLD}1. Scripts de Fase 3:{RESET}")
    all_ok &= check(
        (base_dir / "trainer_utils_phase3.py").exists(),
        "trainer_utils_phase3.py",
        "Archivo no encontrado. Debe estar en el directorio raíz."
    )
    all_ok &= check(
        (base_dir / "run_tournament_phase3.py").exists(),
        "run_tournament_phase3.py",
        "Archivo no encontrado. Debe estar en el directorio raíz."
    )
    all_ok &= check(
        (base_dir / "run_tournament_phase3_tmux.sh").exists() and
        (base_dir / "run_tournament_phase3_tmux.sh").stat().st_mode & 0o111,
        "run_tournament_phase3_tmux.sh (ejecutable)",
        "Archivo no encontrado o no es ejecutable. Ejecuta: chmod +x run_tournament_phase3_tmux.sh"
    )
    print()

    # ── 2. Verificar datasets del 1% ──────────────────────────────────────
    print(f"{BOLD}2. Datasets del 1% (MIMIC-CXR):{RESET}")
    
    train_csv = base_dir / "train_split_limpio.csv"
    val_csv   = base_dir / "val_split_limpio.csv"
    test_csv  = base_dir / "test_split_limpio.csv"
    images_dir = base_dir / "imagenes_1_porciento"

    train_ok = check(train_csv.exists(), f"train_split_limpio.csv", f"Ruta: {train_csv}")
    val_ok   = check(val_csv.exists(),   f"val_split_limpio.csv",   f"Ruta: {val_csv}")
    test_ok  = check(test_csv.exists(),  f"test_split_limpio.csv",  f"Ruta: {test_csv}")
    img_ok   = check(images_dir.exists() and images_dir.is_dir(),
                     f"imagenes_1_porciento/", f"Ruta: {images_dir}")
    
    all_ok &= train_ok and val_ok and test_ok and img_ok

    # Contar muestras si los CSVs existen
    if train_ok:
        n_train = len(pd.read_csv(train_csv))
        print(f"    {BLUE}→ Train: {n_train:,} muestras{RESET}")
    if val_ok:
        n_val = len(pd.read_csv(val_csv))
        print(f"    {BLUE}→ Val:   {n_val:,} muestras{RESET}")
    if test_ok:
        n_test = len(pd.read_csv(test_csv))
        print(f"    {BLUE}→ Test:  {n_test:,} muestras{RESET}")
    print()

    # ── 3. Verificar pesos de Fase 2 (bases duales LLM+ViT) ──────────────
    print(f"{BOLD}3. Pesos de Fase 2 (bases duales LLM+ViT):{RESET}")

    phase2_dir = base_dir / "phase2_results-10pct" / "checkpoints"
    
    # Base A (Config_A2)
    base_a_llm = phase2_dir / "best_model_Config_A2" / "llm"
    base_a_vit = phase2_dir / "best_model_Config_A2" / "vit"
    
    # Base B (Config_B1)
    base_b_llm = phase2_dir / "best_model_Config_B1" / "llm"
    base_b_vit = phase2_dir / "best_model_Config_B1" / "vit"

    all_ok &= check(
        base_a_llm.exists() and (base_a_llm / "adapter_config.json").exists(),
        "Base A — LLM LoRA (Config_A2)",
        f"Ruta: {base_a_llm}\nFase 2 debe haber completado Config_A2."
    )
    all_ok &= check(
        base_a_vit.exists() and (base_a_vit / "adapter_config.json").exists(),
        "Base A — ViT LoRA (Config_A2)",
        f"Ruta: {base_a_vit}\nFase 2 debe haber completado Config_A2."
    )
    all_ok &= check(
        base_b_llm.exists() and (base_b_llm / "adapter_config.json").exists(),
        "Base B — LLM LoRA (Config_B1)",
        f"Ruta: {base_b_llm}\nFase 2 debe haber completado Config_B1."
    )
    all_ok &= check(
        base_b_vit.exists() and (base_b_vit / "adapter_config.json").exists(),
        "Base B — ViT LoRA (Config_B1)",
        f"Ruta: {base_b_vit}\nFase 2 debe haber completado Config_B1."
    )
    print()

    # ── 4. Verificar dependencias Python ──────────────────────────────────
    print(f"{BOLD}4. Dependencias Python:{RESET}")

    try:
        import torch
        all_ok &= check(True, f"torch {torch.__version__}")
    except ImportError:
        all_ok &= check(False, "torch", "pip install torch")

    try:
        import transformers
        all_ok &= check(True, f"transformers {transformers.__version__}")
    except ImportError:
        all_ok &= check(False, "transformers", "pip install transformers")

    try:
        import peft
        all_ok &= check(True, f"peft {peft.__version__}")
    except ImportError:
        all_ok &= check(False, "peft", "pip install peft")

    try:
        from radgraph import F1RadGraph
        all_ok &= check(True, "radgraph")
    except ImportError:
        all_ok &= check(False, "radgraph", "pip install radgraph")

    try:
        import cv2
        all_ok &= check(True, f"opencv-python (cv2)")
    except ImportError:
        all_ok &= check(False, "opencv-python", "pip install opencv-python")

    try:
        import nltk
        all_ok &= check(True, f"nltk {nltk.__version__}")
    except ImportError:
        all_ok &= check(False, "nltk", "pip install nltk")

    try:
        from rouge_score import rouge_scorer
        all_ok &= check(True, "rouge-score")
    except ImportError:
        all_ok &= check(False, "rouge-score", "pip install rouge-score")

    print()

    # ── 5. Verificar GPU ──────────────────────────────────────────────────
    print(f"{BOLD}5. Hardware:{RESET}")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        all_ok &= check(cuda_available, "GPU (CUDA)", "Se requiere GPU para entrenamiento.")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"    {BLUE}→ GPU: {gpu_name}{RESET}")
            print(f"    {BLUE}→ VRAM: {vram_total:.1f} GB{RESET}")
            
            if vram_total < 20:
                print(f"    {YELLOW}⚠️  VRAM < 20 GB. Considera reducir batch_size si hay OOM.{RESET}")
    except:
        all_ok &= check(False, "GPU (CUDA)", "torch.cuda no disponible.")
    
    print()

    # ── 6. Verificar módulos del proyecto ────────────────────────────────
    print(f"{BOLD}6. Módulos del proyecto:{RESET}")
    
    sys.path.insert(0, str(base_dir))
    
    try:
        from config import config
        all_ok &= check(True, "config.py")
    except ImportError as e:
        all_ok &= check(False, "config.py", str(e))

    try:
        from model_utils import cargar_procesador
        all_ok &= check(True, "model_utils.py")
    except ImportError as e:
        all_ok &= check(False, "model_utils.py", str(e))

    try:
        from data_loader import MimicCXRDataset
        all_ok &= check(True, "data_loader.py")
    except ImportError as e:
        all_ok &= check(False, "data_loader.py", str(e))

    try:
        from preprocessing import estandarizar_imagen
        all_ok &= check(True, "preprocessing.py")
    except ImportError as e:
        all_ok &= check(False, "preprocessing.py", str(e))

    try:
        from evaluate_final import fase2_evaluacion
        all_ok &= check(True, "evaluate_final.py")
    except ImportError as e:
        all_ok &= check(False, "evaluate_final.py", str(e))

    try:
        from chexbert_official import CheXbertLabeler
        all_ok &= check(True, "chexbert_official.py")
    except ImportError as e:
        all_ok &= check(False, "chexbert_official.py", str(e))

    print()

    # ── Resumen ───────────────────────────────────────────────────────────
    print(f"{BOLD}{'='*70}{RESET}")
    if all_ok:
        print(f"{GREEN}{BOLD}✅ TODAS LAS VERIFICACIONES PASARON{RESET}")
        print(f"\n{BOLD}Listo para ejecutar Fase 3:{RESET}")
        print(f"  python run_tournament_phase3.py")
        print(f"  {BLUE}o{RESET}")
        print(f"  bash run_tournament_phase3_tmux.sh")
    else:
        print(f"{RED}{BOLD}❌ ALGUNAS VERIFICACIONES FALLARON{RESET}")
        print(f"\n{YELLOW}Revisa los errores arriba y corrígelos antes de continuar.{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
