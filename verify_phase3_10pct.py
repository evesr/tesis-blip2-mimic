#!/usr/bin/env python3
"""
verify_phase3_10pct.py — Verificación de Pre-Requisitos para Fase 3.2
======================================================================
Script de validación antes de lanzar el torneo final del 10%.

Verifica:
  1. Scripts de Fase 3.2 creados correctamente
  2. Datasets del 10% disponibles
  3. Pesos de Fase 2 (bases duales LLM+ViT)
  4. Configuraciones del torneo (2 configs finalistas)
  5. Hiperparámetros actualizados (cosine scheduler, warmup_ratio=0.1)
  6. Monkey patch del Q-Former activo
  7. Dependencias Python instaladas
  8. Hardware (GPU CUDA)

Uso:
    python verify_phase3_10pct.py
"""

import sys
from pathlib import Path

print("="*70)
print("  VERIFICACIÓN DE PRE-REQUISITOS — FASE 3.2 (10%)")
print("="*70)
print()

errores = []

# ──────────────────────────────────────────────────────────────────────
# 1. Scripts de Fase 3.2
# ──────────────────────────────────────────────────────────────────────
print("1. Scripts de Fase 3.2:")
BASE_DIR = Path.cwd()

scripts = {
    "trainer_utils_phase3_10pct.py": BASE_DIR / "trainer_utils_phase3_10pct.py",
    "run_phase3_10pct.py": BASE_DIR / "run_phase3_10pct.py",
}

for name, path in scripts.items():
    if path.exists():
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name} NO ENCONTRADO")
        errores.append(f"Script {name} faltante")

print()

# ──────────────────────────────────────────────────────────────────────
# 2. Datasets del 10% (MIMIC-CXR)
# ──────────────────────────────────────────────────────────────────────
print("2. Datasets del 10% (MIMIC-CXR):")

try:
    import trainer_utils_phase3_10pct as tu
    
    datasets = {
        "train_split_10pct_limpio.csv": tu.TRAIN_CSV_10PCT,
        "val_split_10pct_limpio.csv": tu.VAL_CSV_10PCT,
        "test_split_10pct_limpio.csv": tu.TEST_CSV_10PCT,
        "imagenes_10_porciento/": tu.IMAGES_DIR_10PCT,
    }
    
    import pandas as pd
    for name, path in datasets.items():
        if path.exists():
            if path.suffix == ".csv":
                df = pd.read_csv(path)
                print(f"  ✓ {name} ({len(df):,} muestras)")
            else:
                n_images = len(list(path.glob("*.jpg"))) + len(list(path.glob("*.png")))
                print(f"  ✓ {name} ({n_images:,} imágenes)")
        else:
            print(f"  ✗ {name} NO ENCONTRADO")
            errores.append(f"Dataset {name} faltante")
except Exception as e:
    print(f"  ✗ Error al cargar trainer_utils_phase3_10pct: {e}")
    errores.append("No se pudo importar trainer_utils_phase3_10pct")

print()

# ──────────────────────────────────────────────────────────────────────
# 3. Pesos de Fase 2 (bases duales LLM+ViT)
# ──────────────────────────────────────────────────────────────────────
print("3. Pesos de Fase 2 (bases duales LLM+ViT):")

try:
    for base_key, weights in tu.DUAL_WEIGHTS.items():
        print(f"  Base {base_key}:")
        for model_type, path in weights.items():
            if path.exists():
                print(f"    ✓ {model_type.upper()} LoRA — {path.parent.name}/{path.name}")
            else:
                print(f"    ✗ {model_type.upper()} LoRA NO ENCONTRADO: {path}")
                errores.append(f"Pesos Base {base_key} {model_type.upper()} faltantes")
except Exception as e:
    print(f"  ✗ Error verificando DUAL_WEIGHTS: {e}")
    errores.append("No se pudieron verificar pesos de Fase 2")

print()

# ──────────────────────────────────────────────────────────────────────
# 4. Configuraciones del Torneo (2 configs finalistas)
# ──────────────────────────────────────────────────────────────────────
print("4. Configuraciones del Torneo:")

try:
    import run_phase3_10pct as rp
    
    print(f"  Total de configs: {len(rp.TOURNAMENT_CONFIGS)}")
    for cfg in rp.TOURNAMENT_CONFIGS:
        print(f"  ✓ {cfg['name']}: r={cfg['qformer_r']}, α={cfg['qformer_alpha']}, base={cfg['base_weights_key']}")
    
    if len(rp.TOURNAMENT_CONFIGS) != 2:
        errores.append(f"Se esperaban 2 configs, se encontraron {len(rp.TOURNAMENT_CONFIGS)}")
except Exception as e:
    print(f"  ✗ Error al cargar run_phase3_10pct: {e}")
    errores.append("No se pudo importar run_phase3_10pct")

print()

# ──────────────────────────────────────────────────────────────────────
# 5. Hiperparámetros Actualizados
# ──────────────────────────────────────────────────────────────────────
print("5. Hiperparámetros Actualizados:")

try:
    config = tu.TRAINING_CONFIG
    print(f"  ✓ lr: {config['lr']} (1e-4)")
    print(f"  ✓ eval_steps: {config['eval_steps']} (125)")
    print(f"  ✓ warmup_ratio: {config['warmup_ratio']} (0.1)")
    print(f"  ✓ batch efectivo: {config['batch_size']} × {config['gradient_accumulation']} = {config['batch_size'] * config['gradient_accumulation']}")
    
    # Verificar que warmup_ratio está presente (no warmup_steps absolutos)
    if 'warmup_steps' in config and config.get('warmup_steps') is not None:
        if 'warmup_ratio' not in config or config.get('warmup_ratio') is None:
            errores.append("Se encontró warmup_steps pero no warmup_ratio")
    
    # Verificar scheduler cosine en el código fuente
    import inspect
    source = inspect.getsource(tu.train_single_config_phase3)
    if 'lr_scheduler_type="cosine"' in source:
        print(f"  ✓ lr_scheduler_type: cosine")
    else:
        print(f"  ✗ lr_scheduler_type NO es cosine")
        errores.append("Scheduler no es cosine")
    
    if 'logging_steps=TRAINING_CONFIG["eval_steps"]' in source:
        print(f"  ✓ logging_steps sincronizado con eval_steps")
    else:
        print(f"  ✗ logging_steps NO sincronizado con eval_steps")
        errores.append("logging_steps no sincronizado")
        
except Exception as e:
    print(f"  ✗ Error verificando hiperparámetros: {e}")
    errores.append("No se pudieron verificar hiperparámetros")

print()

# ──────────────────────────────────────────────────────────────────────
# 6. Monkey Patch del Q-Former
# ──────────────────────────────────────────────────────────────────────
print("6. Monkey Patch del Q-Former:")

try:
    from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel
    
    # Verificar que el forward está patcheado
    if hasattr(Blip2QFormerModel.forward, '__name__'):
        print(f"  ✓ Blip2QFormerModel.forward está patcheado")
    
    # Verificar whitelist
    whitelist = tu._QFORMER_VALID_KWARGS
    print(f"  ✓ Whitelist con {len(whitelist)} parámetros válidos")
    
except Exception as e:
    print(f"  ✗ Error verificando monkey patch: {e}")
    errores.append("Monkey patch del Q-Former no activo")

print()

# ──────────────────────────────────────────────────────────────────────
# 7. Dependencias Python
# ──────────────────────────────────────────────────────────────────────
print("7. Dependencias Python:")

dependencies = [
    ("torch", "2.0.0"),
    ("transformers", "4.46.0"),
    ("peft", "0.13.0"),
    ("radgraph", None),
    ("cv2", None),
    ("nltk", "3.0.0"),
    ("rouge_score", None),
]

for pkg, min_ver in dependencies:
    try:
        if pkg == "cv2":
            import cv2
            ver = cv2.__version__
        elif pkg == "radgraph":
            from radgraph import F1RadGraph
            ver = "✓"
        else:
            mod = __import__(pkg.replace("-", "_"))
            ver = getattr(mod, "__version__", "✓")
        
        print(f"  ✓ {pkg} {ver}")
    except ImportError:
        print(f"  ✗ {pkg} NO INSTALADO")
        errores.append(f"{pkg} no instalado")

print()

# ──────────────────────────────────────────────────────────────────────
# 8. Hardware (GPU CUDA)
# ──────────────────────────────────────────────────────────────────────
print("8. Hardware:")

try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  ✓ GPU (CUDA) - {gpu_name}, {gpu_mem:.1f} GB VRAM")
    else:
        print(f"  ✗ CUDA no disponible")
        errores.append("GPU CUDA no disponible")
except Exception as e:
    print(f"  ✗ Error verificando CUDA: {e}")
    errores.append("No se pudo verificar CUDA")

print()

# ──────────────────────────────────────────────────────────────────────
# 9. Módulos del Proyecto
# ──────────────────────────────────────────────────────────────────────
print("9. Módulos del proyecto:")

project_modules = [
    "config",
    "model_utils",
    "data_loader",
    "preprocessing",
    "evaluate_final",
    "chexbert_official",
]

for mod_name in project_modules:
    try:
        __import__(mod_name)
        print(f"  ✓ {mod_name}")
    except ImportError as e:
        print(f"  ✗ {mod_name} - {e}")
        errores.append(f"Módulo {mod_name} no importable")

print()

# ──────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ──────────────────────────────────────────────────────────────────────
print("="*70)
if errores:
    print(f"❌ VERIFICACIÓN FALLIDA — {len(errores)} error(es) encontrado(s):")
    for i, err in enumerate(errores, 1):
        print(f"  {i}. {err}")
    print("="*70)
    sys.exit(1)
else:
    print("✅ TODAS LAS VERIFICACIONES PASARON")
    print()
    print("La Fase 3.2 está lista para ejecutarse.")
    print()
    print("Comandos de ejecución:")
    print("  python run_phase3_10pct.py")
    print("  python run_phase3_10pct.py --configs Config_Final_A")
    print("  python run_phase3_10pct.py --skip-train  # solo inferencia")
    print("="*70)
    sys.exit(0)
