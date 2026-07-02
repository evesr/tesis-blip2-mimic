#!/usr/bin/env python3
"""
verify_fase2_2.py — Verificación pre-ejecución Fase 2.2
========================================================
Valida que todos los archivos necesarios existan antes de
lanzar python run_tournament.py
"""

import sys
from pathlib import Path

BASE_DIR = Path("/workspace/tesis-blip2-mimic")

# Archivos críticos
CRITICAL_FILES = {
    "CSVs del 10%": [
        "train_split_10pct_limpio.csv",
        "val_split_10pct_limpio.csv",
        "test_split_10pct_limpio.csv",
    ],
    "Directorios": [
        "imagenes_10_porciento",
    ],
    "Pesos pre-entrenados Fase 1": [
        "final_results/checkpoints/best_model_Config_A",
        "final_results/checkpoints/best_model_Config_B",
    ],
    "Scripts de ejecución": [
        "trainer_utils.py",
        "run_tournament.py",
        "evaluate_final.py",
    ],
}

def main():
    print("=" * 70)
    print("  VERIFICACIÓN FASE 2.2 — ESCALAMIENTO AL 10%")
    print("=" * 70)
    
    errors = []
    
    for category, files in CRITICAL_FILES.items():
        print(f"\n{category}:")
        for fname in files:
            fpath = BASE_DIR / fname
            if fpath.exists():
                if fpath.is_dir():
                    n_imgs = len(list(fpath.glob("*.jpg")))
                    print(f"  ✓ {fname:45} ({n_imgs:,} imágenes)")
                else:
                    size_mb = fpath.stat().st_size / (1024 * 1024)
                    print(f"  ✓ {fname:45} ({size_mb:.1f} MB)")
            else:
                print(f"  ✗ {fname:45} ← NO EXISTE")
                errors.append(fname)
    
    print("\n" + "=" * 70)
    
    if errors:
        print("⚠️  FALTAN ARCHIVOS CRÍTICOS:")
        for err in errors:
            print(f"   - {err}")
        print("\nEjecuta primero:")
        print("   python muestreo_10_porciento.py")
        print("   python train_semifinal.py  (si no tienes checkpoints Fase 1)")
        print("=" * 70)
        return 1
    else:
        print("✅ TODOS LOS ARCHIVOS LISTOS")
        print("\nPuedes ejecutar:")
        print("   python run_tournament.py")
        print("   # o con TMux:")
        print('   tmux new-session -d -s fase2 "python run_tournament.py 2>&1 | tee phase2_results/tournament_10pct.log"')
        print("=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
