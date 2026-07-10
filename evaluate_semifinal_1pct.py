#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_semifinal_1pct.py — Inferencia + BLEU-4/ROUGE-L de los modelos
SEMIFINAL (phase1_semifinal_results) sobre el Test Set 1% COMPLETO.

Modelos (mejor checkpoint = best_step del ranking):
    • Config_4  (r=16, alpha=32)  -> checkpoint-3200
    • Config_5  (r=16, alpha=16)  -> checkpoint-5100
    • Config_6  (r=32, alpha=64)  -> checkpoint-2650

Reutiliza SIN duplicar la inferencia batched ya validada de `evaluate_final.py`
(FASE 1), pero apuntando a:
    • Test CSV  : test_split_completo_1pct.csv   (397 muestras, imagen garantizada)
    • Imagenes  : imagenes_50_porciento           (banco disponible)
    • Salida    : phase1_semifinal_1pct_eval_results/

Metricas: SOLO BLEU-4 y ROUGE-L (peticion del usuario). NO carga CheXbert ni
F1RadGraph -> evaluacion ligera y rapida.

Salidas:
    • predicciones_Config_{4,5,6}.csv              (inferencia)
    • resultados_granulares_Config_{4,5,6}.csv     (+ bleu_4, rouge_l por fila)
    • metricas_comparativas_semifinal_1pct.csv     (CSV comparativo unico)

Uso:
    python evaluate_semifinal_1pct.py
    python evaluate_semifinal_1pct.py --skip-inference
    python evaluate_semifinal_1pct.py --configs Config_4 Config_6

Autor : Evelyn Silva Rozas
Fecha : Julio 2026
"""

import argparse
import dataclasses
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

# ── Config (mutar rutas ANTES de construir datasets) ────────────────────────
from config import config

BASE = config.paths.base_dir
config.paths.images_dir = BASE / "imagenes_50_porciento"
config.paths.test_csv   = BASE / "test_split_completo_1pct.csv"

# ── Reusar el pipeline validado (inferencia + helpers de metricas) ──────────
import evaluate_final as ef  # noqa: E402  (import despues de mutar config)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("semifinal_1pct")

# Modelos SEMIFINAL — mejor checkpoint por config (best_step del ranking)
CONFIGS = [
    {"name": "Config_4", "r": 16, "alpha": 32,
     "model_dir": BASE / "phase1_semifinal_results" / "Config_4" / "checkpoint-3200"},
    {"name": "Config_5", "r": 16, "alpha": 16,
     "model_dir": BASE / "phase1_semifinal_results" / "Config_5" / "checkpoint-5100"},
    {"name": "Config_6", "r": 32, "alpha": 64,
     "model_dir": BASE / "phase1_semifinal_results" / "Config_6" / "checkpoint-2650"},
]

OUTPUT_DIR = BASE / "phase1_semifinal_1pct_eval_results"
OUT_COMPARATIVO = OUTPUT_DIR / "metricas_comparativas_semifinal_1pct.csv"

ROUND = 4


# ─────────────────────────────────────────────────────────────────────────────
# Compatibilidad de checkpoints (peft 0.19.1 -> instalado 0.13.2)
# ─────────────────────────────────────────────────────────────────────────────

def _sanitizar_checkpoint(model_dir: Path, work_dir: Path) -> Path:
    """
    Devuelve un directorio de checkpoint compatible con la version de peft
    instalada. Los checkpoints SEMIFINAL se guardaron con peft 0.19.1 e incluyen
    claves en adapter_config.json que la version instalada (0.13.2) no acepta
    (p.ej. 'alora_invocation_tokens', 'qalora_group_size', ...). Aqui se crea una
    copia SANEADA: adapter_config.json filtrado a los campos validos de
    LoraConfig + un symlink a adapter_model.safetensors (sin tocar el original).
    """
    from peft import LoraConfig

    src_cfg = model_dir / "adapter_config.json"
    src_wts = model_dir / "adapter_model.safetensors"
    if not src_cfg.exists() or not src_wts.exists():
        return model_dir  # nada que sanear

    cfg = json.loads(src_cfg.read_text())
    valid = {f.name for f in dataclasses.fields(LoraConfig)}
    unknown = [k for k in cfg if k not in valid]
    if not unknown:
        return model_dir  # ya compatible

    cfg_clean = {k: v for k, v in cfg.items() if k in valid}
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "adapter_config.json").write_text(json.dumps(cfg_clean, indent=2))

    dst_wts = work_dir / "adapter_model.safetensors"
    if dst_wts.exists() or dst_wts.is_symlink():
        dst_wts.unlink()
    os.symlink(src_wts.resolve(), dst_wts)

    logger.info("  Adapter saneado (%d claves removidas) -> %s",
                len(unknown), work_dir)
    return work_dir


# ─────────────────────────────────────────────────────────────────────────────
# FASE 2 ligera — solo BLEU-4 + ROUGE-L
# ─────────────────────────────────────────────────────────────────────────────

def evaluar_bleu_rouge(preds_path: Path, cfg_name: str, output_dir: Path) -> pd.DataFrame:
    """Calcula BLEU-4 y ROUGE-L fila por fila sobre las predicciones."""
    logger.info("=" * 65)
    logger.info("  Metricas BLEU-4 / ROUGE-L: %s", cfg_name)
    logger.info("=" * 65)

    df = pd.read_csv(preds_path)
    logger.info("  Filas: %d", len(df))

    rows_out = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  Metricas {cfg_name}", unit="row"):
        ref = str(row["reference_report"]).strip() or "no findings"
        hyp = str(row["generated_report"]).strip() or "no findings"

        try:
            bleu = ef._bleu4(ref, hyp)
        except Exception as exc:
            logger.warning("  BLEU-4 error: %s", exc)
            bleu = float("nan")
        try:
            rouge = ef._rouge_l(ref, hyp)
        except Exception as exc:
            logger.warning("  ROUGE-L error: %s", exc)
            rouge = float("nan")

        rows_out.append({
            "dicom_id":         row.get("dicom_id", ""),
            "study_id":         row.get("study_id", ""),
            "reference_report": ref,
            "generated_report": hyp,
            "bleu_4":           bleu,
            "rouge_l":          rouge,
        })

    df_out = pd.DataFrame(rows_out)
    granular_path = output_dir / f"resultados_granulares_{cfg_name}.csv"
    df_out.to_csv(granular_path, index=False)
    logger.info("  CSV granular: %s", granular_path)
    return df_out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inferencia + BLEU-4/ROUGE-L de los modelos SEMIFINAL sobre test 1% completo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--num-samples", type=int, default=None,
                   help="Limitar a N muestras (debug).")
    p.add_argument("--skip-inference", action="store_true",
                   help="Saltar inferencia y reutilizar predicciones_X.csv existentes.")
    p.add_argument("--configs", nargs="+",
                   choices=["Config_4", "Config_5", "Config_6"],
                   default=["Config_4", "Config_5", "Config_6"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("  EVALUACION SEMIFINAL — Test 1%% COMPLETO (397) — SOLO BLEU/ROUGE")
    logger.info("  Test CSV : %s", config.paths.test_csv.name)
    logger.info("  Imagenes : %s", config.paths.images_dir.name)
    logger.info("  Salida   : %s", output_dir)
    logger.info("=" * 65)

    configs_a_evaluar = [c for c in CONFIGS if c["name"] in args.configs]

    granular_dfs: Dict[str, pd.DataFrame] = {}

    for cfg_dict in configs_a_evaluar:
        cfg_name   = cfg_dict["name"]
        preds_path = output_dir / f"predicciones_{cfg_name}.csv"

        # ── FASE 1: inferencia (con resume simple) ──
        reuse = False
        if args.skip_inference and preds_path.exists():
            reuse = True
        elif preds_path.exists():
            try:
                n_prev = sum(1 for _ in open(preds_path)) - 1
                n_test = sum(1 for _ in open(config.paths.test_csv)) - 1
                if n_prev >= n_test:
                    logger.info("[FASE 1] %s ya completo (%d filas) — se reutiliza.",
                                cfg_name, n_prev)
                    reuse = True
            except Exception:
                reuse = False

        if reuse:
            logger.info("[FASE 1] Reutilizando %s", preds_path.name)
        else:
            # Sanear el checkpoint (peft 0.19.1 -> instalado) antes de cargar LoRA
            sane_dir = _sanitizar_checkpoint(
                Path(cfg_dict["model_dir"]),
                output_dir / "_sanitized_adapters" / cfg_name,
            )
            cfg_run = dict(cfg_dict, model_dir=sane_dir)
            preds_path = ef.fase1_inferencia(
                config_dict     = cfg_run,
                output_dir      = output_dir,
                num_samples     = args.num_samples,
                wrap_full_model = True,   # SEMIFINAL: LoRA entrenada sobre el Blip2 completo
            )
            gc.collect()
            torch.cuda.empty_cache()

        # ── FASE 2: SOLO BLEU-4 + ROUGE-L ──
        df_gran = evaluar_bleu_rouge(preds_path, cfg_name, output_dir)
        granular_dfs[cfg_name] = df_gran

    # ── CSV comparativo unico ──
    if granular_dfs:
        logger.info("\n" + "=" * 65)
        logger.info("  Construyendo CSV comparativo (BLEU-4 / ROUGE-L)...")

        cfg_meta = {c["name"]: c for c in CONFIGS}
        filas = []
        for cfg_name, df in granular_dfs.items():
            meta = cfg_meta.get(cfg_name, {})
            filas.append({
                "config":    cfg_name,
                "r":         meta.get("r", ""),
                "alpha":     meta.get("alpha", ""),
                "n_samples": len(df),
                "bleu_4":    round(float(df["bleu_4"].mean(skipna=True)), ROUND),
                "rouge_l":   round(float(df["rouge_l"].mean(skipna=True)), ROUND),
            })

        df_cmp = pd.DataFrame(filas).sort_values("config").reset_index(drop=True)
        out_comparativo = output_dir / "metricas_comparativas_semifinal_1pct.csv"
        df_cmp.to_csv(out_comparativo, index=False)

        print("\n" + "=" * 65)
        print("  METRICAS COMPARATIVAS SEMIFINAL — Test 1% completo")
        print("=" * 65)
        print(df_cmp.to_string(index=False))
        print("=" * 65)
        print(f"\n[OK] {out_comparativo}\n")

    logger.info("Evaluacion SEMIFINAL (test 1%% completo) finalizada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
