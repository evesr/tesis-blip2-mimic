#!/usr/bin/env python3
"""
train_experiment2_raddino.py — Lanzador Experimento 2 (Fase 4): RadDINO + Puente
================================================================================
Entry point que ENTRENA la arquitectura del Experimento 2 construida por
`trainer_utils_exp2_raddino.build_experiment_2_model`:

    ViT genérico (EVA-CLIP)  →  SUSTITUIDO por RadDINO (microsoft/rad-dino)
        • RadDINO backbone   : CONGELADO, solo LoRA entrenable (query, value)
        • proyector 768→1408 : NUEVO, Full FT (interfaz con el Q-Former)
    language_model (OPT-2.7b): LoRA Fase 2 fusionado + CONGELADO (requires_grad=False)
    qformer + language_projection + query_tokens : Full FT (SIN LoRA)

Inspirado en `train_phase3_puente.py` / `run_phase3_50pct.py`, reutilizando la
misma maquinaria de entrenamiento del proyecto:
    • LengthAwareTrainer + precompute_train_lengths  (group_by_length sin I/O)
    • TrainingArguments idénticas a Fase 3 (bf16, cosine, warmup, save/eval steps)
    • EarlyStoppingCallback

GARANTÍAS PEDIDAS
-----------------
  ✓ GUARDA CHECKPOINTS : `Experiment2Trainer` persiste, cada `save_steps` y en el
                         mejor val_loss, SOLO los parámetros entrenables
                         (RadDINO-LoRA + proyector + Q-Former + language_projection
                         + query_tokens ≈ 108.5M params, ~430 MB) en
                         `exp2_state.pt` — NO el modelo completo de ~5.7 GB.
  ✓ CSV DE LOSSES      : `Experiment2LossCallback` apendiza en CADA evaluación una
                         fila (Step | Train_Loss | Val_Loss | LR_ViT | LR_Bridge)
                         a `history_exp2_{name}.csv`, persistido en disco al vuelo.

  ✓ RESUME-AWARE       : reanuda automáticamente desde el último checkpoint válido
                         y recupera el historial/best_val_loss previos.

Uso (desde tmux)
----------------
    python train_experiment2_raddino.py
    python train_experiment2_raddino.py --fresh            # ignora checkpoints previos
    python train_experiment2_raddino.py --no-query-tokens  # congela query_tokens
    python train_experiment2_raddino.py --smoke            # 1 paso, valida tubería

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Variables de entorno (ANTES de cualquier import de transformers/torch)
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import gc
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# cv2 DEBE desactivarse ANTES de torch/DataLoader (consistente con el proyecto)
import cv2
cv2.setNumThreads(0)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
)

# ── Módulos del proyecto ─────────────────────────────────────────────────────
from config import config
from model_utils import cargar_procesador

# Maquinaria reutilizada de Fase 2 (group_by_length sin I/O de imágenes)
from trainer_utils import LengthAwareTrainer, precompute_train_lengths

# Constructor + helpers del Experimento 2 (creados en trainer_utils_exp2_raddino.py)
from trainer_utils_exp2_raddino import (
    BASE_MODEL_NAME,
    RADDINO_MODEL_ID,
    PHASE2_LLM_DIR,
    DEFAULT_PROMPT,
    OUTPUT_DIR,
    CHECKPOINTS_DIR,
    TRAINING_CONFIG,
    build_experiment_2_model,
    build_experiment_2_optimizer,
    build_experiment_2_datasets,
)

warnings.filterwarnings("ignore")

# Nombre único de la corrida (define los nombres de checkpoints y CSV)
RUN_NAME: str = "Exp2_RadDINO"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TRAINER ESPECIALIZADO — guarda SOLO los parámetros entrenables
# ─────────────────────────────────────────────────────────────────────────────

class Experiment2Trainer(LengthAwareTrainer):
    """
    Trainer que sobreescribe save_model / _load_from_checkpoint / _load_best_model
    para persistir ÚNICAMENTE los parámetros con requires_grad=True:
        RadDINO-LoRA + proyector 768→1408 + Q-Former + language_projection + query_tokens.

    Motivación (idéntica a BridgeOnlyTrainer de Fase 3)
    ---------------------------------------------------
    El modelo completo pesa ~5.7 GB en bfloat16. Guardarlo entero en cada
    save_step es lento (CPU-bound), llena el disco (save_total_limit×5.7 GB) y
    safetensors falla por los pesos atados del OPT (lm_head ↔ embed_tokens).
    Guardar solo los ~108.5M params entrenables (~430 MB) elimina los 3 problemas.

    Para reconstruir el modelo en inferencia: volver a llamar a
    build_experiment_2_model() y cargar este `exp2_state.pt` con strict=False
    (los nombres coinciden porque la arquitectura se reconstruye igual).
    """

    STATE_FILENAME = "exp2_state.pt"

    # ── save_model: solo los parámetros entrenables ─────────────────────────
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        trainable_state = {
            n: p.detach().cpu()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        torch.save(trainable_state, os.path.join(output_dir, self.STATE_FILENAME))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    # ── Helper: cargar los pesos entrenables desde un dir de checkpoint ─────
    def _load_exp2_from(self, ckpt_dir: str) -> None:
        path = os.path.join(ckpt_dir, self.STATE_FILENAME)
        if not os.path.isfile(path):
            return
        # weights_only=True: defensa frente a checkpoints maliciosos (PyTorch ≥ 2.4).
        state = torch.load(path, map_location="cpu", weights_only=True)
        # strict=False: el .pt solo contiene los entrenables; el resto del modelo
        # (RadDINO backbone congelado + LLM fusionado + base del Q-Former) ya está
        # en su sitio desde build_experiment_2_model().
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"   [resume] WARN: {len(unexpected)} keys inesperadas en {self.STATE_FILENAME}")

    # ── Resume desde checkpoint intermedio del Trainer ──────────────────────
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if os.path.isfile(os.path.join(resume_from_checkpoint, self.STATE_FILENAME)):
            self._load_exp2_from(resume_from_checkpoint)
        else:
            super()._load_from_checkpoint(resume_from_checkpoint, model)

    # ── Cargar el mejor modelo al final (load_best_model_at_end=True) ───────
    def _load_best_model(self):
        ckpt = self.state.best_model_checkpoint
        if ckpt is None:
            return
        if os.path.isfile(os.path.join(ckpt, self.STATE_FILENAME)):
            self._load_exp2_from(ckpt)
        else:
            super()._load_best_model()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CALLBACK — CSV de losses (cada eval) + checkpoint best-only
# ─────────────────────────────────────────────────────────────────────────────

class Experiment2LossCallback(TrainerCallback):
    """
    Ejecutado tras cada evaluate() del Trainer. Análogo a Phase3PuenteAuditCallback.

      • CSV de losses : apendiza  Step | Train_Loss | Val_Loss | LR_ViT | LR_Bridge
                        a `history_exp2_{name}.csv` (persistido EN CADA eval).
      • Checkpoint best: cuando val_loss mejora, guarda el state_dict de SOLO los
                        parámetros entrenables en `best_model_{name}/exp2_state.pt`
                        + `exp2_meta.json` con la metadata para reconstruir/evaluar.
      • RESUME-AWARE  : recupera historial y best_val_loss si ya existían.
    """

    def __init__(self, config_name: str, output_dir: Path, checkpoints_dir: Path):
        self.config_name     = config_name
        self.output_dir      = output_dir
        self.checkpoints_dir = checkpoints_dir
        self.history_path    = output_dir / f"history_exp2_{config_name}.csv"
        self.best_model_dir  = checkpoints_dir / f"best_model_{config_name}"

        self.best_val_loss: float      = float("inf")
        self.history:       List[Dict] = []

        # Resume-aware: recuperar historial previo y mejor val_loss
        if self.history_path.exists():
            try:
                prev = pd.read_csv(self.history_path)
                self.history = prev.to_dict("records")
                valid = prev["Val_Loss"].dropna()
                if len(valid) > 0:
                    self.best_val_loss = float(valid.min())
                print(f"   [RESUME] Historial Exp2 recuperado: {len(self.history)} "
                      f"evals, best_val_loss={self.best_val_loss:.4f}")
            except Exception as exc:
                print(f"   [RESUME] No se pudo leer historial previo: {exc}")

    def _current_lrs(self, kwargs) -> Dict[str, Optional[float]]:
        """Extrae los LR actuales de los 2 grupos (ViT-LoRA / Puente) del optimizer."""
        lrs: Dict[str, Optional[float]] = {"LR_ViT": None, "LR_Bridge": None}
        opt = kwargs.get("optimizer", None)
        if opt is not None and getattr(opt, "param_groups", None):
            if len(opt.param_groups) >= 1:
                lrs["LR_ViT"] = opt.param_groups[0].get("lr")
            if len(opt.param_groups) >= 2:
                lrs["LR_Bridge"] = opt.param_groups[1].get("lr")
        return lrs

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return

        step     = state.global_step
        val_loss = (metrics or {}).get("eval_loss", None)

        # Último train_loss logueado (entre eval_steps)
        train_loss = None
        if state.log_history:
            for log in reversed(state.log_history):
                if "loss" in log and "eval_loss" not in log:
                    train_loss = log.get("loss")
                    break

        lrs = self._current_lrs(kwargs)

        # ── CSV de losses: persistido EN CADA EVALUACIÓN ────────────────────
        self.history.append({
            "Step":       step,
            "Train_Loss": train_loss,
            "Val_Loss":   val_loss,
            "LR_ViT":     lrs["LR_ViT"],
            "LR_Bridge":  lrs["LR_Bridge"],
        })
        pd.DataFrame(self.history).to_csv(self.history_path, index=False)

        train_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        val_str   = f"{val_loss:.4f}"   if val_loss   is not None else "N/A"
        print(f"\n   [Step {step}] train_loss={train_str}  val_loss={val_str}"
              f"  → CSV actualizado ({self.history_path.name})")

        # ── Checkpoint best-only: solo parámetros entrenables ───────────────
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_dir.mkdir(parents=True, exist_ok=True)

            trainable_state = {
                n: p.detach().cpu()
                for n, p in model.named_parameters()
                if p.requires_grad
            }
            state_path = self.best_model_dir / "exp2_state.pt"
            torch.save(trainable_state, str(state_path))

            # Metadata para reconstruir el modelo en inferencia/evaluación
            meta = {
                "config_name":      self.config_name,
                "experiment":       "exp2_raddino_puente",
                "step":             step,
                "val_loss":         float(val_loss),
                "param_count":      sum(t.numel() for t in trainable_state.values()),
                "trained_modules":  sorted({n.split(".", 1)[0] for n in trainable_state.keys()}),
                "base_model":       BASE_MODEL_NAME,
                "raddino_model":    RADDINO_MODEL_ID,
                "phase2_llm_dir":   str(PHASE2_LLM_DIR),
                "default_prompt":   DEFAULT_PROMPT,
                "saved_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            (self.best_model_dir / "exp2_meta.json").write_text(
                pd.Series(meta).to_json(indent=2), encoding="utf-8"
            )
            print(f"   ✓ Checkpoint best guardado (val_loss={val_loss:.4f}, "
                  f"{meta['param_count']:,} params)\n     → {state_path}")

        gc.collect()
        torch.cuda.empty_cache()
        model.train()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GRÁFICO DE CURVAS Train vs Val (igual a Fase 2/3)
# ─────────────────────────────────────────────────────────────────────────────

def plot_experiment2_curves(config_name: str, output_dir: Path, plots_dir: Path) -> None:
    """Genera 1 PNG: Train vs Val Loss para la corrida del Experimento 2."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"history_exp2_{config_name}.csv"
    if not history_path.exists():
        print(f"   [plot] Sin historial: {history_path}")
        return

    df = pd.read_csv(history_path)
    if "Train_Loss" not in df.columns or "Val_Loss" not in df.columns:
        print(f"   [plot] CSV sin columnas Train/Val Loss: {history_path}")
        return
    df = df.dropna(subset=["Train_Loss", "Val_Loss"])
    if df.empty:
        print(f"   [plot] CSV vacío tras dropna: {history_path}")
        return

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df["Step"], df["Train_Loss"], label="Train Loss",
            color="#2E86AB", linewidth=2, marker="o", markersize=4)
    ax.plot(df["Step"], df["Val_Loss"], label="Val Loss",
            color="#A23B72", linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Train vs Val Loss — {config_name} (Exp. 2 · RadDINO + Puente)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = plots_dir / f"01_train_val_loss_{config_name}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [plot] {out.name} guardado → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train_experiment2(
    processor,
    train_dataset,
    val_dataset,
    train_query_tokens: bool = True,
    fresh: bool = False,
    smoke: bool = False,
) -> Dict:
    """
    Entrena la arquitectura del Experimento 2 (RadDINO + puente).

    Flujo (idéntico en espíritu a train_puente_config de Fase 3):
        1. build_experiment_2_model()           → arquitectura + auditoría.
        2. build_experiment_2_optimizer()       → AdamW diferencial (2 grupos).
        3. TrainingArguments (bf16, cosine, warmup, save/eval por steps).
        4. Experiment2Trainer (guarda solo entrenables) + EarlyStopping + CSV cb.
        5. Reanuda desde el último checkpoint válido (salvo --fresh).
    """
    name = RUN_NAME
    print(f"\n{'='*80}")
    print(f"  EXPERIMENTO 2 — RadDINO + Puente (Q-Former + language_projection)")
    print(f"  Config       : {name}")
    print(f"  base_model   : {BASE_MODEL_NAME}")
    print(f"  RadDINO      : {RADDINO_MODEL_ID}")
    print(f"  LLM LoRA F2  : {PHASE2_LLM_DIR}")
    print(f"  lr_vit_lora={TRAINING_CONFIG['lr_vit_lora']:.0e}  "
          f"lr_bridge={TRAINING_CONFIG['lr_bridge']:.0e}  "
          f"batch_ef={TRAINING_CONFIG['batch_size']*TRAINING_CONFIG['gradient_accumulation']}  "
          f"epochs={TRAINING_CONFIG['max_epochs']}")
    print(f"{'='*80}\n")

    gc.collect()
    torch.cuda.empty_cache()

    # ── 1. Modelo (RadDINO + LoRA + proyector + puente Full FT) ──────────────
    model = build_experiment_2_model(train_query_tokens=train_query_tokens)

    # ── 2. Optimizador diferencial (ViT-LoRA / Puente) ───────────────────────
    optimizer, _ = build_experiment_2_optimizer(model)

    # ── 3. Directorio de salida de la corrida ────────────────────────────────
    config_output_dir = OUTPUT_DIR / name
    config_output_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. TrainingArguments (idénticas a Fase 3 salvo nombres) ──────────────
    max_steps = 1 if smoke else -1
    training_args = TrainingArguments(
        output_dir=str(config_output_dir),
        num_train_epochs=TRAINING_CONFIG["max_epochs"],
        max_steps=max_steps,                       # smoke=1 paso; -1 = ignora
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        per_device_eval_batch_size=24,
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation"],
        learning_rate=TRAINING_CONFIG["lr_bridge"],   # informativo; el optimizer fija los LR reales
        lr_scheduler_type="cosine",
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG["eval_steps"] if not smoke else 1,
        save_strategy="steps",
        save_steps=TRAINING_CONFIG["eval_steps"] if not smoke else 1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100 if not smoke else 1,
        logging_dir=str(config_output_dir / "logs"),
        fp16=False,
        bf16=True,
        gradient_checkpointing=False,              # tenemos VRAM para activaciones
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        group_by_length=True,
        dataloader_num_workers=0 if smoke else 32,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=None if smoke else 6,
        dataloader_persistent_workers=False,       # OBLIGATORIO (deadlock al resumir)
        dataloader_drop_last=True,
        bf16_full_eval=True,
        save_total_limit=2,
    )

    # ── 5. Callbacks y collator ──────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer, padding=True
    )
    loss_cb = Experiment2LossCallback(
        config_name     = name,
        output_dir      = OUTPUT_DIR,
        checkpoints_dir = CHECKPOINTS_DIR,
    )
    early_stop_cb = EarlyStoppingCallback(
        early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"]
    )

    # ── 6. Pre-computar longitudes (group_by_length sin I/O de imágenes) ─────
    print("   [lengths] Pre-computando longitudes de secuencia (train/val)...")
    train_lengths = precompute_train_lengths(train_dataset)
    eval_lengths  = precompute_train_lengths(val_dataset)

    trainer = Experiment2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        callbacks=[early_stop_cb, loss_cb],
        train_lengths=train_lengths,
        eval_lengths=eval_lengths,
    )

    # ── 7. Detectar checkpoint previo (rescate) ──────────────────────────────
    resume_ckpt = None
    if not fresh:
        all_ckpts = sorted(
            config_output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        )
        valid_ckpts = [c for c in all_ckpts if (c / "trainer_state.json").exists()]
        for inc in [c for c in all_ckpts if c not in valid_ckpts]:
            print(f"   [RESCATE] ⚠ Checkpoint incompleto descartado: {inc.name}")
        if valid_ckpts:
            resume_ckpt = str(valid_ckpts[-1])
            print(f"   [RESCATE] {len(valid_ckpts)} checkpoint(s) válido(s). "
                  f"Reanudando desde {valid_ckpts[-1].name}...")
    if resume_ckpt:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        print("   [NUEVO] Sin checkpoints válidos previos. Iniciando desde cero.")
        trainer.train()

    # ── 8. Resumen de la corrida ─────────────────────────────────────────────
    eval_logs  = [l for l in trainer.state.log_history if "eval_loss" in l]
    train_logs = [l for l in trainer.state.log_history
                  if "loss" in l and "eval_loss" not in l]
    best_val_loss = trainer.state.best_metric
    best_step = None
    if eval_logs and best_val_loss is not None:
        for log in eval_logs:
            if abs(log.get("eval_loss", float("inf")) - best_val_loss) < 1e-6:
                best_step = log.get("step")
                break

    result = {
        "config_name":      name,
        "experiment":       "exp2_raddino_puente",
        "lr_vit_lora":      TRAINING_CONFIG["lr_vit_lora"],
        "lr_bridge":        TRAINING_CONFIG["lr_bridge"],
        "total_steps":      trainer.state.global_step,
        "best_val_loss":    best_val_loss,
        "best_step":        best_step,
        "final_train_loss": train_logs[-1]["loss"] if train_logs else None,
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"\n   ✓ Completado: steps={result['total_steps']}  "
          f"best_val_loss={result['best_val_loss']}  @  step {result['best_step']}")

    # ── 9. Liberar VRAM ──────────────────────────────────────────────────────
    del model, trainer, loss_cb, early_stop_cb, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6.  AUDITORÍA DE ENTORNO + PRE-REQUISITOS
# ─────────────────────────────────────────────────────────────────────────────

def _audit_environment() -> None:
    """Auditoría de entorno y verificación de pre-requisitos."""
    print("\n" + "=" * 80)
    print("  ENTORNO — Experimento 2 (RadDINO + Puente)")
    print("=" * 80)
    print(f"  Hora           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PyTorch        : {torch.__version__}")
    print(f"  CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA           : {torch.version.cuda}")
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  base_model     : {BASE_MODEL_NAME}")
    print(f"  RadDINO        : {RADDINO_MODEL_ID}")
    print(f"  OUTPUT_DIR     : {OUTPUT_DIR}")
    print(f"  CHECKPOINTS    : {CHECKPOINTS_DIR}")
    print(f"  prompt diseñado: {DEFAULT_PROMPT!r}")
    print("=" * 80 + "\n")

    # Verificar que existe el adaptador LoRA del LLM (Fase 2)
    missing = []
    for path, label in [
        (PHASE2_LLM_DIR, "PHASE2_LLM_DIR (LLM LoRA)"),
        (PHASE2_LLM_DIR / "adapter_config.json", "LLM adapter_config.json"),
    ]:
        if not path.exists():
            missing.append((label, path))
    if missing:
        print("✗ ARCHIVOS FALTANTES (no se puede iniciar el Experimento 2):")
        for label, path in missing:
            print(f"    · {label}: {path}")
        sys.exit(1)
    print("✓ Pre-requisitos OK: adaptador LoRA del LLM (Fase 2) detectado.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrena el Experimento 2 (RadDINO + Puente Full-FT).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fresh", action="store_true",
                   help="Ignora checkpoints previos y entrena desde cero.")
    p.add_argument("--no-query-tokens", action="store_true",
                   help="Congela query_tokens (entrena solo qformer + language_projection + proj).")
    p.add_argument("--smoke", action="store_true",
                   help="Corre 1 solo paso para validar la tubería (sin entrenar de verdad).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _audit_environment()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Procesador (compartido entre datasets)
    print("→ Cargando procesador BLIP-2...")
    processor = cargar_procesador(BASE_MODEL_NAME)

    # Datasets del 50% con el PROMPT DISEÑADO (build_experiment_2_datasets ya lo fija)
    print("\n→ Cargando datasets (50 % de MIMIC-CXR, prompt diseñado)...")
    train_ds, val_ds = build_experiment_2_datasets(processor)

    # ── SMOKE TEST: recortar datasets a un puñado de muestras ────────────────
    # En --smoke el Trainer hace 1 paso y EVALÚA. Sin recorte, esa evaluación
    # correría sobre las 38,989 muestras de validación completas (≈30+ min
    # cargando imágenes + CLAHE mono-hilo con num_workers=0) — ese es el cuello
    # de botella observado. Truncamos .data a N filas para que el smoke valide
    # la TUBERÍA completa (build → step → eval → checkpoint → CSV) en ~1-2 min.
    if args.smoke:
        N_TRAIN_SMOKE, N_VAL_SMOKE = 48, 24
        train_ds.data = train_ds.data.head(N_TRAIN_SMOKE).reset_index(drop=True)
        val_ds.data   = val_ds.data.head(N_VAL_SMOKE).reset_index(drop=True)
        print(f"   [SMOKE] Datasets recortados: train={len(train_ds)}  "
              f"val={len(val_ds)} (evita la eval sobre 39k muestras).")

    # Entrenamiento
    print("\n→ Iniciando entrenamiento del Experimento 2...")
    result = train_experiment2(
        processor          = processor,
        train_dataset      = train_ds,
        val_dataset        = val_ds,
        train_query_tokens = not args.no_query_tokens,
        fresh              = args.fresh,
        smoke              = args.smoke,
    )

    # Persistir resumen incremental
    intermediate_csv = OUTPUT_DIR / "intermediate_results_exp2.csv"
    df_result = pd.DataFrame([result])
    if intermediate_csv.exists():
        prev = pd.read_csv(intermediate_csv)
        df_result = pd.concat([prev, df_result], ignore_index=True)
    df_result.to_csv(intermediate_csv, index=False)
    print(f"\n✓ Resultado guardado: {intermediate_csv}")

    # Graficar curvas Train vs Val
    print("\n→ Generando gráfico de curvas Train vs Val...")
    plot_experiment2_curves(
        config_name = RUN_NAME,
        output_dir  = OUTPUT_DIR,
        plots_dir   = plots_dir,
    )

    # CSV final
    final_csv = OUTPUT_DIR / "final_results_exp2.csv"
    df_result.to_csv(final_csv, index=False)
    print(f"✓ Resultado final: {final_csv}")

    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("  EXPERIMENTO 2 — ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print(f"  best_val_loss = {result['best_val_loss']}  @  step {result['best_step']}")
    print(f"  Pesos finales : {CHECKPOINTS_DIR / f'best_model_{RUN_NAME}'}")
    print(f"  History CSV   : {OUTPUT_DIR / f'history_exp2_{RUN_NAME}.csv'}")
    print(f"  Hora          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
