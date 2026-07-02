#!/usr/bin/env python3
"""
run_final_experiment.py — Lanzador EXPERIMENTO FINAL: RadDINO + Asclepius-R + Puente
====================================================================================
Entry point que ENTRENA la arquitectura del Experimento FINAL construida por
`trainer_utils_final_experiment.build_final_experiment_model`:

    vision_model  (RadDINO + LoRA Exp2, fusionada)  : CONGELADO (100%)
    language_model (Asclepius-R / LLaMA-7B clínico) : CONGELADO (100%), SIN LoRA
    qformer + language_projection (+ adaptador 2560→H_llm) + query_tokens : Full FT

Reutiliza EXACTAMENTE la maquinaria de entrenamiento del proyecto (idéntica a la
Fase 3 PUENTE / Experimento 2 / Experimento 3):
    • LengthAwareTrainer + precompute_train_lengths  (group_by_length sin I/O)
    • TrainingArguments (bf16, cosine, warmup, save/eval steps)  ← MISMOS hiperparámetros
    • EarlyStoppingCallback

⚠ CORRECCIÓN CRÍTICA: learning_rate = 1e-5 (NO 1e-4). El Q-Former Full-FT (~188M)
  con lr=1e-4 hacía REBOTAR la val_loss al salir del warmup (explosión de
  gradientes / destrucción del espacio latente). Ver TRAINING_CONFIG.

GARANTÍAS
---------
  ✓ PROMPT DISEÑADO   : se usa EXPLÍCITAMENTE config.inference.default_prompt
                        (DEFAULT_PROMPT). NO self.prompt ni prompt dinámico.
  ✓ GUARDA CHECKPOINTS: `FinalExperimentTrainer` persiste SOLO los parámetros
                        entrenables (Q-Former + language_projection + adaptador +
                        query_tokens) en `final_state.pt` (no el modelo completo).
  ✓ CSV DE LOSSES     : `FinalExperimentLossCallback` apendiza en CADA evaluación
                        una fila (Step | Train_Loss | Val_Loss | LR) a
                        `history_final_{name}.csv` y grafica las curvas al final.
  ✓ EVALUACIÓN AUTO   : al terminar, invoca AUTOMÁTICAMENTE las 4 ramas de
                        inferencia/evaluación con las funciones existentes
                        (CheXpert 14 patologías + BLEU-4 + ROUGE-L + RadGraph +
                        ClinicalBERT; reportes global y granular).

Uso (desde tmux)
----------------
    python run_final_experiment.py
    python run_final_experiment.py --fresh                # ignora checkpoints previos
    python run_final_experiment.py --no-query-tokens      # congela query_tokens
    python run_final_experiment.py --smoke                # 1 paso, valida la tubería
    python run_final_experiment.py --skip-eval            # entrena sin evaluación final
    python run_final_experiment.py --print-accion-obligatoria  # cómo obtener Asclepius-R

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
import subprocess
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

# Maquinaria reutilizada de Fase 2/3 (group_by_length sin I/O de imágenes)
from trainer_utils import LengthAwareTrainer, precompute_train_lengths

# Constructor + helpers del Experimento FINAL
from trainer_utils_final_experiment import (
    BASE_MODEL_NAME,
    ASCLEPIUS_R_DIR,
    ASCLEPIUS_VARIANT,
    EXP2_STATE_PT,
    DEFAULT_PROMPT,
    OUTPUT_DIR,
    CHECKPOINTS_DIR,
    TRAINING_CONFIG,
    build_final_experiment_model,
    build_final_experiment_optimizer,
    build_final_experiment_datasets,
    preparar_procesador_exp3,
    _asclepius_disponible,
)

warnings.filterwarnings("ignore")

# Nombre único de la corrida (define los nombres de checkpoints y CSV)
RUN_NAME: str = "Exp_Final"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MENSAJE "ACCION OBLIGATORIA" — cómo obtener Asclepius-R (acceso credenciado)
# ─────────────────────────────────────────────────────────────────────────────

def _mensaje_accion_obligatoria() -> str:
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          ⚠  ACCION OBLIGATORIA  ⚠                              ║
║         Descarga del modelo Asclepius-R (LLM clínico, acceso credenciado)      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Asclepius-R es el LLM clínico entrenado sobre notas REALES de MIMIC-III. Es de
ACCESO CREDENCIADO en PhysioNet (no se puede redistribuir ni descargar sin
credenciales) — por eso NO está en el repositorio y debes obtenerlo tú con tus
credenciales. NO uses la versión sintética pública (starmpcc/Asclepius-*) ni
cuantizaciones: el enunciado exige el modelo COMPLETO y la variante "-R".

──────────────────────────────────────────────────────────────────────────────
PRE-REQUISITOS (una sola vez, en tu cuenta de PhysioNet)
──────────────────────────────────────────────────────────────────────────────
  1) Ser usuario CREDENCIADO:        https://physionet.org/settings/credentialing/
  2) Completar el training CITI ("Data or Specimens Only Research").
  3) Firmar el DUA del recurso:      https://physionet.org/sign-dua/asclepius-r/1.1.0/
  Página del recurso:                https://physionet.org/content/asclepius-r/1.1.0/

──────────────────────────────────────────────────────────────────────────────
DESCARGA — destino: {ASCLEPIUS_R_DIR}
──────────────────────────────────────────────────────────────────────────────
  Variante por defecto: {ASCLEPIUS_VARIANT}  (LLaMA-7B, hidden=4096)
    mkdir -p "{ASCLEPIUS_R_DIR.parent}"
    cd "{ASCLEPIUS_R_DIR.parent}"
    wget -r -N -c -np --user TU_USUARIO_PHYSIONET --ask-password \\
      https://physionet.org/files/asclepius-r/1.1.0/{ASCLEPIUS_VARIANT}/
    # mueve la carpeta descargada a:  {ASCLEPIUS_R_DIR}

──────────────────────────────────────────────────────────────────────────────
VERIFICACIÓN (debe listar config.json + *.bin/*.safetensors + tokenizer)
──────────────────────────────────────────────────────────────────────────────
    ls -la "{ASCLEPIUS_R_DIR}"

Cuando el directorio anterior contenga el modelo, vuelve a ejecutar:
    python run_final_experiment.py
══════════════════════════════════════════════════════════════════════════════
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAINER ESPECIALIZADO — guarda SOLO los parámetros entrenables
# ─────────────────────────────────────────────────────────────────────────────

class FinalExperimentTrainer(LengthAwareTrainer):
    """
    Trainer que sobreescribe save_model / _load_from_checkpoint / _load_best_model
    para persistir ÚNICAMENTE los parámetros con requires_grad=True:
        Q-Former + language_projection (LP preentrenada + adaptador) + query_tokens.

    El modelo completo (con Asclepius-R + RadDINO) pesa decenas de GB; guardar solo
    los ~190M params entrenables del puente evita llenar el disco. Para reconstruir
    en inferencia: volver a llamar a build_final_experiment_model() y cargar este
    `final_state.pt` con strict=False.
    """

    STATE_FILENAME = "final_state.pt"

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

    def _load_final_from(self, ckpt_dir: str) -> None:
        path = os.path.join(ckpt_dir, self.STATE_FILENAME)
        if not os.path.isfile(path):
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"   [resume] WARN: {len(unexpected)} keys inesperadas en {self.STATE_FILENAME}")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if os.path.isfile(os.path.join(resume_from_checkpoint, self.STATE_FILENAME)):
            self._load_final_from(resume_from_checkpoint)
        else:
            super()._load_from_checkpoint(resume_from_checkpoint, model)

    def _load_best_model(self):
        ckpt = self.state.best_model_checkpoint
        if ckpt is None:
            return
        if os.path.isfile(os.path.join(ckpt, self.STATE_FILENAME)):
            self._load_final_from(ckpt)
        else:
            super()._load_best_model()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CALLBACK — CSV de losses (cada eval) + checkpoint best-only
# ─────────────────────────────────────────────────────────────────────────────

class FinalExperimentLossCallback(TrainerCallback):
    """
    Ejecutado tras cada evaluate(). Extrae la loss POR CADA eval_step y la persiste.

      • CSV de losses : apendiza  Step | Train_Loss | Val_Loss | LR  a
                        `history_final_{name}.csv` (persistido EN CADA eval).
      • Checkpoint best: cuando val_loss mejora, guarda el state_dict de SOLO los
                        parámetros entrenables en `best_model_{name}/final_state.pt`
                        + `final_meta.json` con la metadata para reconstruir/evaluar.
      • RESUME-AWARE  : recupera historial y best_val_loss si ya existían.
    """

    def __init__(self, config_name: str, output_dir: Path, checkpoints_dir: Path):
        self.config_name     = config_name
        self.output_dir      = output_dir
        self.checkpoints_dir = checkpoints_dir
        self.history_path    = output_dir / f"history_final_{config_name}.csv"
        self.best_model_dir  = checkpoints_dir / f"best_model_{config_name}"

        self.best_val_loss: float      = float("inf")
        self.history:       List[Dict] = []

        if self.history_path.exists():
            try:
                prev = pd.read_csv(self.history_path)
                self.history = prev.to_dict("records")
                valid = prev["Val_Loss"].dropna()
                if len(valid) > 0:
                    self.best_val_loss = float(valid.min())
                print(f"   [RESUME] Historial FINAL recuperado: {len(self.history)} "
                      f"evals, best_val_loss={self.best_val_loss:.4f}")
            except Exception as exc:
                print(f"   [RESUME] No se pudo leer historial previo: {exc}")

    def _current_lr(self, kwargs) -> Optional[float]:
        opt = kwargs.get("optimizer", None)
        if opt is not None and getattr(opt, "param_groups", None):
            return opt.param_groups[0].get("lr")
        return None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return

        step     = state.global_step
        val_loss = (metrics or {}).get("eval_loss", None)

        train_loss = None
        if state.log_history:
            for log in reversed(state.log_history):
                if "loss" in log and "eval_loss" not in log:
                    train_loss = log.get("loss")
                    break

        lr = self._current_lr(kwargs)

        self.history.append({
            "Step":       step,
            "Train_Loss": train_loss,
            "Val_Loss":   val_loss,
            "LR":         lr,
        })
        pd.DataFrame(self.history).to_csv(self.history_path, index=False)

        train_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        val_str   = f"{val_loss:.4f}"   if val_loss   is not None else "N/A"
        print(f"\n   [Step {step}] train_loss={train_str}  val_loss={val_str}"
              f"  → CSV actualizado ({self.history_path.name})")

        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_dir.mkdir(parents=True, exist_ok=True)

            trainable_state = {
                n: p.detach().cpu()
                for n, p in model.named_parameters()
                if p.requires_grad
            }
            state_path = self.best_model_dir / "final_state.pt"
            torch.save(trainable_state, str(state_path))

            meta = {
                "config_name":       self.config_name,
                "experiment":        "final_raddino_asclepius_puente",
                "step":              step,
                "val_loss":          float(val_loss),
                "param_count":       sum(t.numel() for t in trainable_state.values()),
                "trained_modules":   sorted({n.split(".", 1)[0] for n in trainable_state.keys()}),
                "base_model":        BASE_MODEL_NAME,
                "asclepius_dir":     str(ASCLEPIUS_R_DIR),
                "asclepius_variant": ASCLEPIUS_VARIANT,
                "raddino_weights":   str(EXP2_STATE_PT),
                "lr_bridge":         TRAINING_CONFIG["lr_bridge"],
                "default_prompt":    DEFAULT_PROMPT,
                "saved_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            (self.best_model_dir / "final_meta.json").write_text(
                pd.Series(meta).to_json(indent=2), encoding="utf-8"
            )
            print(f"   ✓ Checkpoint best guardado (val_loss={val_loss:.4f}, "
                  f"{meta['param_count']:,} params)\n     → {state_path}")

        gc.collect()
        torch.cuda.empty_cache()
        model.train()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GRÁFICO DE CURVAS Train vs Val
# ─────────────────────────────────────────────────────────────────────────────

def plot_final_experiment_curves(config_name: str, output_dir: Path, plots_dir: Path) -> None:
    """Genera 1 PNG: Train vs Val Loss para la corrida del Experimento FINAL."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"history_final_{config_name}.csv"
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
    ax.set_title(f"Train vs Val Loss — {config_name} (Exp. FINAL · RadDINO + Asclepius-R + Puente, lr=1e-5)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = plots_dir / f"01_train_val_loss_{config_name}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [plot] {out.name} guardado → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train_final_experiment(
    processor,
    train_dataset,
    val_dataset,
    train_query_tokens: bool = True,
    fresh: bool = False,
    smoke: bool = False,
) -> Dict:
    """Entrena la arquitectura del Experimento FINAL (RadDINO + Asclepius-R + puente)."""
    name = RUN_NAME
    print(f"\n{'='*80}")
    print(f"  EXPERIMENTO FINAL — RadDINO + Asclepius-R + Puente (Q-Former + language_projection)")
    print(f"  Config       : {name}")
    print(f"  base_model   : {BASE_MODEL_NAME}")
    print(f"  Asclepius-R  : {ASCLEPIUS_R_DIR}")
    print(f"  RadDINO wts  : {EXP2_STATE_PT}")
    print(f"  lr_bridge={TRAINING_CONFIG['lr_bridge']:.0e}  (⚠ CORREGIDO)  "
          f"batch_ef={TRAINING_CONFIG['batch_size']*TRAINING_CONFIG['gradient_accumulation']}  "
          f"epochs={TRAINING_CONFIG['max_epochs']}")
    print(f"{'='*80}\n")

    gc.collect()
    torch.cuda.empty_cache()

    # ── 1. Modelo (RadDINO congelado + Asclepius-R congelado + puente Full FT) ─
    model = build_final_experiment_model(
        train_query_tokens=train_query_tokens,
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
    )

    # ── 2. Optimizador (un solo grupo: el puente, lr=1e-5) ───────────────────
    optimizer, _ = build_final_experiment_optimizer(model)

    # ── 3. Directorio de salida de la corrida ────────────────────────────────
    config_output_dir = OUTPUT_DIR / name
    config_output_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. TrainingArguments (MISMOS hiperparámetros de control; lr=1e-5) ─────
    max_steps = 1 if smoke else -1
    training_args = TrainingArguments(
        output_dir=str(config_output_dir),
        num_train_epochs=TRAINING_CONFIG["max_epochs"],
        max_steps=max_steps,
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        per_device_eval_batch_size=24,   # eval forward-only + padding dinámico → mucho margen (=Exp2/Exp3)
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation"],
        learning_rate=TRAINING_CONFIG["lr_bridge"],   # ⚠ 1e-5 (CORREGIDO)
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
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        group_by_length=True,
        dataloader_num_workers=0 if smoke else 16,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=None if smoke else 4,
        dataloader_persistent_workers=False,       # OBLIGATORIO (deadlock al resumir)
        dataloader_drop_last=True,
        bf16_full_eval=True,
        save_total_limit=2,
    )

    # ── 5. Callbacks y collator ──────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer, padding=True
    )
    loss_cb = FinalExperimentLossCallback(
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

    trainer = FinalExperimentTrainer(
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
        "experiment":       "final_raddino_asclepius_puente",
        "lr_bridge":        TRAINING_CONFIG["lr_bridge"],
        "total_steps":      trainer.state.global_step,
        "best_val_loss":    best_val_loss,
        "best_step":        best_step,
        "final_train_loss": train_logs[-1]["loss"] if train_logs else None,
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"\n   ✓ Completado: steps={result['total_steps']}  "
          f"best_val_loss={result['best_val_loss']}  @  step {result['best_step']}")

    del model, trainer, loss_cb, early_stop_cb, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7.  EVALUACIÓN POST-ENTRENAMIENTO — invoca AUTOMÁTICAMENTE las 4 ramas
# ─────────────────────────────────────────────────────────────────────────────

def _run_subprocess(desc: str, cmd: List[str]) -> int:
    """Ejecuta un subproceso del MISMO intérprete, propagando stdout/stderr."""
    print(f"\n{'─'*80}\n  ▶ {desc}\n    $ {' '.join(cmd)}\n{'─'*80}")
    try:
        proc = subprocess.run(cmd, cwd=str(config.paths.base_dir))
        if proc.returncode != 0:
            print(f"  ⚠ '{desc}' devolvió código {proc.returncode}.")
        return proc.returncode
    except Exception as exc:
        print(f"  ✗ Error ejecutando '{desc}': {exc}")
        return 1


def evaluacion_post_entrenamiento(smoke: bool = False) -> None:
    """
    Invoca AUTOMÁTICAMENTE las 4 ramas de inferencia/evaluación con las funciones
    existentes (mismos GEN_PARAMS y métricas), sobre el checkpoint recién entrenado
    del Experimento FINAL:

      RAMA 1 — Prompt DISEÑADO (directa)        : evaluate_final_experiment.py
               → CheXpert (14 patologías) + BLEU-4 + ROUGE-L + RadGraph + ClinicalBERT.
      RAMA 2 — Prompt DISEÑADO (post-procesada) : vllm_formatter (Llama-3.1-8b) +
               reevaluar_sucio_vs_limpio → BLEU-4 + ROUGE-L + RadGraph + ClinicalBERT.
      RAMA 3 — Prompt NUEVO/CoT (directa)       : exp1_cot_inference_final_experiment.py
               → CheXpert (14) + BLEU-4 + ROUGE-L + RadGraph + ClinicalBERT.
      RAMA 4 — Prompt NUEVO/CoT (post-procesada): vllm_formatter (Llama-3.1-8b) +
               reevaluar_sucio_vs_limpio → BLEU-4 + ROUGE-L + RadGraph + ClinicalBERT.

    Las ramas 1 y 3 usan el modelo entrenado (BLIP-2/RadDINO/Asclepius). Las ramas
    2 y 4 cargan el LLM externo (vLLM) — se ejecutan después, en el mismo flujo.
    """
    py = sys.executable
    extra = ["--num-samples", "40"] if smoke else []

    print("\n" + "=" * 80)
    print("  EVALUACIÓN POST-ENTRENAMIENTO · EXPERIMENTO FINAL (4 ramas)")
    print("=" * 80)

    # ── RAMA 1 — Prompt diseñado (inferencia + 5 métricas, CheXpert por 14 patologías)
    _run_subprocess(
        "RAMA 1/4 · Prompt DISEÑADO — inferencia + CheXpert(14) + BLEU-4 + ROUGE-L + RadGraph + ClinicalBERT",
        [py, "evaluate_final_experiment.py", *extra],
    )

    # ── RAMA 3 — Prompt nuevo/CoT (inferencia + 5 métricas)
    _run_subprocess(
        "RAMA 3/4 · Prompt NUEVO (CoT) — inferencia + CheXpert(14) + BLEU-4 + ROUGE-L + RadGraph + ClinicalBERT",
        [py, "exp1_cot_inference_final_experiment.py", *extra],
    )

    # ── RAMAS 2 y 4 — post-procesado por LLM externo (vLLM Llama-3.1-8B) + re-eval
    eval_llmext = OUTPUT_DIR / "evaluation" / "predicciones_Exp_Final_llmexterno_llama8b.csv"
    cot_llmext  = OUTPUT_DIR / "exp1_cot" / "exp1_final_formatted_final_experiment_llama8b.csv"

    print("\n" + "=" * 80)
    print("  RAMAS 2 y 4 · Post-procesado por LLM EXTERNO (Llama-3.1-8B, vLLM)")
    print("=" * 80)
    # Una sola carga del modelo externo procesa ambas rutas (máxima eficiencia)
    rc_fmt = _run_subprocess(
        "RAMAS 2+4 · Formateo con LLM externo (Llama-3.1-8b) — diseñado + nuevo",
        [py, "vllm_formatter.py", "--tasks", "disenado_final", "nuevo_final"],
    )
    if rc_fmt == 0:
        # RAMA 2 — re-evaluación sucio vs limpio (prompt diseñado)
        _run_subprocess(
            "RAMA 2/4 · Re-evaluación sucio-vs-limpio (prompt diseñado, formateado)",
            [py, "reevaluar_sucio_vs_limpio.py", "--csv", str(eval_llmext),
             "--col-limpio", "formatted_report"],
        )
        # RAMA 4 — re-evaluación sucio vs limpio (prompt nuevo/CoT)
        _run_subprocess(
            "RAMA 4/4 · Re-evaluación sucio-vs-limpio (prompt nuevo/CoT, formateado)",
            [py, "reevaluar_sucio_vs_limpio.py", "--csv", str(cot_llmext),
             "--col-limpio", "formatted_report"],
        )
    else:
        print("  ⚠ El formateo con LLM externo no terminó OK; ramas 2/4 omitidas.")
        print("    Reintenta manualmente:")
        print(f"      {py} vllm_formatter.py --tasks disenado_final nuevo_final")
        print(f"      {py} reevaluar_sucio_vs_limpio.py --csv {eval_llmext} --col-limpio formatted_report")
        print(f"      {py} reevaluar_sucio_vs_limpio.py --csv {cot_llmext} --col-limpio formatted_report")

    print("\n" + "=" * 80)
    print("  EVALUACIÓN POST-ENTRENAMIENTO (4 ramas) FINALIZADA")
    print("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  AUDITORÍA DE ENTORNO + PRE-REQUISITOS
# ─────────────────────────────────────────────────────────────────────────────

def _audit_environment() -> None:
    print("\n" + "=" * 80)
    print("  ENTORNO — Experimento FINAL (RadDINO + Asclepius-R + Puente)")
    print("=" * 80)
    print(f"  Hora           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PyTorch        : {torch.__version__}")
    print(f"  CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA           : {torch.version.cuda}")
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  base_model     : {BASE_MODEL_NAME}")
    print(f"  Asclepius-R    : {ASCLEPIUS_R_DIR}  (variante {ASCLEPIUS_VARIANT})")
    print(f"  RadDINO weights: {EXP2_STATE_PT}")
    print(f"  OUTPUT_DIR     : {OUTPUT_DIR}")
    print(f"  lr_bridge      : {TRAINING_CONFIG['lr_bridge']:.0e}  (⚠ CORREGIDO de 1e-4)")
    print(f"  prompt diseñado: {DEFAULT_PROMPT!r}")
    print("=" * 80 + "\n")

    missing = []
    if not EXP2_STATE_PT.exists():
        missing.append(("RadDINO weights (exp2_state.pt)", EXP2_STATE_PT))
    if missing:
        print("✗ ARCHIVOS FALTANTES (no se puede iniciar el Experimento FINAL):")
        for label, path in missing:
            print(f"    · {label}: {path}")
        sys.exit(1)

    # Asclepius-R: pre-requisito de acceso credenciado → ACCION OBLIGATORIA
    if not _asclepius_disponible(ASCLEPIUS_R_DIR):
        print(_mensaje_accion_obligatoria())
        sys.exit(2)

    print("✓ Pre-requisitos OK: RadDINO (exp2_state.pt) + Asclepius-R detectados.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrena el Experimento FINAL (RadDINO + Asclepius-R + Puente Full-FT, lr=1e-5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fresh", action="store_true",
                   help="Ignora checkpoints previos y entrena desde cero.")
    p.add_argument("--no-query-tokens", action="store_true",
                   help="Congela query_tokens (entrena solo qformer + language_projection).")
    p.add_argument("--smoke", action="store_true",
                   help="Corre 1 solo paso para validar la tubería (sin entrenar de verdad).")
    p.add_argument("--skip-eval", action="store_true",
                   help="No ejecutar la evaluación post-entrenamiento.")
    p.add_argument("--print-accion-obligatoria", action="store_true",
                   help="Imprime cómo obtener Asclepius-R (acceso credenciado) y termina.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.print_accion_obligatoria:
        print(_mensaje_accion_obligatoria())
        return 0

    _audit_environment()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Procesador BLIP-2 con el TOKENIZER de Asclepius (intercambio obligatorio)
    print("→ Cargando procesador BLIP-2 + tokenizer Asclepius-R...")
    processor = cargar_procesador(BASE_MODEL_NAME)
    processor = preparar_procesador_exp3(processor, ASCLEPIUS_R_DIR)

    # Datasets del 50% con el PROMPT DISEÑADO (explícito)
    print("\n→ Cargando datasets (50 % de MIMIC-CXR, prompt diseñado)...")
    train_ds, val_ds = build_final_experiment_datasets(processor)

    if args.smoke:
        N_TRAIN_SMOKE, N_VAL_SMOKE = 48, 24
        train_ds.data = train_ds.data.head(N_TRAIN_SMOKE).reset_index(drop=True)
        val_ds.data   = val_ds.data.head(N_VAL_SMOKE).reset_index(drop=True)
        print(f"   [SMOKE] Datasets recortados: train={len(train_ds)}  val={len(val_ds)}.")

    print("\n→ Iniciando entrenamiento del Experimento FINAL...")
    result = train_final_experiment(
        processor          = processor,
        train_dataset      = train_ds,
        val_dataset        = val_ds,
        train_query_tokens = not args.no_query_tokens,
        fresh              = args.fresh,
        smoke              = args.smoke,
    )

    # Persistir resumen incremental
    intermediate_csv = OUTPUT_DIR / "intermediate_results_final.csv"
    df_result = pd.DataFrame([result])
    if intermediate_csv.exists():
        prev = pd.read_csv(intermediate_csv)
        df_result = pd.concat([prev, df_result], ignore_index=True)
    df_result.to_csv(intermediate_csv, index=False)
    print(f"\n✓ Resultado guardado: {intermediate_csv}")

    # Graficar curvas Train vs Val
    print("\n→ Generando gráfico de curvas Train vs Val...")
    plot_final_experiment_curves(RUN_NAME, OUTPUT_DIR, plots_dir)

    final_csv = OUTPUT_DIR / "final_results_final_experiment.csv"
    df_result.to_csv(final_csv, index=False)
    print(f"✓ Resultado final: {final_csv}")

    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("  EXPERIMENTO FINAL — ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print(f"  best_val_loss = {result['best_val_loss']}  @  step {result['best_step']}")
    print(f"  Pesos finales : {CHECKPOINTS_DIR / f'best_model_{RUN_NAME}'}")
    print(f"  History CSV   : {OUTPUT_DIR / f'history_final_{RUN_NAME}.csv'}")
    print("=" * 80 + "\n")

    # ── Evaluación post-entrenamiento (4 ramas automáticas) ──────────────────
    if not args.skip_eval and not args.smoke:
        evaluacion_post_entrenamiento(smoke=False)
    elif args.smoke:
        print("   [SMOKE] Evaluación post-entrenamiento omitida en modo smoke.")
    else:
        print("   [--skip-eval] Evaluación post-entrenamiento omitida por el usuario.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
