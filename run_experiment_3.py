#!/usr/bin/env python3
"""
run_experiment_3.py — Lanzador Experimento 3 (Fase 5): ViT(Fase 3) + Asclepius-R + Puente
==========================================================================================
Entry point que ENTRENA la arquitectura del Experimento 3 construida por
`trainer_utils_exp3_asclepius.build_experiment_3_model`:

    vision_model (EVA-CLIP + LoRA Fase 3) : CONGELADO (100%)
    language_model (OPT-2.7b)             : SUSTITUIDO por Asclepius-R (LLaMA clínico),
                                            CONGELADO (100%), SIN LoRA
    qformer + language_projection (+ adaptador 2560→H_llm) + query_tokens : Full FT (SIN LoRA)

Reutiliza la maquinaria de entrenamiento del proyecto (idéntico a Fase 3 PUENTE /
Experimento 2):
    • LengthAwareTrainer + precompute_train_lengths  (group_by_length sin I/O)
    • TrainingArguments (bf16, cosine, warmup, save/eval steps)
    • EarlyStoppingCallback

GARANTÍAS
---------
  ✓ PROMPT DISEÑADO   : se usa EXCLUSIVAMENTE config.inference.default_prompt
                        (DEFAULT_PROMPT). NO se usa self.prompt ni prompt dinámico.
  ✓ GUARDA CHECKPOINTS: `Experiment3Trainer` persiste SOLO los parámetros
                        entrenables (Q-Former + language_projection + adaptador +
                        query_tokens) en `exp3_state.pt` (no el modelo completo).
  ✓ CSV DE LOSSES     : `Experiment3LossCallback` apendiza en CADA evaluación una
                        fila (Step | Train_Loss | Val_Loss | LR) a
                        `history_exp3_{name}.csv` y grafica las curvas al final.
  ✓ EVALUACIÓN AUTO   : al terminar, replica las inferencias/evaluaciones de la
                        fase anterior (prompt diseñado + prompt nuevo/CoT, con
                        CheXpert por 14 patologías + BLEU-4 + ROUGE-L + RadGraph +
                        ClinicalBERT) usando las funciones existentes.

Uso (desde tmux)
----------------
    python run_experiment_3.py
    python run_experiment_3.py --fresh                 # ignora checkpoints previos
    python run_experiment_3.py --no-query-tokens       # congela query_tokens
    python run_experiment_3.py --smoke                 # 1 paso, valida la tubería
    python run_experiment_3.py --skip-eval             # entrena sin evaluación final
    python run_experiment_3.py --print-accion-obligatoria   # cómo obtener Asclepius-R

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

# Constructor + helpers del Experimento 3
from trainer_utils_exp3_asclepius import (
    BASE_MODEL_NAME,
    ASCLEPIUS_R_DIR,
    ASCLEPIUS_VARIANT,
    PHASE2_VIT_DIR,
    DEFAULT_PROMPT,
    OUTPUT_DIR,
    CHECKPOINTS_DIR,
    TRAINING_CONFIG,
    build_experiment_3_model,
    build_experiment_3_optimizer,
    build_experiment_3_datasets,
    preparar_procesador_exp3,
    _asclepius_disponible,
)

warnings.filterwarnings("ignore")

# Nombre único de la corrida (define los nombres de checkpoints y CSV)
RUN_NAME: str = "Exp3_Asclepius"


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
DESCARGA (elige UNA opción) — destino: {ASCLEPIUS_R_DIR.parent}/
──────────────────────────────────────────────────────────────────────────────
  Variante por defecto de este script: {ASCLEPIUS_VARIANT}  (LLaMA-7B, hidden=4096)
  (Para usar 13B: exporta  ASCLEPIUS_VARIANT=Asclepius-R-13B  antes de ejecutar,
   o coloca los pesos en  {ASCLEPIUS_R_DIR.parent}/Asclepius-R-13B)

  OPCIÓN A · wget (PhysioNet, recursivo):
    mkdir -p "{ASCLEPIUS_R_DIR.parent}"
    cd "{ASCLEPIUS_R_DIR.parent}"
    wget -r -N -c -np --user TU_USUARIO_PHYSIONET --ask-password \\
      https://physionet.org/files/asclepius-r/1.1.0/{ASCLEPIUS_VARIANT}/
    # mueve la carpeta descargada a:  {ASCLEPIUS_R_DIR}

  OPCIÓN B · ya tienes los .bin/.safetensors + config.json + tokenizer:
    Cópialos directamente a:  {ASCLEPIUS_R_DIR}

──────────────────────────────────────────────────────────────────────────────
VERIFICACIÓN (debe listar config.json + *.bin/*.safetensors + tokenizer)
──────────────────────────────────────────────────────────────────────────────
    ls -la "{ASCLEPIUS_R_DIR}"

Cuando el directorio anterior contenga el modelo, vuelve a ejecutar:
    python run_experiment_3.py
══════════════════════════════════════════════════════════════════════════════
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAINER ESPECIALIZADO — guarda SOLO los parámetros entrenables
# ─────────────────────────────────────────────────────────────────────────────

class Experiment3Trainer(LengthAwareTrainer):
    """
    Trainer que sobreescribe save_model / _load_from_checkpoint / _load_best_model
    para persistir ÚNICAMENTE los parámetros con requires_grad=True:
        Q-Former + language_projection (LP preentrenada + adaptador) + query_tokens.

    El modelo completo (con Asclepius-R) pesa decenas de GB; guardar solo los
    ~190M params entrenables del puente evita llenar el disco y el problema de
    pesos atados del LLM. Para reconstruir en inferencia: volver a llamar a
    build_experiment_3_model() y cargar este `exp3_state.pt` con strict=False.
    """

    STATE_FILENAME = "exp3_state.pt"

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

    def _load_exp3_from(self, ckpt_dir: str) -> None:
        path = os.path.join(ckpt_dir, self.STATE_FILENAME)
        if not os.path.isfile(path):
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"   [resume] WARN: {len(unexpected)} keys inesperadas en {self.STATE_FILENAME}")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if os.path.isfile(os.path.join(resume_from_checkpoint, self.STATE_FILENAME)):
            self._load_exp3_from(resume_from_checkpoint)
        else:
            super()._load_from_checkpoint(resume_from_checkpoint, model)

    def _load_best_model(self):
        ckpt = self.state.best_model_checkpoint
        if ckpt is None:
            return
        if os.path.isfile(os.path.join(ckpt, self.STATE_FILENAME)):
            self._load_exp3_from(ckpt)
        else:
            super()._load_best_model()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CALLBACK — CSV de losses (cada eval) + checkpoint best-only
# ─────────────────────────────────────────────────────────────────────────────

class Experiment3LossCallback(TrainerCallback):
    """
    Ejecutado tras cada evaluate(). Análogo a Experiment2LossCallback pero con un
    único grupo de LR (el puente).

      • CSV de losses : apendiza  Step | Train_Loss | Val_Loss | LR  a
                        `history_exp3_{name}.csv` (persistido EN CADA eval).
      • Checkpoint best: cuando val_loss mejora, guarda el state_dict de SOLO los
                        parámetros entrenables en `best_model_{name}/exp3_state.pt`
                        + `exp3_meta.json` con la metadata para reconstruir/evaluar.
      • RESUME-AWARE  : recupera historial y best_val_loss si ya existían.
    """

    def __init__(self, config_name: str, output_dir: Path, checkpoints_dir: Path):
        self.config_name     = config_name
        self.output_dir      = output_dir
        self.checkpoints_dir = checkpoints_dir
        self.history_path    = output_dir / f"history_exp3_{config_name}.csv"
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
                print(f"   [RESUME] Historial Exp3 recuperado: {len(self.history)} "
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
            state_path = self.best_model_dir / "exp3_state.pt"
            torch.save(trainable_state, str(state_path))

            meta = {
                "config_name":     self.config_name,
                "experiment":      "exp3_asclepius_puente",
                "step":            step,
                "val_loss":        float(val_loss),
                "param_count":     sum(t.numel() for t in trainable_state.values()),
                "trained_modules": sorted({n.split(".", 1)[0] for n in trainable_state.keys()}),
                "base_model":      BASE_MODEL_NAME,
                "asclepius_dir":   str(ASCLEPIUS_R_DIR),
                "asclepius_variant": ASCLEPIUS_VARIANT,
                "vit_lora_dir":    str(PHASE2_VIT_DIR),
                "default_prompt":  DEFAULT_PROMPT,
                "saved_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            (self.best_model_dir / "exp3_meta.json").write_text(
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

def plot_experiment3_curves(config_name: str, output_dir: Path, plots_dir: Path) -> None:
    """Genera 1 PNG: Train vs Val Loss para la corrida del Experimento 3."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"history_exp3_{config_name}.csv"
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
    ax.set_title(f"Train vs Val Loss — {config_name} (Exp. 3 · ViT F3 + Asclepius-R + Puente)",
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

def train_experiment3(
    processor,
    train_dataset,
    val_dataset,
    train_query_tokens: bool = True,
    fresh: bool = False,
    smoke: bool = False,
) -> Dict:
    """Entrena la arquitectura del Experimento 3 (ViT F3 + Asclepius-R + puente)."""
    name = RUN_NAME
    print(f"\n{'='*80}")
    print(f"  EXPERIMENTO 3 — ViT(Fase 3) + Asclepius-R + Puente (Q-Former + language_projection)")
    print(f"  Config       : {name}")
    print(f"  base_model   : {BASE_MODEL_NAME}")
    print(f"  Asclepius-R  : {ASCLEPIUS_R_DIR}")
    print(f"  ViT LoRA F3  : {PHASE2_VIT_DIR}")
    print(f"  lr_bridge={TRAINING_CONFIG['lr_bridge']:.0e}  "
          f"batch_ef={TRAINING_CONFIG['batch_size']*TRAINING_CONFIG['gradient_accumulation']}  "
          f"epochs={TRAINING_CONFIG['max_epochs']}")
    print(f"{'='*80}\n")

    gc.collect()
    torch.cuda.empty_cache()

    # ── 1. Modelo (ViT F3 congelado + Asclepius-R congelado + puente Full FT) ─
    model = build_experiment_3_model(
        train_query_tokens=train_query_tokens,
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
    )

    # ── 2. Optimizador (un solo grupo: el puente) ────────────────────────────
    optimizer, _ = build_experiment_3_optimizer(model)

    # ── 3. Directorio de salida de la corrida ────────────────────────────────
    config_output_dir = OUTPUT_DIR / name
    config_output_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. TrainingArguments (idénticas a Fase 3 salvo batch/GC para LLaMA-7B) ─
    max_steps = 1 if smoke else -1
    training_args = TrainingArguments(
        output_dir=str(config_output_dir),
        num_train_epochs=TRAINING_CONFIG["max_epochs"],
        max_steps=max_steps,
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        per_device_eval_batch_size=24,   # eval forward-only (sin backward/grad/optim) + padding dinmico  mucho margen; 24 = Exp2, ~2x ms rpido
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation"],
        learning_rate=TRAINING_CONFIG["lr_bridge"],
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
    loss_cb = Experiment3LossCallback(
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

    trainer = Experiment3Trainer(
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
        "experiment":       "exp3_asclepius_puente",
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
# 7.  EVALUACIÓN POST-ENTRENAMIENTO (replica las 4 rutas de la fase anterior)
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
    Replica AUTOMÁTICAMENTE las inferencias/evaluaciones de la fase anterior con
    las funciones existentes (mismos GEN_PARAMS y métricas), sobre el checkpoint
    recién entrenado del Experimento 3:

      RUTA 1 — Prompt DISEÑADO  : evaluate_exp3_asclepius.py
               → CheXpert (14 patologías, GT vs generado) + BLEU-4 + ROUGE-L
                 + F1-RadGraph + Clinical BERTScore  (granular + global CSV).
      RUTA 3 — Prompt NUEVO/CoT : exp1_cot_inference_exp3_asclepius.py
               → mismas 5 métricas con el prompt Chain-of-Thought.

    RUTAS 2 y 4 (post-procesado por LLM EXTERNO Llama-3.1-8B): requieren cargar el
    LLM externo y se ejecutan como jobs separados (vLLM) — se imprimen los
    comandos exactos al final (flujo establecido del proyecto).
    """
    py = sys.executable
    extra = ["--num-samples", "40"] if smoke else []

    print("\n" + "=" * 80)
    print("  EVALUACIÓN POST-ENTRENAMIENTO · EXPERIMENTO 3 (rutas con el modelo)")
    print("=" * 80)

    # RUTA 1 — Prompt diseñado (inferencia + 5 métricas, CheXpert por 14 patologías)
    _run_subprocess(
        "RUTA 1/4 · Prompt DISEÑADO — inferencia + CheXpert(14) + BLEU-4 + ROUGE-L + RadGraph + ClinicalBERT",
        [py, "evaluate_exp3_asclepius.py", *extra],
    )

    # RUTA 3 — Prompt nuevo/CoT (inferencia + 5 métricas)
    _run_subprocess(
        "RUTA 3/4 · Prompt NUEVO (CoT) — inferencia + CheXpert(14) + BLEU-4 + ROUGE-L + RadGraph + ClinicalBERT",
        [py, "exp1_cot_inference_exp3_asclepius.py", *extra],
    )

    # RUTAS 2 y 4 — post-procesado por LLM externo (vLLM Llama-3.1-8B) → re-evaluación
    eval_llmext = OUTPUT_DIR / "evaluation" / "predicciones_Exp3_Asclepius_llmexterno_llama8b.csv"
    cot_llmext  = OUTPUT_DIR / "exp1_cot" / "exp1_final_formatted_exp3_asclepius_llama8b.csv"
    print("\n" + "=" * 80)
    print("  RUTAS 2 y 4 · Post-procesado por LLM EXTERNO (Llama-3.1-8B) — ejecutar aparte")
    print("=" * 80)
    print("  Estas rutas cargan el LLM externo (vLLM) y se corren como jobs propios:")
    print("    # RUTA 2 (prompt diseñado, formateado por LLM externo) + re-eval:")
    print("    bash run_vllm_formatter_tmux.sh --tasks=disenado_exp3asclepius")
    print(f"    python reevaluar_sucio_vs_limpio.py --csv {eval_llmext}")
    print("    # RUTA 4 (prompt nuevo/CoT, formateado por LLM externo) + re-eval:")
    print("    bash run_vllm_formatter_tmux.sh --tasks=nuevo_exp3asclepius")
    print(f"    python reevaluar_sucio_vs_limpio.py --csv {cot_llmext}")
    print("    # Comparativos finales (4 variantes Exp3 + Fase 3 + Exp2):")
    print("    python actualizar_comparativos_exp3.py")
    print("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  AUDITORÍA DE ENTORNO + PRE-REQUISITOS
# ─────────────────────────────────────────────────────────────────────────────

def _audit_environment() -> None:
    print("\n" + "=" * 80)
    print("  ENTORNO — Experimento 3 (ViT F3 + Asclepius-R + Puente)")
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
    print(f"  ViT LoRA F3    : {PHASE2_VIT_DIR}")
    print(f"  OUTPUT_DIR     : {OUTPUT_DIR}")
    print(f"  prompt diseñado: {DEFAULT_PROMPT!r}")
    print("=" * 80 + "\n")

    missing = []
    if not (PHASE2_VIT_DIR / "adapter_config.json").exists():
        missing.append(("ViT LoRA F3 adapter_config.json", PHASE2_VIT_DIR / "adapter_config.json"))
    if missing:
        print("✗ ARCHIVOS FALTANTES (no se puede iniciar el Experimento 3):")
        for label, path in missing:
            print(f"    · {label}: {path}")
        sys.exit(1)

    # Asclepius-R: pre-requisito de acceso credenciado → ACCION OBLIGATORIA
    if not _asclepius_disponible(ASCLEPIUS_R_DIR):
        print(_mensaje_accion_obligatoria())
        sys.exit(2)

    print("✓ Pre-requisitos OK: ViT LoRA (Fase 3) + Asclepius-R detectados.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrena el Experimento 3 (ViT Fase 3 + Asclepius-R + Puente Full-FT).",
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

    # Datasets del 50% con el PROMPT DISEÑADO
    print("\n→ Cargando datasets (50 % de MIMIC-CXR, prompt diseñado)...")
    train_ds, val_ds = build_experiment_3_datasets(processor)

    if args.smoke:
        N_TRAIN_SMOKE, N_VAL_SMOKE = 48, 24
        train_ds.data = train_ds.data.head(N_TRAIN_SMOKE).reset_index(drop=True)
        val_ds.data   = val_ds.data.head(N_VAL_SMOKE).reset_index(drop=True)
        print(f"   [SMOKE] Datasets recortados: train={len(train_ds)}  val={len(val_ds)}.")

    print("\n→ Iniciando entrenamiento del Experimento 3...")
    result = train_experiment3(
        processor          = processor,
        train_dataset      = train_ds,
        val_dataset        = val_ds,
        train_query_tokens = not args.no_query_tokens,
        fresh              = args.fresh,
        smoke              = args.smoke,
    )

    # Persistir resumen incremental
    intermediate_csv = OUTPUT_DIR / "intermediate_results_exp3.csv"
    df_result = pd.DataFrame([result])
    if intermediate_csv.exists():
        prev = pd.read_csv(intermediate_csv)
        df_result = pd.concat([prev, df_result], ignore_index=True)
    df_result.to_csv(intermediate_csv, index=False)
    print(f"\n✓ Resultado guardado: {intermediate_csv}")

    # Graficar curvas Train vs Val
    print("\n→ Generando gráfico de curvas Train vs Val...")
    plot_experiment3_curves(RUN_NAME, OUTPUT_DIR, plots_dir)

    final_csv = OUTPUT_DIR / "final_results_exp3.csv"
    df_result.to_csv(final_csv, index=False)
    print(f"✓ Resultado final: {final_csv}")

    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("  EXPERIMENTO 3 — ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    print(f"  best_val_loss = {result['best_val_loss']}  @  step {result['best_step']}")
    print(f"  Pesos finales : {CHECKPOINTS_DIR / f'best_model_{RUN_NAME}'}")
    print(f"  History CSV   : {OUTPUT_DIR / f'history_exp3_{RUN_NAME}.csv'}")
    print("=" * 80 + "\n")

    # ── Evaluación post-entrenamiento (rutas con el modelo) ──────────────────
    if not args.skip_eval and not args.smoke:
        evaluacion_post_entrenamiento(smoke=False)
    elif args.smoke:
        print("   [SMOKE] Evaluación post-entrenamiento omitida en modo smoke.")
    else:
        print("   [--skip-eval] Evaluación post-entrenamiento omitida por el usuario.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
