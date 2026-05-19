"""
Entrenamiento Gran Final - 2 Configuraciones LoRA con Early Stopping por Steps
===============================================================================

Objetivos implementados:

OBJETIVO 1 - Infraestructura:
  - cv2.setNumThreads(0) antes de importar torch (previene deadlocks)
  - DataLoader con num_workers=4 y pin_memory=True
  - Aislamiento total por config: Trainer nuevo -> global_step y epoch se
    resetean a 0 en cada config (evita bug de duplicacion de epocas)
  - max_epochs = 50 (limite fijo por config)

OBJETIVO 2 - Prompting dinamico:
  - MimicCXRDataset._mapear_vista(): PA/AP -> Frontal, LL/LATERAL -> Lateral
  - Prompt: "[Context: {vista} view] {config.inference.default_prompt}"
  - max_length=512, truncation=True en el procesador

OBJETIVO 3 - Entrenamiento y acumulacion:
  - gradient_accumulation_steps=4 -> batch efectivo = 16
  - eval_strategy="steps", eval_steps=500
  - EarlyStoppingCallback: patience=10 evaluaciones consecutivas sin mejora

OBJETIVO 4 - Validacion y Checkpoints:
  - Val_loss calculada sobre el set completo de validacion
  - Checkpoint guardado SOLO cuando val_loss mejora (best_model_Config_X)
  - max_new_tokens=400 en model.generate()

OBJETIVO 5 - Auditoria ligera (10 muestras por eval):
  - history_final_Config_X.csv: Step, Val_Loss, BLEU-4, ROUGE-L
  - audit_final_Config_X.txt: Step | Vista | Prediccion | Real | BLEU-4 | ROUGE-L
    (sin truncar strings)

OBJETIVO 6 - Graficos finales:
  - 01_val_loss.png, 02_bleu4.png, 03_rougeL.png (ambas configs en cada figura)

Autor: Evelyn Silva Rozas
Fecha: Mayo 2026
"""

import os
import gc
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# OBJETIVO 1: cv2 DEBE importarse y desactivarse ANTES de torch/DataLoader
# para prevenir deadlocks entre OpenCV y los workers de PyTorch.
import cv2
cv2.setNumThreads(0)

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, get_peft_model

# Metricas NLP
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from config import config
from data_loader import MimicCXRDataset
from model_utils import cargar_procesador, cargar_modelo_base

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# CONFIGURACION GLOBAL
# Config A: r=16, alpha=32  (ganadora semifinal Config_4)
# Config B: r=32, alpha=64  (ganadora semifinal Config_6)
# ============================================================================

FINAL_CONFIGS = [
    {'name': 'Config_A', 'r': 16, 'alpha': 32},
    {'name': 'Config_B', 'r': 32, 'alpha': 64},
]

TRAINING_CONFIG = {
    'lr':                      1e-4,
    'max_epochs':              50,    # OBJETIVO 1: limite fijo por config
    'early_stopping_patience': 10,    # OBJETIVO 4: patience en nº de evaluaciones
    'batch_size':              4,
    'gradient_accumulation':   4,     # OBJETIVO 3: batch efectivo = 16
    'eval_steps':              500,   # OBJETIVO 4: evaluar cada 500 steps
    'audit_samples':           10,    # OBJETIVO 5: muestras para audit/metricas
    'max_new_tokens':          400,   # OBJETIVO 4: generacion completa
}

OUTPUT_DIR      = config.paths.base_dir / "final_results"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

print("\n" + "="*80)
print("ENTRENAMIENTO GRAN FINAL - 2 CONFIGURACIONES LoRA")
print("="*80)
print("   Config A: r=16, alpha=32  |  Config B: r=32, alpha=64")
print(f"   max_epochs={TRAINING_CONFIG['max_epochs']}  |  "
      f"early_stop patience={TRAINING_CONFIG['early_stopping_patience']} evals  |  "
      f"eval_steps={TRAINING_CONFIG['eval_steps']}")
print(f"   batch={TRAINING_CONFIG['batch_size']} x grad_accum={TRAINING_CONFIG['gradient_accumulation']}"
      f" -> batch_efectivo={TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['gradient_accumulation']}")
print(f"   max_new_tokens={TRAINING_CONFIG['max_new_tokens']}  |  "
      f"audit_samples={TRAINING_CONFIG['audit_samples']}")
print(f"   Resultados en: {OUTPUT_DIR}")
print("="*80 + "\n")

# ============================================================================
# UTILIDADES DE METRICAS NLP
# ============================================================================

_rouge_scorer_inst = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
_smoothing         = SmoothingFunction()


def _bleu4(reference: str, candidate: str) -> float:
    """Calcula BLEU-4 con smoothing method1."""
    try:
        return sentence_bleu(
            [reference.lower().split()],
            candidate.lower().split(),
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=_smoothing.method1,
        )
    except Exception:
        return 0.0


def _rougeL(reference: str, candidate: str) -> float:
    """Calcula ROUGE-L F-measure."""
    return _rouge_scorer_inst.score(reference, candidate)['rougeL'].fmeasure


def generate_sample(model, sample: dict, processor, device, max_new_tokens: int) -> str:
    """
    Genera texto para una muestra del dataset usando pixel_values.
    Retorna el string decodificado (sin tokens especiales).
    """
    pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=1,    # greedy para velocidad en validacion
            use_cache=False     # <--- ¡AQUÍ ESTÁ LA SOLUCIÓN! (Evita el choque con checkpointing)    
        )
    return processor.batch_decode(out, skip_special_tokens=True)[0]


# ============================================================================
# CALLBACK DE AUDITORIA, METRICAS Y CHECKPOINTS
# ============================================================================

class FinalAuditCallback(TrainerCallback):
    """
    Callback ejecutado tras cada evaluacion del Trainer (cada eval_steps=500).

    OBJETIVO 4 — Checkpoints:
      Guarda pesos en checkpoints/best_model_Config_X SOLO si val_loss mejora.
      No se guarda 'latest_model' para no desperdiciar VRAM/disco.

    OBJETIVO 5 — Auditoria ligera:
      - Selecciona 10 muestras aleatorias del val set.
      - Genera reporte con max_new_tokens=400.
      - Calcula BLEU-4 y ROUGE-L por muestra y como promedio.
      - Appends a history_final_Config_X.csv: Step, Val_Loss, BLEU-4, ROUGE-L
      - Appends a audit_final_Config_X.txt:
          Step | Vista | Prediccion | Real | BLEU-4 | ROUGE-L
          (strings completos, sin truncar)
    """

    def __init__(
        self,
        val_dataset:     MimicCXRDataset,
        processor,
        config_name:     str,
        output_dir:      Path,
        checkpoints_dir: Path,
        audit_samples:   int = 10,
        max_new_tokens:  int = 400,
    ):
        self.val_dataset     = val_dataset
        self.processor       = processor
        self.config_name     = config_name
        self.output_dir      = output_dir
        self.checkpoints_dir = checkpoints_dir
        self.audit_samples   = audit_samples
        self.max_new_tokens  = max_new_tokens

        # Paths de salida
        self.history_path   = output_dir / f"history_final_{config_name}.csv"
        self.audit_path     = output_dir / f"audit_final_{config_name}.txt"
        self.best_model_dir = checkpoints_dir / f"best_model_{config_name}"

        self.best_val_loss        = float('inf')
        self.history: List[Dict]  = []

        # Inicializar audit log limpio para esta config
        self.audit_path.write_text(
            f"# Audit Log Gran Final -- {config_name}\n"
            f"# Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# Columnas: Step | Vista | Prediccion | Real | BLEU-4 | ROUGE-L\n\n"
        )

    def on_evaluate(
        self,
        args:    TrainingArguments,
        state:   TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict] = None,
        **kwargs,
    ):
        """Ejecutado despues de cada evaluate() del Trainer."""

        model = kwargs.get('model', None)
        if model is None:
            return

        step     = state.global_step
        val_loss = (metrics or {}).get('eval_loss', None)

        # Determinar device del modelo
        try:
            device = next(p for p in model.parameters() if p.requires_grad).device
        except StopIteration:
            device = next(model.parameters()).device

        model.eval()

        # --- OBJETIVO 5: Seleccionar 10 muestras aleatorias ---
        n       = min(self.audit_samples, len(self.val_dataset))
        indices = np.random.choice(len(self.val_dataset), n, replace=False)

        bleu4_scores  = []
        rougeL_scores = []

        audit_lines = []

        with torch.no_grad():
            for idx in indices:
                try:
                    sample = self.val_dataset[idx]
                    row    = self.val_dataset.data.iloc[idx]
                    vista  = self.val_dataset._mapear_vista(row.get('ViewPosition', None))

                    # Generar reporte con max_new_tokens=400
                    generated = generate_sample(model, sample, self.processor, device, self.max_new_tokens)

                    # Ground truth decodificado
                    reference = self.processor.batch_decode(
                        [sample['labels']], skip_special_tokens=True
                    )[0]

                    # Calcular metricas por muestra
                    b4  = _bleu4(reference, generated)
                    rl  = _rougeL(reference, generated)
                    bleu4_scores.append(b4)
                    rougeL_scores.append(rl)

                    # Audit line: sin truncar strings
                    audit_lines.append(
                        f"Step: {step} | Vista: {vista} | "
                        f"Prediccion: {generated} | "
                        f"Real: {reference} | "
                        f"BLEU-4: {b4:.4f} | ROUGE-L: {rl:.4f}\n"
                    )

                except Exception as e:
                    audit_lines.append(f"Step: {step} | Error en muestra {idx}: {e}\n")

        # Promedios
        mean_bleu4  = float(np.mean(bleu4_scores))  if bleu4_scores  else 0.0
        mean_rougeL = float(np.mean(rougeL_scores)) if rougeL_scores else 0.0

        # --- OBJETIVO 5: Guardar metricas en history CSV ---
        self.history.append({
            'Step':    step,
            'Val_Loss': val_loss,
            'BLEU-4':  mean_bleu4,
            'ROUGE-L': mean_rougeL,
        })
        pd.DataFrame(self.history).to_csv(self.history_path, index=False)

        print(
            f"\n   [Step {step}] "
            f"val_loss={val_loss:.4f}  BLEU-4={mean_bleu4:.4f}  ROUGE-L={mean_rougeL:.4f}"
        )

        # --- OBJETIVO 4: Guardar checkpoint SOLO si val_loss mejoro ---
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            # Guardar solo los adaptadores LoRA del language_model
            lm = getattr(model, 'language_model', model)
            lm.save_pretrained(str(self.best_model_dir))
            print(f"   Nuevo mejor modelo guardado (val_loss={val_loss:.4f}) -> {self.best_model_dir}")

        # --- OBJETIVO 5: Escribir audit log (sin truncar) ---
        with open(self.audit_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'─' * 100}\n")
            for line in audit_lines:
                f.write(line)

        # OBJETIVO 1: Liberar VRAM despues de generacion
        gc.collect()
        torch.cuda.empty_cache()
        model.train()


# ============================================================================
# CARGA DE DATOS (una sola vez, compartidos entre ambas configs)
# ============================================================================

print("Cargando datasets...\n")

processor = cargar_procesador(model_name=config.model.model_name)

train_dataset = MimicCXRDataset(
    csv_path=config.paths.train_csv,
    images_dir=config.paths.images_dir,
    processor=processor,
    image_size=config.data.image_size,
    clahe_clip_limit=config.data.clahe_clip_limit,
    clahe_tile_grid_size=config.data.clahe_tile_grid_size,
    max_length=512,         # OBJETIVO 2: limite explicito de tokens
    padding=config.data.padding,
    prompt=config.inference.default_prompt,  # inyección del prompt de entrenamiento
)

val_dataset = MimicCXRDataset(
    csv_path=config.paths.val_csv,
    images_dir=config.paths.images_dir,
    processor=processor,
    image_size=config.data.image_size,
    clahe_clip_limit=config.data.clahe_clip_limit,
    clahe_tile_grid_size=config.data.clahe_tile_grid_size,
    max_length=512,         # OBJETIVO 2: limite explicito de tokens
    padding=config.data.padding,
    prompt=config.inference.default_prompt,  # inyección del prompt de entrenamiento
)

data_collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True)

print(f"   Train: {len(train_dataset):,} muestras")
print(f"   Val:   {len(val_dataset):,} muestras\n")


# ============================================================================
# FUNCION DE ENTRENAMIENTO POR CONFIGURACION
# ============================================================================

def train_single_config(config_dict: dict, config_num: int, total_configs: int) -> Dict:
    """
    Entrena una configuracion LoRA de forma completamente aislada.

    OBJETIVO 1 — Aislamiento:
      Trainer nuevo por config -> global_step=0, epoch=0 al iniciar.
      Al terminar, destruye modelo + trainer + callbacks y libera VRAM.
    """
    print(f"\n{'='*80}")
    print(f"CONFIG {config_num}/{total_configs}: {config_dict['name']}  "
          f"r={config_dict['r']}, alpha={config_dict['alpha']}, lr={TRAINING_CONFIG['lr']:.1e}")
    print(f"{'='*80}\n")

    # OBJETIVO 1: Limpiar VRAM antes de iniciar esta config
    gc.collect()
    torch.cuda.empty_cache()

    # Cargar modelo base FRESCO (pesos independientes entre configs)
    print("   Cargando modelo base BLIP-2...")
    base_model = cargar_modelo_base(
        model_name=config.model.model_name,
        use_quantization=False,
        device_map="auto",
    )

    # Aplicar LoRA con los hiperparametros de esta config
    print(f"   Aplicando LoRA (r={config_dict['r']}, alpha={config_dict['alpha']})...")
    lora_cfg = LoraConfig(
        r=config_dict['r'],
        lora_alpha=config_dict['alpha'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    # Aplicar LoRA SOLO al language_model interno (OPT), no al wrapper BLIP-2.
    # En PEFT >= 0.10, get_peft_model sobre el modelo completo BLIP-2 intenta
    # pasar inputs_embeds al forward() de nivel superior, que no lo acepta.
    base_model.language_model = get_peft_model(base_model.language_model, lora_cfg)
    model = base_model
    model.language_model.print_trainable_parameters()

    # Directorio de salida especifico de esta config
    config_output_dir = OUTPUT_DIR / config_dict['name']
    config_output_dir.mkdir(exist_ok=True)

    # OBJETIVO 3 y 4: TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(config_output_dir),

        # OBJETIVO 1: limite fijo de 50 epocas
        num_train_epochs=TRAINING_CONFIG['max_epochs'],

        per_device_train_batch_size=TRAINING_CONFIG['batch_size'],
        per_device_eval_batch_size=TRAINING_CONFIG['batch_size'],

        # OBJETIVO 3: batch efectivo = batch_size x gradient_accumulation = 16
        gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation'],

        learning_rate=TRAINING_CONFIG['lr'],

        # OBJETIVO 4: evaluar cada 500 steps
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG['eval_steps'],
        save_strategy="steps",
        save_steps=TRAINING_CONFIG['eval_steps'],

        # Necesario para EarlyStoppingCallback
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=100,
        logging_dir=str(config_output_dir / "logs"),

        # bf16 en lugar de fp16: RTX 3090 (Ampere) tiene soporte nativo bfloat16.
        # bf16=True desactiva GradScaler -> elimina el crash de unscale fp16.
        fp16=False,
        bf16=True,

        # gradient_checkpointing: recomputa activaciones en el backward
        # en lugar de mantenerlas en VRAM. Ahorra ~60% de memoria de activaciones
        # a cambio de ~20% mas de tiempo de computo.
        gradient_checkpointing=True,

        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],

        # OBJETIVO 1: num_workers=4 + pin_memory=True (cv2.setNumThreads(0) previene deadlocks)
        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        save_total_limit=2,
    )

    # OBJETIVO 5: Callback de auditoria
    audit_cb = FinalAuditCallback(
        val_dataset=val_dataset,
        processor=processor,
        config_name=config_dict['name'],
        output_dir=OUTPUT_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        audit_samples=TRAINING_CONFIG['audit_samples'],
        max_new_tokens=TRAINING_CONFIG['max_new_tokens'],
    )

    # OBJETIVO 4: Early stopping — patience=10 evaluaciones sin mejora
    early_stop_cb = EarlyStoppingCallback(
        early_stopping_patience=TRAINING_CONFIG['early_stopping_patience']
    )

    # OBJETIVO 1: Trainer NUEVO -> global_step=0, epoch=0 (aislamiento completo)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[early_stop_cb, audit_cb],
    )

    print(
        f"\n   Entrenando (max {TRAINING_CONFIG['max_epochs']} epocas, "
        f"early stop = {TRAINING_CONFIG['early_stopping_patience']} evals sin mejora, "
        f"eval cada {TRAINING_CONFIG['eval_steps']} steps)...\n"
    )

    # MODIFICADO PARA RESCATE: detectar checkpoints existentes y reanudar si los hay
    checkpoints_existentes = sorted(config_output_dir.glob("checkpoint-*"))
    hay_checkpoints = len(checkpoints_existentes) > 0

    if hay_checkpoints:
        ultimo_checkpoint = checkpoints_existentes[-1]
        print(f"   [RESCATE] Se detectaron {len(checkpoints_existentes)} checkpoint(s) en {config_output_dir}")
        print(f"   [RESCATE] Reanudando desde: {ultimo_checkpoint.name}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print(f"   [NUEVO] No se detectaron checkpoints en {config_output_dir}. Iniciando entrenamiento desde cero.")
        trainer.train()

    # Extraer resultados del log history del Trainer
    eval_logs  = [l for l in trainer.state.log_history if 'eval_loss' in l]
    train_logs = [l for l in trainer.state.log_history if 'loss' in l and 'eval_loss' not in l]

    best_val_loss = trainer.state.best_metric
    best_step     = None
    if eval_logs and best_val_loss is not None:
        for log in eval_logs:
            if abs(log.get('eval_loss', float('inf')) - best_val_loss) < 1e-6:
                best_step = log.get('step', None)
                break

    result = {
        'config_name':      config_dict['name'],
        'r':                config_dict['r'],
        'alpha':            config_dict['alpha'],
        'lr':               TRAINING_CONFIG['lr'],
        'total_steps':      trainer.state.global_step,
        'best_val_loss':    best_val_loss,
        'best_step':        best_step,
        'final_train_loss': train_logs[-1]['loss'] if train_logs else None,
        'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    print(f"\n   Completado | total_steps={result['total_steps']} | "
          f"best_val_loss={result['best_val_loss']} @ step {result['best_step']}")

    # OBJETIVO 1: Destruir modelo y trainer para liberar VRAM antes de la siguiente config
    del model, base_model, trainer, audit_cb, early_stop_cb
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================================
# BUCLE PRINCIPAL
# ============================================================================

def main():
    """Entrena las 2 configuraciones en secuencia y genera graficos comparativos."""

    all_results: List[Dict] = []
    total_configs = len(FINAL_CONFIGS)

    print("INICIANDO ENTRENAMIENTO GRAN FINAL...\n")

    for i, cfg in enumerate(FINAL_CONFIGS, 1):
        try:
            result = train_single_config(cfg, i, total_configs)
            all_results.append(result)
            # Persistir resultados parciales por si el proceso se interrumpe
            pd.DataFrame(all_results).to_csv(
                OUTPUT_DIR / "intermediate_results_final.csv", index=False
            )
        except Exception as e:
            import traceback
            print(f"\nERROR en {cfg['name']}: {e}")
            traceback.print_exc()
            continue

    if not all_results:
        print("No hay resultados para analizar.")
        return

    # Guardar ranking final ordenado por mejor val_loss
    df_results = pd.DataFrame(all_results).sort_values('best_val_loss')
    df_results.to_csv(OUTPUT_DIR / "final_results_ranked.csv", index=False)
    print("\nRANKING FINAL:\n")
    print(df_results.to_string(index=False))

    # ========================================================================
    # OBJETIVO 6: GRAFICOS COMPARATIVOS
    # Lee history_final_Config_A.csv y history_final_Config_B.csv
    # Genera 3 PNG con ambas configs superpuestas para comparacion.
    # ========================================================================

    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    palette = {
        'Config_A': 'steelblue',
        'Config_B': 'coral',
    }

    histories: Dict[str, pd.DataFrame] = {}
    for cfg in FINAL_CONFIGS:
        p = OUTPUT_DIR / f"history_final_{cfg['name']}.csv"
        if p.exists():
            histories[cfg['name']] = pd.read_csv(p)
        else:
            print(f"Advertencia: no se encontro historial para {cfg['name']} en {p}")

    if not histories:
        print("No hay historiales disponibles para graficar.")
    else:
        # --- Grafico 1: Val Loss vs Steps ---
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, df in histories.items():
            ax.plot(
                df['Step'], df['Val_Loss'],
                label=name, color=palette.get(name, None),
                linewidth=2, marker='o', markersize=4,
            )
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Val Loss', fontsize=12)
        ax.set_title('Val Loss vs Steps -- Gran Final', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "01_val_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   01_val_loss.png guardado")

        # --- Grafico 2: BLEU-4 vs Steps ---
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, df in histories.items():
            ax.plot(
                df['Step'], df['BLEU-4'],
                label=name, color=palette.get(name, None),
                linewidth=2, marker='o', markersize=4,
            )
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('BLEU-4', fontsize=12)
        ax.set_title('BLEU-4 vs Steps -- Gran Final', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "02_bleu4.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   02_bleu4.png guardado")

        # --- Grafico 3: ROUGE-L vs Steps ---
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, df in histories.items():
            ax.plot(
                df['Step'], df['ROUGE-L'],
                label=name, color=palette.get(name, None),
                linewidth=2, marker='o', markersize=4,
            )
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('ROUGE-L', fontsize=12)
        ax.set_title('ROUGE-L vs Steps -- Gran Final', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "03_rougeL.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   03_rougeL.png guardado")

        print(f"\nGraficos guardados en: {plots_dir}")

    # Limpieza final
    gc.collect()
    torch.cuda.empty_cache()

    # Mostrar ganadora
    winner = df_results.iloc[0]
    print("\n" + "="*80)
    print("ENTRENAMIENTO GRAN FINAL COMPLETADO")
    print("="*80)
    print(f"\nMEJOR CONFIG: {winner['config_name']}  "
          f"r={winner['r']}, alpha={winner['alpha']}")
    print(f"   best_val_loss={winner['best_val_loss']:.4f} @ step {winner['best_step']}")
    print(f"\nOutputs en {OUTPUT_DIR}:")
    print(f"   final_results_ranked.csv")
    print(f"   history_final_Config_A.csv  /  history_final_Config_B.csv")
    print(f"   audit_final_Config_A.txt    /  audit_final_Config_B.txt")
    print(f"   checkpoints/best_model_Config_A/  /  checkpoints/best_model_Config_B/")
    print(f"   plots/01_val_loss.png  /  02_bleu4.png  /  03_rougeL.png\n")


# ============================================================================
# EJECUTAR
# ============================================================================

if __name__ == "__main__":
    main()
