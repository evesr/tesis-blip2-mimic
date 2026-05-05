"""
Entrenamiento Semifinal - 3 Configuraciones LoRA con Early Stopping por Steps
==============================================================================

Objetivos implementados:

OBJETIVO 1 - Infraestructura:
  • cv2.setNumThreads(0) antes de importar torch (previene deadlocks)
  • DataLoader con num_workers=4 y pin_memory=True
  • Aislamiento total por config: Trainer nuevo → global_step y epoch se
    resetean a 0 en cada config (evita bug de duplicación de épocas)
  • max_epochs = 50 (límite fijo por config)

OBJETIVO 2 - Prompting dinámico:
  • MimicCXRDataset._mapear_vista(): PA/AP → Frontal, LL/LATERAL → Lateral
  • Prompt en __getitem__: "[Context: {vista} view] {config.inference.default_prompt}"
    (editado en data_loader.py)

OBJETIVO 3 - Entrenamiento y checkpoints:
  • gradient_accumulation_steps=4 → batch efectivo = 16
  • eval_strategy="steps", eval_steps=50
  • EarlyStoppingCallback: patience=10 evaluaciones (no épocas)
  • SemifinalAuditCallback guarda:
      - ./checkpoints/latest_model_Config_X en cada evaluación
      - ./checkpoints/best_model_Config_X  cuando val_loss mejora

OBJETIVO 4 - Auditoría:
  • history_Config_X.csv: columnas Step, Val_Loss, BLEU, ROUGE-L
  • audit_Config_X.txt:   Step | Vista | Predicción | Real  (3 muestras/eval)

OBJETIVO 5 - Gráficos finales:
  • Lee los 3 history_*.csv generados y guarda:
      plots/01_val_loss.png, plots/02_bleu.png, plots/03_rougeL.png

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

# Métricas NLP
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from config import config
from data_loader import MimicCXRDataset
from model_utils import cargar_procesador, cargar_modelo_base

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# OBJETIVO 1: CONFIGURACIÓN GLOBAL
# Configs correctas: Config4 (r=16,a=32), Config5 (r=16,a=16), Config6 (r=32,a=64)
# ============================================================================

SEMIFINAL_CONFIGS = [
    {'name': 'Config_4', 'r': 16, 'alpha': 32},
    {'name': 'Config_5', 'r': 16, 'alpha': 16},
    {'name': 'Config_6', 'r': 32, 'alpha': 64},
]

TRAINING_CONFIG = {
    'lr': 1e-4,
    'max_epochs': 50,               # OBJETIVO 1: límite fijo por config
    'early_stopping_patience': 10,  # OBJETIVO 3: patience en nº de evaluaciones
    'batch_size': 4,
    'gradient_accumulation': 4,     # OBJETIVO 3: batch efectivo = 16
    'eval_steps': 50,               # OBJETIVO 3: evaluar cada 50 steps
    'eval_samples': 50,             # muestras para BLEU/ROUGE en cada eval
    'audit_samples': 3,             # reportes de muestra en audit log
}

OUTPUT_DIR = config.paths.base_dir / "semifinal_results"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINTS_DIR.mkdir(exist_ok=True)

print("\n" + "="*80)
print("🏆 ENTRENAMIENTO SEMIFINAL - 3 CONFIGURACIONES LoRA")
print("="*80)
print("   Config 4: r=16, α=32  |  Config 5: r=16, α=16  |  Config 6: r=32, α=64")
print(f"   max_epochs={TRAINING_CONFIG['max_epochs']}  |  "
      f"early_stop patience={TRAINING_CONFIG['early_stopping_patience']} evals  |  "
      f"eval_steps={TRAINING_CONFIG['eval_steps']}")
print(f"   batch={TRAINING_CONFIG['batch_size']} × grad_accum={TRAINING_CONFIG['gradient_accumulation']}"
      f" → batch_efectivo={TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['gradient_accumulation']}")
print(f"   Resultados en: {OUTPUT_DIR}")
print("="*80 + "\n")

# ============================================================================
# UTILIDADES DE MÉTRICAS NLP
# ============================================================================

_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
_smoothing = SmoothingFunction()


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
    return _rouge_scorer.score(reference, candidate)['rougeL'].fmeasure


def evaluate_nlp_metrics(
    model, val_dataset: MimicCXRDataset, processor, device, num_samples: int = 50
) -> Dict[str, float]:
    """
    Evalúa BLEU-4 y ROUGE-L sobre un subset aleatorio del val set.
    Usa model.generate() con greedy decoding (num_beams=1) para velocidad.
    Libera memoria con gc.collect() + torch.cuda.empty_cache() al terminar.
    """
    model.eval()
    bleu4_scores, rougeL_scores = [], []
    indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)

    with torch.no_grad():
        for idx in indices:
            try:
                sample = val_dataset[idx]
                pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
                outputs = model.generate(pixel_values=pixel_values, max_new_tokens=100, num_beams=1)
                generated = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                reference = processor.batch_decode([sample['labels']], skip_special_tokens=True)[0]
                bleu4_scores.append(_bleu4(reference, generated))
                rougeL_scores.append(_rougeL(reference, generated))
            except Exception:
                continue

    # OBJETIVO 1: Liberar memoria después de generar (post-generación)
    gc.collect()
    torch.cuda.empty_cache()
    model.train()

    return {
        'bleu4':  float(np.mean(bleu4_scores))  if bleu4_scores  else 0.0,
        'rougeL': float(np.mean(rougeL_scores)) if rougeL_scores else 0.0,
    }


# ============================================================================
# OBJETIVO 4: CALLBACK DE AUDITORÍA, MÉTRICAS Y CHECKPOINTS
# ============================================================================

class SemifinalAuditCallback(TrainerCallback):
    """
    Callback ejecutado tras cada evaluación del Trainer.

    OBJETIVO 3 — Checkpoints:
      • Guarda pesos en ./checkpoints/latest_model_Config_X  (siempre)
      • Guarda pesos en ./checkpoints/best_model_Config_X   (solo si val_loss mejora)

    OBJETIVO 4 — Persistencia de auditoría:
      • Appends a history_Config_X.csv: Step, Val_Loss, BLEU, ROUGE-L
      • Appends a audit_Config_X.txt:   3 reportes formato
        "Step: N | Vista: V | Predicción: P | Real: R"
    """

    def __init__(
        self,
        val_dataset: MimicCXRDataset,
        processor,
        config_name: str,
        output_dir: Path,
        checkpoints_dir: Path,
        eval_samples: int = 50,
        audit_samples: int = 3,
    ):
        self.val_dataset     = val_dataset
        self.processor       = processor
        self.config_name     = config_name
        self.output_dir      = output_dir
        self.checkpoints_dir = checkpoints_dir
        self.eval_samples    = eval_samples
        self.audit_samples   = audit_samples

        # Paths de salida
        self.history_path    = output_dir / f"history_{config_name}.csv"
        self.audit_path      = output_dir / f"audit_{config_name}.txt"
        self.best_model_dir  = checkpoints_dir / f"best_model_{config_name}"
        self.latest_model_dir = checkpoints_dir / f"latest_model_{config_name}"

        self.best_val_loss   = float('inf')
        self.history: List[Dict] = []

        # Inicializar archivos limpios para esta config
        self.audit_path.write_text(
            f"# Audit Log — {config_name}\n"
            f"# Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict] = None,
        **kwargs,
    ):
        """Se ejecuta después de cada llamada a evaluate() dentro del Trainer."""

        # El Trainer pasa el modelo actual en kwargs (puede haber cambiado tras
        # load_best_model_at_end o restore_callback_states)
        model  = kwargs.get('model', None)
        if model is None:
            return  # Sin modelo, no podemos hacer nada

        step     = state.global_step
        val_loss = (metrics or {}).get('eval_loss', None)

        # Determinar el device del modelo (toma el primer parámetro entrenable)
        try:
            device = next(p for p in model.parameters() if p.requires_grad).device
        except StopIteration:
            device = next(model.parameters()).device

        # --- OBJETIVO 4: Calcular BLEU y ROUGE-L con model.generate() ---
        nlp = evaluate_nlp_metrics(model, self.val_dataset, self.processor, device, self.eval_samples)

        # --- OBJETIVO 4: Guardar métricas en history_Config_X.csv ---
        self.history.append({
            'Step':    step,
            'Val_Loss': val_loss,
            'BLEU':    nlp['bleu4'],
            'ROUGE-L': nlp['rougeL'],
        })
        pd.DataFrame(self.history).to_csv(self.history_path, index=False)

        print(
            f"\n   📊 [Step {step}] "
            f"val_loss={val_loss:.4f}  BLEU={nlp['bleu4']:.4f}  ROUGE-L={nlp['rougeL']:.4f}"
        )

        # --- OBJETIVO 3: Guardar latest_model en cada evaluación ---
        model.save_pretrained(str(self.latest_model_dir))

        # --- OBJETIVO 3: Guardar best_model si val_loss mejoró ---
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            model.save_pretrained(str(self.best_model_dir))
            print(f"   🏆 Nuevo mejor modelo guardado (val_loss={val_loss:.4f})")

        # --- OBJETIVO 4: Audit log — 3 reportes de muestra ---
        audit_indices = np.random.choice(
            len(self.val_dataset), min(self.audit_samples, len(self.val_dataset)), replace=False
        )
        with open(self.audit_path, 'a') as f:
            f.write(f"\n{'─' * 80}\n")
            for idx in audit_indices:
                try:
                    sample = self.val_dataset[idx]
                    row    = self.val_dataset.data.iloc[idx]
                    # Obtener vista usando el mismo método del Dataset
                    view   = self.val_dataset._mapear_vista(row.get('ViewPosition', None))

                    pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = model.generate(pixel_values=pixel_values, max_new_tokens=100, num_beams=1)
                    generated = self.processor.batch_decode(out, skip_special_tokens=True)[0]
                    reference = self.processor.batch_decode([sample['labels']], skip_special_tokens=True)[0]

                    f.write(
                        f"Step: {step} | Vista: {view} | "
                        f"Predicción: {generated[:300]} | "
                        f"Real: {reference[:300]}\n"
                    )
                except Exception as e:
                    f.write(f"Step: {step} | Error en muestra {idx}: {e}\n")

        # OBJETIVO 1: Liberar memoria post-generación del audit log
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================================
# CARGAR DATOS — una sola vez, compartidos entre las 3 configs
# ============================================================================

print("📂 Cargando datasets...\n")

processor = cargar_procesador(model_name=config.model.model_name)

train_dataset = MimicCXRDataset(
    csv_path=config.paths.train_csv,
    images_dir=config.paths.images_dir,
    processor=processor,
    image_size=config.data.image_size,
    clahe_clip_limit=config.data.clahe_clip_limit,
    clahe_tile_grid_size=config.data.clahe_tile_grid_size,
    max_length=config.data.max_length,
    padding=config.data.padding,
)

val_dataset = MimicCXRDataset(
    csv_path=config.paths.val_csv,
    images_dir=config.paths.images_dir,
    processor=processor,
    image_size=config.data.image_size,
    clahe_clip_limit=config.data.clahe_clip_limit,
    clahe_tile_grid_size=config.data.clahe_tile_grid_size,
    max_length=config.data.max_length,
    padding=config.data.padding,
)

data_collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, padding=True)

print(f"   ✅ Train: {len(train_dataset):,} muestras")
print(f"   ✅ Val:   {len(val_dataset):,} muestras\n")


# ============================================================================
# FUNCIÓN DE ENTRENAMIENTO POR CONFIGURACIÓN
# ============================================================================

def train_single_config(config_dict: dict, config_num: int, total_configs: int) -> Dict:
    """
    Entrena una configuración LoRA completa de forma aislada.

    OBJETIVO 1 — Aislamiento de configuración:
      Se crea un Trainer NUEVO por config → global_step y epoch se reinician
      a 0 automáticamente (el Trainer no hereda estado de configs anteriores).
      Destruir model + trainer + callbacks al finalizar antes de la siguiente config.
    """
    print(f"\n{'='*80}")
    print(f"🔧 CONFIG {config_num}/{total_configs}: {config_dict['name']}  "
          f"r={config_dict['r']}, α={config_dict['alpha']}, lr={TRAINING_CONFIG['lr']:.1e}")
    print(f"{'='*80}\n")

    # OBJETIVO 1: Limpiar memoria antes de iniciar (aislamiento)
    gc.collect()
    torch.cuda.empty_cache()

    # Cargar modelo base FRESCO para esta config (pesos independientes)
    print("   📥 Cargando modelo base...")
    base_model = cargar_modelo_base(
        model_name=config.model.model_name,
        use_quantization=False,
        device_map="auto",
    )

    # Aplicar LoRA con los hiperparámetros de esta config
    print(f"   🔧 Aplicando LoRA (r={config_dict['r']}, α={config_dict['alpha']})...")
    lora_cfg = LoraConfig(
        r=config_dict['r'],
        lora_alpha=config_dict['alpha'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # Obtener device para generación en callbacks
    try:
        device = next(p for p in model.parameters() if p.requires_grad).device
    except StopIteration:
        device = next(model.parameters()).device

    # Directorio de salida de esta config
    config_output_dir = OUTPUT_DIR / config_dict['name']
    config_output_dir.mkdir(exist_ok=True)

    # OBJETIVO 3: eval_strategy y save_strategy basados en steps (no épocas)
    # Esto permite que EarlyStoppingCallback cuente evaluaciones, no épocas.
    training_args = TrainingArguments(
        output_dir=str(config_output_dir),

        # OBJETIVO 1: límite fijo de 50 épocas
        num_train_epochs=TRAINING_CONFIG['max_epochs'],

        per_device_train_batch_size=TRAINING_CONFIG['batch_size'],
        per_device_eval_batch_size=TRAINING_CONFIG['batch_size'],

        # OBJETIVO 3: batch efectivo = batch_size × gradient_accumulation = 16
        gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation'],

        learning_rate=TRAINING_CONFIG['lr'],

        # OBJETIVO 3: Evaluar y guardar cada 50 steps (sincronizados)
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG['eval_steps'],
        save_strategy="steps",
        save_steps=TRAINING_CONFIG['eval_steps'],

        # Necesario para que EarlyStoppingCallback pueda restaurar el mejor modelo
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=50,
        logging_dir=str(config_output_dir / "logs"),

        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],

        # OBJETIVO 1: DataLoader sin deadlocks + eficiente en GPU
        dataloader_num_workers=4,   # cv2.setNumThreads(0) previene deadlocks
        dataloader_pin_memory=True,  # acelera transferencia CPU → GPU

        save_total_limit=3,
    )

    # OBJETIVO 4: Callback de auditoría (métricas CSV + audit log + checkpoints)
    audit_cb = SemifinalAuditCallback(
        val_dataset=val_dataset,
        processor=processor,
        config_name=config_dict['name'],
        output_dir=OUTPUT_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        eval_samples=TRAINING_CONFIG['eval_samples'],
        audit_samples=TRAINING_CONFIG['audit_samples'],
    )

    # OBJETIVO 3: Early stopping basado en nº de evaluaciones (steps)
    # patience=10 → si en 10 evaluaciones consecutivas val_loss no mejora, para.
    early_stop_cb = EarlyStoppingCallback(
        early_stopping_patience=TRAINING_CONFIG['early_stopping_patience']
    )

    # OBJETIVO 1: Trainer NUEVO → global_step=0, epoch=0 (aislamiento)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[early_stop_cb, audit_cb],
    )

    print(
        f"\n   🚀 Entrenando (max {TRAINING_CONFIG['max_epochs']} épocas, "
        f"early stop = {TRAINING_CONFIG['early_stopping_patience']} evals sin mejora)...\n"
    )
    trainer.train()

    # Extraer resultados del log history
    eval_logs  = [l for l in trainer.state.log_history if 'eval_loss' in l]
    train_logs = [l for l in trainer.state.log_history if 'loss' in l and 'eval_loss' not in l]

    best_val_loss = trainer.state.best_metric
    best_step = None
    if eval_logs and best_val_loss is not None:
        for l in eval_logs:
            if abs(l.get('eval_loss', float('inf')) - best_val_loss) < 1e-6:
                best_step = l.get('step', None)
                break

    result = {
        'config_name':        config_dict['name'],
        'r':                  config_dict['r'],
        'alpha':              config_dict['alpha'],
        'lr':                 TRAINING_CONFIG['lr'],
        'total_steps':        trainer.state.global_step,
        'best_val_loss':      best_val_loss,
        'best_step':          best_step,
        'final_train_loss':   train_logs[-1]['loss'] if train_logs else None,
        'timestamp':          datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    print(f"\n   ✅ Completado | total_steps={result['total_steps']} | "
          f"best_val_loss={result['best_val_loss']} @ step {result['best_step']}")

    # OBJETIVO 1: Destruir modelo y trainer para aislar la siguiente config
    del model, base_model, trainer, audit_cb, early_stop_cb
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================================
# BUCLE PRINCIPAL
# ============================================================================

def main():
    """Entrena las 3 configuraciones en secuencia y genera gráficos comparativos."""

    all_results: List[Dict] = []
    total_configs = len(SEMIFINAL_CONFIGS)

    print("🏁 INICIANDO ENTRENAMIENTO SEMIFINAL...\n")

    for i, cfg in enumerate(SEMIFINAL_CONFIGS, 1):
        try:
            result = train_single_config(cfg, i, total_configs)
            all_results.append(result)
            # Persistir resultados intermedios por si el proceso se interrumpe
            pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "intermediate_results.csv", index=False)
        except Exception as e:
            import traceback
            print(f"\n❌ ERROR en {cfg['name']}: {e}")
            traceback.print_exc()
            continue

    if not all_results:
        print("❌ No hay resultados para analizar.")
        return

    # Guardar ranking final ordenado por mejor val_loss
    df_results = pd.DataFrame(all_results).sort_values('best_val_loss')
    df_results.to_csv(OUTPUT_DIR / "final_results_ranked.csv", index=False)
    print("\n🏆 RANKING FINAL:\n")
    print(df_results.to_string(index=False))

    # ========================================================================
    # OBJETIVO 5: GRÁFICOS COMPARATIVOS DESDE LOS 3 CSVs DE HISTORIAL
    # Cada CSV tiene columnas: Step, Val_Loss, BLEU, ROUGE-L
    # Se generan 3 PNG de evolución por steps (no barras de resumen).
    # ========================================================================

    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Leer los 3 archivos history_Config_X.csv
    palette = {'Config_4': 'steelblue', 'Config_5': 'coral', 'Config_6': 'mediumseagreen'}
    histories: Dict[str, pd.DataFrame] = {}
    for cfg in SEMIFINAL_CONFIGS:
        p = OUTPUT_DIR / f"history_{cfg['name']}.csv"
        if p.exists():
            histories[cfg['name']] = pd.read_csv(p)
        else:
            print(f"⚠️  No se encontró historial para {cfg['name']} en {p}")

    if not histories:
        print("⚠️  No hay historiales disponibles para graficar.")
    else:
        # Gráfico 1: Val Loss por step
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, df in histories.items():
            ax.plot(df['Step'], df['Val_Loss'], label=name,
                    color=palette.get(name, None), linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Step')
        ax.set_ylabel('Val Loss')
        ax.set_title('Val Loss por Step — Semifinal')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "01_val_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ 01_val_loss.png")

        # Gráfico 2: BLEU-4 por step
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, df in histories.items():
            ax.plot(df['Step'], df['BLEU'], label=name,
                    color=palette.get(name, None), linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Step')
        ax.set_ylabel('BLEU-4')
        ax.set_title('BLEU-4 por Step — Semifinal')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "02_bleu.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ 02_bleu.png")

        # Gráfico 3: ROUGE-L por step
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, df in histories.items():
            ax.plot(df['Step'], df['ROUGE-L'], label=name,
                    color=palette.get(name, None), linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Step')
        ax.set_ylabel('ROUGE-L')
        ax.set_title('ROUGE-L por Step — Semifinal')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "03_rougeL.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ 03_rougeL.png")

        print(f"\n✅ Gráficos guardados en: {plots_dir}")

    # Limpieza final
    gc.collect()
    torch.cuda.empty_cache()

    # Mostrar ganador
    winner = df_results.iloc[0]
    print("\n" + "="*80)
    print("🎉 ENTRENAMIENTO SEMIFINAL COMPLETADO")
    print("="*80)
    print(f"\n🏆 MEJOR CONFIG: {winner['config_name']}  "
          f"r={winner['r']}, α={winner['alpha']}")
    print(f"   best_val_loss={winner['best_val_loss']:.4f} @ step {winner['best_step']}")
    print(f"\n📁 Outputs:")
    print(f"   {OUTPUT_DIR}/final_results_ranked.csv")
    print(f"   {OUTPUT_DIR}/history_Config_X.csv  (una por config)")
    print(f"   {OUTPUT_DIR}/audit_Config_X.txt    (una por config)")
    print(f"   {CHECKPOINTS_DIR}/best_model_Config_X/")
    print(f"   {CHECKPOINTS_DIR}/latest_model_Config_X/")
    print(f"   {plots_dir}/01_val_loss.png  02_bleu.png  03_rougeL.png\n")


# ============================================================================
# EJECUTAR
# ============================================================================

if __name__ == "__main__":
    main()
