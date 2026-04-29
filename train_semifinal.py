"""
Entrenamiento Semifinal - Top 3 Configuraciones LoRA con Early Stopping
========================================================================

OBJETIVO 2: Script de entrenamiento con:
- 3 configuraciones específicas de LoRA (r=16/α=32, r=16/α=16, r=32/α=64)
- Early stopping (patience=3 épocas)
- 50 épocas máximo (pero early stopping detiene antes)
- Gráficos comparativos finales
- Limpieza agresiva de memoria

OBJETIVO 3: Optimizaciones:
- cv2.setNumThreads(0) para prevenir deadlocks
- dataloader_num_workers=4 y pin_memory=True

Autor: Sistema de ML
Fecha: Abril 2026
"""

import os
import gc
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# OBJETIVO 3 - MODIFICACIÓN: Prevención de deadlocks OpenCV/PyTorch
import cv2
cv2.setNumThreads(0)  # ← CRÍTICO: Desactivar multithreading de OpenCV

import torch
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback  # OBJETIVO 2: Early stopping
)
from peft import LoraConfig, get_peft_model, TaskType

# NLP Metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from config import config
from data_loader import MimicCXRDataset
from model_utils import cargar_procesador, cargar_modelo_base

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# CONFIGURACIÓN DEL ENTRENAMIENTO SEMIFINAL
# ============================================================================

print("\n" + "="*80)
print("🏆 ENTRENAMIENTO SEMIFINAL - TOP 3 CONFIGURACIONES LoRA")
print("="*80)
print("\n📋 CONFIGURACIÓN:")
print("   • Configuraciones a evaluar:")
print("     - Config 4: r=16, alpha=32")
print("     - Config 5: r=32, alpha=32")  # ← CORREGIDO
print("     - Config 6: r=32, alpha=64")
print("   • Learning rate: 1e-4 (fijo)")
print("   • Épocas máximo: 50")
print("   • Early Stopping: patience=3 (detiene si val_loss no mejora)")
print("\n🛡️ OPTIMIZACIONES:")
print("   ✓ cv2.setNumThreads(0) - Prevención deadlocks")
print("   ✓ dataloader_num_workers=4 - CPU eficiente")
print("   ✓ dataloader_pin_memory=True - GPU eficiente")
print("   ✓ torch.cuda.empty_cache() entre configs")
print("="*80 + "\n")

# OBJETIVO 2 - CONFIGURACIÓN: 3 configuraciones específicas
SEMIFINAL_CONFIGS = [
    # {'name': 'Config_4', 'r': 16, 'alpha': 32},  # ← YA ENTRENADA - SALTADA
    {'name': 'Config_5', 'r': 32, 'alpha': 32},  # ← CORREGIDO de r=16 a r=32
    {'name': 'Config_6', 'r': 32, 'alpha': 64},
]

TRAINING_CONFIG = {
    'lr': 1e-4,                    # Learning rate fijo
    'max_epochs': 50,              # Máximo (early stopping detiene antes)
    'early_stopping_patience': 3,  # OBJETIVO 2: Patience para early stopping
    'batch_size': 4,
    'gradient_accumulation': 4,
    'eval_samples': 50,            # Muestras para BLEU/ROUGE
}

OUTPUT_DIR = config.paths.base_dir / "semifinal_results"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"📁 Resultados se guardarán en: {OUTPUT_DIR}\n")

# ============================================================================
# INICIALIZACIÓN DE MÉTRICAS
# ============================================================================

rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smoothing = SmoothingFunction()

def calculate_bleu4(reference: str, candidate: str) -> float:
    """Calcula BLEU-4."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    try:
        score = sentence_bleu(
            [ref_tokens],
            cand_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing.method1
        )
        return score
    except:
        return 0.0

def calculate_rougeL(reference: str, candidate: str) -> float:
    """Calcula ROUGE-L F-measure."""
    scores = rouge_scorer_obj.score(reference, candidate)
    return scores['rougeL'].fmeasure

# ============================================================================
# CARGAR DATOS
# ============================================================================

print("📂 Cargando datasets...\n")

# Cargar procesador usando model_utils.py
processor = cargar_procesador(model_name=config.model.model_name)

# Crear datasets (con soft metadata tagging ya integrado en MimicCXRDataset)
train_dataset = MimicCXRDataset(
    csv_path=config.paths.train_csv,
    images_dir=config.paths.images_dir,
    processor=processor,
    image_size=config.data.image_size,
    clahe_clip_limit=config.data.clahe_clip_limit,
    clahe_tile_grid_size=config.data.clahe_tile_grid_size,
    max_length=config.data.max_length,
    padding=config.data.padding
)

val_dataset = MimicCXRDataset(
    csv_path=config.paths.val_csv,
    images_dir=config.paths.images_dir,
    processor=processor,
    image_size=config.data.image_size,
    clahe_clip_limit=config.data.clahe_clip_limit,
    clahe_tile_grid_size=config.data.clahe_tile_grid_size,
    max_length=config.data.max_length,
    padding=config.data.padding
)

print(f"   ✅ Train: {len(train_dataset)} muestras")
print(f"   ✅ Val: {len(val_dataset)} muestras\n")

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.tokenizer,
    padding=True
)

# ============================================================================
# FUNCIÓN DE EVALUACIÓN CON MÉTRICAS NLP
# ============================================================================

def evaluate_model_on_val(model, val_dataset, processor, device, num_samples=50):
    """
    Evalúa modelo en subset del val set calculando BLEU-4 y ROUGE-L.
    """
    model.eval()
    
    bleu4_scores = []
    rougeL_scores = []
    
    # Tomar subset aleatorio del val set
    indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
    
    with torch.no_grad():
        for idx in indices:
            try:
                # Obtener muestra
                sample = val_dataset[idx]
                
                # Preparar imagen para generación
                pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
                
                # Tokenizar prompt (ya incluye contexto de vista en el dataset)
                # El dataset ya formateó el prompt con "[Context: X view] Question: ..."
                # Aquí solo generamos sin prompt adicional
                outputs = model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=100,
                    num_beams=1,  # Greedy para velocidad
                    early_stopping=True
                )
                
                # Decodificar
                generated = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                reference = processor.batch_decode([sample['labels']], skip_special_tokens=True)[0]
                
                # Calcular métricas
                bleu4 = calculate_bleu4(reference, generated)
                rougeL = calculate_rougeL(reference, generated)
                
                bleu4_scores.append(bleu4)
                rougeL_scores.append(rougeL)
                
            except Exception as e:
                continue
    
    model.train()
    
    return {
        'bleu4': np.mean(bleu4_scores) if bleu4_scores else 0.0,
        'rougeL': np.mean(rougeL_scores) if rougeL_scores else 0.0
    }

# ============================================================================
# FUNCIÓN DE ENTRENAMIENTO POR CONFIGURACIÓN CON EARLY STOPPING
# ============================================================================

def train_single_config(config_dict: dict, config_num: int, total_configs: int):
    """
    Entrena modelo con una configuración específica usando Early Stopping.
    
    OBJETIVO 2 - IMPLEMENTACIÓN:
    - NO anidar bucles de épocas manualmente
    - Dejar que Trainer maneje las épocas (num_train_epochs=50)
    - Early stopping detiene automáticamente si val_loss no mejora en 3 épocas
    - Registrar mejor modelo y época
    """
    print(f"\n{'='*80}")
    print(f"🔧 CONFIGURACIÓN {config_num}/{total_configs}: {config_dict['name']}")
    print(f"   r={config_dict['r']}, alpha={config_dict['alpha']}, lr={TRAINING_CONFIG['lr']:.2e}")
    print(f"{'='*80}\n")
    
    # OBJETIVO 2: Limpiar memoria antes de empezar
    gc.collect()
    torch.cuda.empty_cache()
    
    # Cargar modelo base FRESCO
    print("   📥 Cargando modelo base...")
    base_model = cargar_modelo_base(
        model_name=config.model.model_name,
        use_quantization=False,
        device_map="auto"
    )
    
    # Configurar LoRA
    print(f"   🔧 Aplicando LoRA (r={config_dict['r']}, alpha={config_dict['alpha']})...")
    lora_config = LoraConfig(
        r=config_dict['r'],
        lora_alpha=config_dict['alpha'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    device = next(model.parameters()).device
    
    # Configurar directorio temporal
    temp_output = OUTPUT_DIR / config_dict['name']
    temp_output.mkdir(exist_ok=True)
    
    # OBJETIVO 2 - CONFIGURACIÓN: TrainingArguments con Early Stopping
    training_args = TrainingArguments(
        output_dir=str(temp_output),
        
        # OBJETIVO 2: 50 épocas máximo (early stopping detiene antes)
        num_train_epochs=TRAINING_CONFIG['max_epochs'],
        
        per_device_train_batch_size=TRAINING_CONFIG['batch_size'],
        per_device_eval_batch_size=TRAINING_CONFIG['batch_size'],
        gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation'],
        learning_rate=TRAINING_CONFIG['lr'],
        
        # OBJETIVO 2: Configuración para Early Stopping
        eval_strategy="epoch",            # Evaluar cada época
        save_strategy="epoch",            # Guardar cada época
        load_best_model_at_end=True,      # Cargar mejor modelo al final
        metric_for_best_model="eval_loss", # Métrica a monitorear
        greater_is_better=False,          # Menor loss es mejor
        
        logging_steps=50,
        logging_dir=str(temp_output / "logs"),
        
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        
        # OBJETIVO 3 - OPTIMIZACIÓN: CPU/GPU eficiente
        dataloader_num_workers=4,         # ← 4 workers (cv2.setNumThreads(0) evita deadlock)
        dataloader_pin_memory=True,       # ← Acelera transferencia CPU→GPU
        
        save_total_limit=3,               # Solo mantener 3 mejores checkpoints
    )
    
    # OBJETIVO 2 - EARLY STOPPING: Configurar callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=TRAINING_CONFIG['early_stopping_patience']  # 3 épocas
    )
    
    # Crear Trainer con Early Stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping_callback]  # ← OBJETIVO 2: Early stopping
    )
    
    # OBJETIVO 2 - ENTRENAMIENTO: Una sola llamada a trainer.train()
    # El Trainer manejará las épocas internamente y detendrá automáticamente
    # si val_loss no mejora durante 3 épocas consecutivas
    print(f"\n   🚀 Entrenando hasta {TRAINING_CONFIG['max_epochs']} épocas ")
    print(f"      (Early stopping: patience={TRAINING_CONFIG['early_stopping_patience']})...\n")
    
    # Entrenar (el Trainer maneja épocas y early stopping)
    trainer.train()
    
    # OBJETIVO 2 - RESULTADOS: Registrar mejor modelo y época
    best_epoch = None
    best_val_loss = None
    
    # Extraer información del mejor modelo
    if hasattr(trainer.state, 'best_metric'):
        best_val_loss = trainer.state.best_metric
    
    # Encontrar en qué época se alcanzó el mejor resultado
    if hasattr(trainer.state, 'log_history'):
        eval_logs = [log for log in trainer.state.log_history if 'eval_loss' in log]
        if eval_logs and best_val_loss is not None:
            for log in eval_logs:
                if abs(log.get('eval_loss', float('inf')) - best_val_loss) < 1e-6:
                    best_epoch = log.get('epoch', None)
                    break
    
    # Calcular métricas NLP en el mejor modelo
    print(f"\n   📊 Calculando métricas NLP en mejor modelo...")
    nlp_metrics = evaluate_model_on_val(
        model, 
        val_dataset, 
        processor,
        device, 
        num_samples=TRAINING_CONFIG['eval_samples']
    )
    
    # Obtener train_loss final
    train_logs = [log for log in trainer.state.log_history if 'loss' in log and 'eval_loss' not in log]
    final_train_loss = train_logs[-1]['loss'] if train_logs else None
    
    # Preparar resultados
    result = {
        'config_name': config_dict['name'],
        'r': config_dict['r'],
        'alpha': config_dict['alpha'],
        'lr': TRAINING_CONFIG['lr'],
        'total_epochs_trained': trainer.state.epoch if hasattr(trainer.state, 'epoch') else None,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'final_train_loss': final_train_loss,
        'bleu4': nlp_metrics['bleu4'],
        'rougeL': nlp_metrics['rougeL'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print(f"\n   ✅ Entrenamiento completado:")
    print(f"      • Total épocas: {result['total_epochs_trained']}")
    print(f"      • Mejor época: {result['best_epoch']}")
    print(f"      • Mejor val_loss: {result['best_val_loss']:.4f}")
    print(f"      • Train loss final: {result['final_train_loss']:.4f}")
    print(f"      • BLEU-4: {result['bleu4']:.4f}")
    print(f"      • ROUGE-L: {result['rougeL']:.4f}")
    
    # OBJETIVO 2: Limpiar memoria después de entrenar
    del model
    del base_model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return result

# ============================================================================
# BUCLE PRINCIPAL - ENTRENAR 3 CONFIGURACIONES
# ============================================================================

def main():
    """Función principal del entrenamiento semifinal."""
    
    all_results = []
    
    print("\n🏁 INICIANDO ENTRENAMIENTO SEMIFINAL...\n")
    
    total_configs = len(SEMIFINAL_CONFIGS)
    
    for i, config_dict in enumerate(SEMIFINAL_CONFIGS, 1):
        try:
            result = train_single_config(config_dict, i, total_configs)
            all_results.append(result)
            
            # Guardar resultados intermedios
            df_intermediate = pd.DataFrame(all_results)
            df_intermediate.to_csv(OUTPUT_DIR / "intermediate_results.csv", index=False)
            
        except Exception as e:
            print(f"\n❌ ERROR en {config_dict['name']}: {str(e)}\n")
            continue
    
    # ========================================================================
    # ANÁLISIS FINAL Y GRÁFICOS
    # ========================================================================
    
    print("\n" + "="*80)
    print("📊 GENERANDO ANÁLISIS FINAL Y GRÁFICOS")
    print("="*80 + "\n")
    
    if not all_results:
        print("❌ No hay resultados para analizar.")
        return
    
    # Crear DataFrame final
    df_results = pd.DataFrame(all_results)
    
    # Ordenar por mejor val_loss
    df_results = df_results.sort_values('best_val_loss')
    
    # Guardar resultados finales
    df_results.to_csv(OUTPUT_DIR / "final_results_ranked.csv", index=False)
    
    print("🏆 RANKING FINAL (ordenado por best_val_loss):\n")
    print(df_results.to_string(index=False))
    print("\n")
    
    # ========================================================================
    # GRÁFICOS COMPARATIVOS (como en grid_search)
    # ========================================================================
    
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Gráfico 1: Comparación de Val Loss
    plt.figure(figsize=(10, 6))
    plt.bar(df_results['config_name'], df_results['best_val_loss'], color='steelblue')
    plt.xlabel('Configuración')
    plt.ylabel('Best Val Loss')
    plt.title('Comparación de Mejor Val Loss - Semifinal')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_val_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 2: Comparación de BLEU-4
    plt.figure(figsize=(10, 6))
    plt.bar(df_results['config_name'], df_results['bleu4'], color='coral')
    plt.xlabel('Configuración')
    plt.ylabel('BLEU-4')
    plt.title('Comparación de BLEU-4 - Semifinal')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_bleu4.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 3: Comparación de ROUGE-L
    plt.figure(figsize=(10, 6))
    plt.bar(df_results['config_name'], df_results['rougeL'], color='mediumseagreen')
    plt.xlabel('Configuración')
    plt.ylabel('ROUGE-L')
    plt.title('Comparación de ROUGE-L - Semifinal')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_rougeL.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 4: Épocas entrenadas
    plt.figure(figsize=(10, 6))
    plt.bar(df_results['config_name'], df_results['total_epochs_trained'], color='plum')
    plt.axhline(y=TRAINING_CONFIG['max_epochs'], color='red', linestyle='--', 
                label=f'Máximo ({TRAINING_CONFIG["max_epochs"]} épocas)')
    plt.xlabel('Configuración')
    plt.ylabel('Épocas Entrenadas')
    plt.title('Épocas Entrenadas (Early Stopping)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_epochs.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gráfico 5: Dashboard completo (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Val Loss
    axes[0, 0].bar(df_results['config_name'], df_results['best_val_loss'], color='steelblue')
    axes[0, 0].set_title('Best Val Loss')
    axes[0, 0].set_ylabel('Val Loss')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # BLEU-4
    axes[0, 1].bar(df_results['config_name'], df_results['bleu4'], color='coral')
    axes[0, 1].set_title('BLEU-4')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # ROUGE-L
    axes[1, 0].bar(df_results['config_name'], df_results['rougeL'], color='mediumseagreen')
    axes[1, 0].set_title('ROUGE-L')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Épocas
    axes[1, 1].bar(df_results['config_name'], df_results['total_epochs_trained'], color='plum')
    axes[1, 1].axhline(y=TRAINING_CONFIG['max_epochs'], color='red', linestyle='--')
    axes[1, 1].set_title('Épocas Entrenadas')
    axes[1, 1].set_ylabel('Épocas')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Dashboard Semifinal - Comparación de Configuraciones', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(plots_dir / "dashboard_complete.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Gráficos guardados en: {plots_dir}\n")
    
    # Resumen final
    print("="*80)
    print("🎉 ENTRENAMIENTO SEMIFINAL COMPLETADO")
    print("="*80)
    print(f"\n📁 Resultados en: {OUTPUT_DIR}")
    print(f"   • final_results_ranked.csv - Ranking de configuraciones")
    print(f"   • intermediate_results.csv - Resultados intermedios")
    print(f"   • plots/ - Gráficos comparativos\n")
    
    # Mostrar ganador
    winner = df_results.iloc[0]
    print("🏆 MEJOR CONFIGURACIÓN:")
    print(f"   • {winner['config_name']}: r={winner['r']}, alpha={winner['alpha']}")
    print(f"   • Best Val Loss: {winner['best_val_loss']:.4f} (época {winner['best_epoch']})")
    print(f"   • BLEU-4: {winner['bleu4']:.4f}")
    print(f"   • ROUGE-L: {winner['rougeL']:.4f}")
    print(f"   • Épocas entrenadas: {winner['total_epochs_trained']}\n")
    
    # OBJETIVO 2: Limpieza final de memoria
    gc.collect()
    torch.cuda.empty_cache()
    
    print("="*80 + "\n")
    
    # ========================================================================
    # AUTO-PAUSAR POD DE RUNPOD (DESACTIVADO)
    # ========================================================================
    
    # print("🛑 Intentando pausar pod de RunPod...")
    # 
    # try:
    #     # Método 1: Usando runpodctl (si está instalado)
    #     import subprocess
    #     result = subprocess.run(['runpodctl', 'stop', 'pod'], 
    #                           capture_output=True, text=True, timeout=10)
    #     if result.returncode == 0:
    #         print("✅ Pod pausado exitosamente con runpodctl")
    #     else:
    #         # Método 2: Usando variable de entorno RUNPOD_POD_ID y API
    #         import requests
    #         pod_id = os.environ.get('RUNPOD_POD_ID')
    #         api_key = os.environ.get('RUNPOD_API_KEY')
    #         
    #         if pod_id and api_key:
    #             url = f"https://api.runpod.io/graphql"
    #             headers = {"Content-Type": "application/json"}
    #             query = f"""
    #             mutation {{
    #               podStop(input: {{podId: "{pod_id}"}}) {{
    #                 id
    #                 desiredStatus
    #               }}
    #             }}
    #             """
    #             response = requests.post(url, 
    #                                    json={"query": query},
    #                                    headers={"Authorization": api_key})
    #             if response.status_code == 200:
    #                 print("✅ Pod pausado exitosamente con API de RunPod")
    #             else:
    #                 print(f"⚠️  No se pudo pausar automáticamente. Pausa manualmente desde RunPod UI.")
    #         else:
    #             print("⚠️  Variables RUNPOD_POD_ID o RUNPOD_API_KEY no encontradas.")
    #             print("💡 Pausa manualmente el pod desde: https://www.runpod.io/console/pods")
    # except Exception as e:
    #     print(f"⚠️  No se pudo pausar automáticamente: {e}")
    #     print("💡 Pausa manualmente el pod desde: https://www.runpod.io/console/pods")
    
    print("\n✅ Entrenamiento semifinal completado. Pod permanecerá activo.\n")

# ============================================================================
# EJECUTAR
# ============================================================================

if __name__ == "__main__":
    main()
