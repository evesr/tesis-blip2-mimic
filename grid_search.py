"""
Grid Search de Hiperparámetros - LoRA para BLIP2
=================================================
Búsqueda exhaustiva de los mejores hiperparámetros para LoRA.

FLUJO COMPLETO:
--------------
1. PREPROCESAMIENTO DE IMÁGENES (automático por MimicCXRDataset):
   - Cargar imagen en escala de grises
   - Aplicar CLAHE (mejora de contraste)
   - Convertir a RGB
   - Aplicar padding a 224x224
   - Normalización por BLIP2Processor

2. ENTRENAMIENTO (por época):
   - Forward pass con imágenes preprocesadas
   - Cálculo de loss
   - Backward pass
   - Actualización de parámetros LoRA
   - Liberación de memoria (torch.cuda.empty_cache())

3. EVALUACIÓN (después de cada época):
   - Seleccionar 50 muestras aleatorias del val set
   - Para cada muestra:
     a) Preprocesar imagen (mismo pipeline que training)
     b) Tokenizar Prompt B con BLIP2Processor
     c) GENERAR reporte usando model.generate()
        * Prompt B: "Describe the clinical findings, anatomical structures..."
        * max_new_tokens=100 (reportes cortos para velocidad)
        * num_beams=3 (balance velocidad/calidad)
     d) Comparar reporte generado vs referencia
     e) Calcular BLEU-4 y ROUGE-L
   - Promediar métricas de las 50 muestras
   - Registrar: train_loss, val_loss, BLEU-4, ROUGE-L

4. LIMPIEZA DE MEMORIA:
   - torch.cuda.empty_cache() después de cada época
   - Eliminar modelo completamente después de cada combinación
   - gc.collect() agresivo

PROMPT USADO:
------------
Prompt B (seleccionado en ab_test.py):
"Question: Describe the clinical findings, anatomical structures, and 
abnormalities visible in this chest radiograph. Answer:"

- Se tokeniza el prompt con BLIP2Processor
- Se pasa junto con la imagen a model.generate()
- El modelo genera la respuesta después de "Answer:"
- Este prompt demostró mejor balance clínico/naturalidad en pruebas

Espacio de búsqueda:
- r: [8, 16, 32, 64]
- alpha: [r, 2*r] (dinámico)
- lr: [1e-4, 5e-5, 1e-5]
Total: 24 combinaciones × 4 épocas = 96 runs

Métricas evaluadas:
- Train Loss: Loss en conjunto de entrenamiento (calculado por Trainer)
- Val Loss: Loss en conjunto de validación (calculado por Trainer)
- BLEU-4: Precisión de 4-gramas (0=malo, 1=perfecto)
- ROUGE-L: Subsecuencia común más larga (0=malo, 1=perfecto)

Autor: Evelyn Silva Rozas
Fecha: Abril 2026
"""

import os
import gc
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import itertools

import cv2
cv2.setNumThreads(0)  # ← CRÍTICO: Desactivar multithreading de OpenCV (conflicto con PyTorch workers)

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# NLP Metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from config import config
from data_loader import MimicCXRDataset
from model_utils import cargar_procesador, cargar_modelo_base, crear_configuracion_lora
from model_utils import cargar_procesador, cargar_modelo_base, crear_configuracion_lora

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================================
# PROMPT SELECCIONADO (del A/B test - Prompt B ganó)
# ============================================================================
PROMPT_B = "Question: Describe the clinical findings, anatomical structures, and abnormalities visible in this chest radiograph. Answer:"
# Este prompt demostró mejor balance entre precisión clínica y naturalidad
# en el ab_test.py con 10 imágenes del test set

print("\n" + "="*80)
print("🔍 GRID SEARCH DE HIPERPARÁMETROS - LoRA + BLIP2")
print("="*80)
print("\n� CONFIGURACIÓN:")
print("   • Espacio de búsqueda (basado en literatura):")
print("     - r: [8, 16, 32] (rango óptimo)")
print("     - alpha: [r, 2*r] (conservador vs agresivo)")
print("     - lr: 1e-4 (sweet spot LoRA)")
print("   • Total: 6 combinaciones")
print("   • Épocas por combinación: 4")
print("\n🔄 PREPROCESAMIENTO DE IMÁGENES:")
print("   ✓ CLAHE (mejora de contraste)")
print("   ✓ Padding a 224x224")
print("   ✓ Normalización BLIP2")
print("\n📊 EVALUACIÓN POR ÉPOCA:")
print("   ✓ Train Loss (automático)")
print("   ✓ Val Loss (automático)")
print("   ✓ BLEU-4 (generando 50 reportes)")
print("   ✓ ROUGE-L (generando 50 reportes)")
print("\n💡 GENERACIÓN DE REPORTES:")
print("   • Prompt B (del ab_test): 'Describe clinical findings...'")
print("   • max_new_tokens=100")
print("   • num_beams=1 (greedy search para velocidad)")
print("\n⏱️  Estimado: ~2 horas (~$0.94)")
print("="*80 + "\n")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

GRID_CONFIG = {
    'r_values': [8, 16, 32],          # Rango óptimo según literatura (quitar 64 por OOM)
    'alpha_factors': [1, 2],          # alpha = r (conservador) y alpha = 2*r (agresivo)
    'lr_values': [1e-4],              # Sweet spot de LoRA según literatura
    'num_epochs': 4,                  # 4 épocas por combinación
    'batch_size': 4,                  # Aumentado a 4 (hay margen de memoria)
    'gradient_accumulation': 1,       # Sin acumulación = máxima velocidad (effective batch = 4)
    'eval_samples': 50,               # Muestras del val set para métricas NLP
}

# Total: 3 r × 2 alpha × 1 lr = 6 combinaciones (~2 horas, $0.94)

OUTPUT_DIR = config.paths.base_dir / "grid_search_results"
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

# Cargar procesador usando model_utils.py (reutilizado para todas las combinaciones)
processor = cargar_procesador(model_name=config.model.model_name)

# Crear datasets
# IMPORTANTE: MimicCXRDataset aplica preprocesamiento automáticamente:
# 1. estandarizar_imagen() de preprocessing.py:
#    - Carga imagen en escala de grises
#    - Aplica CLAHE (mejora de contraste)
#    - Convierte a RGB
#    - Aplica padding a target_size (224x224)
# 2. BLIP2Processor normaliza valores de píxeles
# Resultado: imágenes listas para BLIP2
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
# FUNCIÓN DE EVALUACIÓN
# ============================================================================

def evaluate_model_on_val(model, val_dataset, device, num_samples=50):
    """
    Evalúa modelo en subset del val set calculando BLEU-4 y ROUGE-L.
    
    FLUJO DE EVALUACIÓN:
    -------------------
    1. Seleccionar N muestras aleatorias del val set
    2. Para cada muestra:
       a) La imagen YA ESTÁ preprocesada (CLAHE + padding) por MimicCXRDataset
       b) Tokenizar Prompt B con BLIP2Processor
       c) GENERAR reporte usando model.generate():
          - pixel_values: imagen preprocesada
          - input_ids: tokens del Prompt B
          - attention_mask: máscara del prompt
          - max_new_tokens=100 (reportes cortos para velocidad)
          - num_beams=3 (balance velocidad/calidad)
          - early_stopping=True
       d) Decodificar tokens → texto
       e) Comparar texto generado vs texto de referencia (ground truth)
       f) Calcular BLEU-4 (precisión de 4-gramas)
       g) Calcular ROUGE-L (subsecuencia común más larga)
    3. Promediar todas las métricas
    4. Retornar métricas agregadas
    
    Args:
        model: Modelo BLIP2 con LoRA a evaluar
        val_dataset: MimicCXRDataset de validación (imágenes YA preprocesadas)
        device: Dispositivo (cuda/cpu)
        num_samples: Número de muestras a evaluar (↓ = más rápido, ↑ = más preciso)
    
    Returns:
        Dict con:
            - bleu4: BLEU-4 promedio (0-1, mayor es mejor)
            - rougeL: ROUGE-L promedio (0-1, mayor es mejor)
    
    Notes:
        - Se usa subset pequeño (50 muestras) para velocidad
        - Las métricas son aproximadas pero suficientes para ranking
        - Si falla una muestra, se salta (try/except)
    """
    model.eval()
    
    bleu4_scores = []
    rougeL_scores = []
    
    # Tomar subset aleatorio del val set
    indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)
    
    with torch.no_grad():
        for idx in indices:
            try:
                # Obtener muestra (imagen YA preprocesada por MimicCXRDataset)
                sample = val_dataset[idx]
                
                # Preparar imagen para generación
                pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
                
                # TOKENIZAR PROMPT B (seleccionado en ab_test)
                # "Question: Describe the clinical findings, anatomical structures..."
                prompt_tokens = processor(
                    text=PROMPT_B,
                    return_tensors="pt"
                ).to(device)
                
                # GENERAR reporte con PROMPT B + IMAGEN
                # El modelo recibe:
                # - pixel_values: imagen preprocesada (CLAHE + padding)
                # - input_ids: tokens del prompt
                # - attention_mask: máscara de atención del prompt
                # Y genera la respuesta después de "Answer:"
                outputs = model.generate(
                    pixel_values=pixel_values,
                    input_ids=prompt_tokens['input_ids'],
                    attention_mask=prompt_tokens['attention_mask'],
                    max_new_tokens=100,      # Límite de tokens (reportes cortos)
                    num_beams=1,             # ← Greedy search (3× más rápido para grid search)
                    early_stopping=True      # Detener al generar token de fin
                )
                
                # Decodificar tokens → texto
                generated = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                
                # Obtener texto de referencia (ground truth)
                # Los labels ya están en el sample (del CSV)
                reference = processor.batch_decode([sample['labels']], skip_special_tokens=True)[0]
                
                # Calcular métricas NLP
                bleu4 = calculate_bleu4(reference, generated)
                rougeL = calculate_rougeL(reference, generated)
                
                bleu4_scores.append(bleu4)
                rougeL_scores.append(rougeL)
                
            except Exception as e:
                # Si falla una muestra (error de generación, etc), continuar
                continue
    
    # Volver a modo entrenamiento
    model.train()
    
    # Retornar promedios
    return {
        'bleu4': np.mean(bleu4_scores) if bleu4_scores else 0.0,
        'rougeL': np.mean(rougeL_scores) if rougeL_scores else 0.0
    }

# ============================================================================
# FUNCIÓN DE ENTRENAMIENTO POR COMBINACIÓN
# ============================================================================

def train_single_combination(r: int, alpha: int, lr: float, combo_num: int, total_combos: int):
    """
    Entrena modelo con una combinación específica de hiperparámetros.
    
    Args:
        r: Rango de LoRA
        alpha: Alpha de LoRA
        lr: Learning rate
        combo_num: Número de combinación actual
        total_combos: Total de combinaciones
    
    Returns:
        Dict con resultados de todas las épocas
    """
    print(f"\n{'='*80}")
    print(f"🔧 COMBINACIÓN {combo_num}/{total_combos}")
    print(f"   r={r}, alpha={alpha}, lr={lr:.2e}")
    print(f"{'='*80}\n")
    
    # Limpiar memoria antes de empezar
    gc.collect()
    torch.cuda.empty_cache()
    
    # ========================================================================
    # Cargar modelo base FRESCO
    # ========================================================================
    print("   📥 Cargando modelo base...")
    # Usar función de model_utils.py
    base_model = cargar_modelo_base(
        model_name=config.model.model_name,
        use_quantization=False,  # Sin cuantización (evita error set_submodule)
        device_map="auto"
    )
    
    # ========================================================================
    # Configurar LoRA usando model_utils.py
    # ========================================================================
    print(f"   🔧 Aplicando LoRA (r={r}, alpha={alpha})...")
    
    # Usar función de model_utils.py para crear config LoRA
    lora_config = crear_configuracion_lora(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    device = next(model.parameters()).device
    
    # ========================================================================
    # Configurar entrenamiento
    # ========================================================================
    temp_output = OUTPUT_DIR / f"temp_r{r}_a{alpha}_lr{lr:.0e}"
    temp_output.mkdir(exist_ok=True)
    
    # DEBUG: Forzar verificación de batch_size
    print(f"\n{'='*80}")
    print(f"DEBUG - VERIFICACIÓN DE BATCH_SIZE:")
    print(f"   GRID_CONFIG['batch_size'] = {GRID_CONFIG['batch_size']}")
    print(f"   gradient_accumulation = {GRID_CONFIG['gradient_accumulation']}")
    print(f"   Effective batch = {GRID_CONFIG['batch_size'] * GRID_CONFIG['gradient_accumulation']}")
    print(f"   Steps esperados por época = {len(train_dataset) / GRID_CONFIG['batch_size']:.0f}")
    print(f"{'='*80}\n")
    
    training_args = TrainingArguments(
        output_dir=str(temp_output),
        num_train_epochs=1,  # ← 1 época por llamada (bucle manual controla total)
        per_device_train_batch_size=GRID_CONFIG['batch_size'],
        per_device_eval_batch_size=GRID_CONFIG['batch_size'],
        gradient_accumulation_steps=GRID_CONFIG['gradient_accumulation'],
        learning_rate=lr,
        
        # Evaluación cada época
        eval_strategy="epoch",
        save_strategy="no",  # NO guardar checkpoints
        
        logging_steps=50,
        logging_dir=str(temp_output / "logs"),
        
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        
        dataloader_num_workers=4,       # ← 4 workers (cv2.setNumThreads(0) evita deadlock)
        dataloader_pin_memory=True,     # ← Acelera transferencia CPU→GPU
        load_best_model_at_end=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # ========================================================================
    # Entrenar y registrar métricas por época
    # ========================================================================
    results_per_epoch = []
    
    print(f"\n   🚀 Entrenando {GRID_CONFIG['num_epochs']} épocas...\n")
    
    for epoch in range(GRID_CONFIG['num_epochs']):
        print(f"   📊 Época {epoch + 1}/{GRID_CONFIG['num_epochs']}")
        
        # ENTRENAR 1 ÉPOCA
        # - Forward pass con imágenes preprocesadas (CLAHE + padding)
        # - Cálculo de loss
        # - Backward pass
        # - Actualización de parámetros LoRA
        trainer.train()
        
        # Obtener train loss de los logs
        train_logs = trainer.state.log_history
        train_loss = None
        for log in reversed(train_logs):
            if 'loss' in log:
                train_loss = log['loss']
                break
        
        # EVALUAR en val set (calcula val_loss automáticamente)
        eval_result = trainer.evaluate()
        val_loss = eval_result.get('eval_loss', None)
        
        # CALCULAR MÉTRICAS NLP (BLEU-4 y ROUGE-L)
        # Esto genera reportes para 50 muestras del val set
        print(f"      📝 Generando reportes para calcular BLEU-4 y ROUGE-L...")
        metrics = evaluate_model_on_val(
            model, 
            val_dataset, 
            device, 
            num_samples=GRID_CONFIG['eval_samples']
        )
        
        # Registrar resultados con timestamp
        epoch_result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'combo_num': combo_num,
            'r': r,
            'alpha': alpha,
            'lr': lr,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'bleu4': metrics['bleu4'],
            'rougeL': metrics['rougeL']
        }
        results_per_epoch.append(epoch_result)
        
        print(f"      ✓ Train Loss: {train_loss:.4f}")
        print(f"      ✓ Val Loss: {val_loss:.4f}")
        print(f"      ✓ BLEU-4: {metrics['bleu4']:.4f} (precisión 4-gramas)")
        print(f"      ✓ ROUGE-L: {metrics['rougeL']:.4f} (subsecuencia común)")
        
        # LIMPIAR MEMORIA después de cada época (crítico para evitar OOM)
        gc.collect()
        torch.cuda.empty_cache()
    
    # ========================================================================
    # Liberar modelo completamente
    # ========================================================================
    print(f"\n   🧹 Liberando memoria del modelo...")
    del model
    del base_model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    # Eliminar directorio temporal
    import shutil
    shutil.rmtree(temp_output, ignore_errors=True)
    
    return results_per_epoch

# ============================================================================
# GRID SEARCH PRINCIPAL
# ============================================================================

print("="*80)
print("🚀 INICIANDO GRID SEARCH")
print("="*80 + "\n")

# Generar todas las combinaciones
combinations = []
for r in GRID_CONFIG['r_values']:
    for alpha_factor in GRID_CONFIG['alpha_factors']:
        alpha = r * alpha_factor
        for lr in GRID_CONFIG['lr_values']:
            combinations.append((r, alpha, lr))

total_combos = len(combinations)
print(f"📊 Total de combinaciones: {total_combos}\n")

# ============================================================================
# CARGAR RESULTADOS PREVIOS (CHECKPOINT RESUME)
# ============================================================================
all_results = []
completed_combos = set()

checkpoint_file = OUTPUT_DIR / "intermediate_results.csv"
if checkpoint_file.exists():
    print("🔄 Detectado checkpoint previo, cargando resultados...\n")
    df_checkpoint = pd.read_csv(checkpoint_file)
    
    # Convertir a lista de dicts
    all_results = df_checkpoint.to_dict('records')
    
    # Identificar combinaciones COMPLETAS (con 4 épocas)
    for r, alpha, lr in combinations:
        combo_data = df_checkpoint[(df_checkpoint['r'] == r) & 
                                    (df_checkpoint['alpha'] == alpha) & 
                                    (df_checkpoint['lr'] == lr)]
        if len(combo_data) == GRID_CONFIG['num_epochs']:
            completed_combos.add((r, alpha, lr))
            print(f"   ✅ Combinación completa: r={r}, alpha={alpha}, lr={lr:.0e}")
    
    print(f"\n📊 Resumen checkpoint:")
    print(f"   • Completas: {len(completed_combos)}/{total_combos}")
    print(f"   • Pendientes: {total_combos - len(completed_combos)}/{total_combos}\n")
else:
    print("📝 No hay checkpoint previo, iniciando desde cero.\n")

# Entrenar cada combinación
start_time = datetime.now()

for i, (r, alpha, lr) in enumerate(combinations, 1):
    # SALTAR si ya está completa
    if (r, alpha, lr) in completed_combos:
        print(f"\n{'='*80}")
        print(f"⏭️  COMBINACIÓN {i}/{total_combos} - SALTANDO (ya completa)")
        print(f"   r={r}, alpha={alpha}, lr={lr:.2e}")
        print(f"{'='*80}\n")
        continue
    
    try:
        results = train_single_combination(r, alpha, lr, i, total_combos)
        all_results.extend(results)
        
        # GUARDAR CHECKPOINT INTERMEDIO (por si falla o se interrumpe)
        df_intermediate = pd.DataFrame(all_results)
        df_intermediate.to_csv(OUTPUT_DIR / "intermediate_results.csv", index=False)
        
        # Guardar progreso para monitoreo en tiempo real
        progress_file = OUTPUT_DIR / "progress.txt"
        with open(progress_file, 'w') as f:
            f.write(f"Completado: {i}/{total_combos} combinaciones\n")
            f.write(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Progreso: {(i/total_combos)*100:.1f}%\n")
            elapsed = datetime.now() - start_time
            f.write(f"Tiempo transcurrido: {elapsed}\n")
            if i > 0:
                avg_time = elapsed / i
                remaining = avg_time * (total_combos - i)
                f.write(f"Tiempo estimado restante: {remaining}\n")
            f.write(f"Progreso: {(i/total_combos)*100:.1f}%\n")
        
    except Exception as e:
        print(f"\n❌ ERROR en combinación r={r}, alpha={alpha}, lr={lr:.2e}")
        print(f"   {e}")
        print("   Continuando con siguiente combinación...\n")
        continue

end_time = datetime.now()
duration = end_time - start_time

print("\n" + "="*80)
print("✅ GRID SEARCH COMPLETADO")
print("="*80)
print(f"\n⏱️  Duración total: {duration}")
print(f"📊 Combinaciones evaluadas: {len(set([(r['r'], r['alpha'], r['lr']) for r in all_results]))}")

# ============================================================================
# GENERAR RESUMEN Y GRÁFICOS
# ============================================================================

print("\n📊 Generando análisis de resultados...\n")

# Verificar que hay resultados
if len(all_results) == 0:
    print("⚠️  No se completó ninguna combinación. Grid search interrumpido antes de tiempo.")
    print("📁 No se generaron archivos de resultados finales.")
    print("\nSaliendo...")
    exit(0)

# Crear DataFrame completo
df_all = pd.DataFrame(all_results)

# Calcular métricas finales por combinación (última época)
df_final = df_all[df_all['epoch'] == GRID_CONFIG['num_epochs']].copy()

# Ordenar por val_loss (menor es mejor)
df_final = df_final.sort_values('val_loss').reset_index(drop=True)

# Guardar resultados completos
df_all.to_csv(OUTPUT_DIR / "all_epochs_results.csv", index=False)
df_final.to_csv(OUTPUT_DIR / "final_results_ranked.csv", index=False)

print(f"💾 Resultados guardados:")
print(f"   • all_epochs_results.csv (todas las épocas)")
print(f"   • final_results_ranked.csv (ordenado por mejor val_loss)")

# Mostrar top 5
print("\n🏆 TOP 5 MEJORES COMBINACIONES (por val_loss):\n")
print(df_final[['r', 'alpha', 'lr', 'val_loss', 'bleu4', 'rougeL']].head(5).to_string(index=False))

# ============================================================================
# GENERAR GRÁFICOS
# ============================================================================

print(f"\n📈 Generando gráficos para las {total_combos} combinaciones...")

plots_dir = OUTPUT_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

for r, alpha, lr in combinations:
    # Filtrar datos de esta combinación
    df_combo = df_all[(df_all['r'] == r) & (df_all['alpha'] == alpha) & (df_all['lr'] == lr)]
    
    if len(df_combo) == 0:
        continue
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df_combo['epoch'], df_combo['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax.plot(df_combo['epoch'], df_combo['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
    
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
    ax.set_ylabel('Loss', fontweight='bold', fontsize=12)
    ax.set_title(f'Loss Curves: r={r}, alpha={alpha}, lr={lr:.2e}', fontweight='bold', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Anotar métricas finales
    final_val_loss = df_combo['val_loss'].iloc[-1]
    final_bleu4 = df_combo['bleu4'].iloc[-1]
    final_rougeL = df_combo['rougeL'].iloc[-1]
    
    textstr = f'Final Val Loss: {final_val_loss:.4f}\nBLEU-4: {final_bleu4:.4f}\nROUGE-L: {final_rougeL:.4f}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Guardar
    plot_name = f"loss_r{r}_a{alpha}_lr{lr:.0e}.png"
    plt.savefig(plots_dir / plot_name, dpi=150, bbox_inches='tight')
    plt.close()

print(f"   ✅ {total_combos} gráficos guardados en: {plots_dir}")

# ============================================================================
# GRÁFICO COMPARATIVO FINAL
# ============================================================================

print("\n📊 Generando gráfico comparativo global...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Val Loss por combinación
ax1 = axes[0, 0]
x_labels = [f"r={r}\na={a}\nlr={lr:.0e}" for r, a, lr in zip(df_final['r'], df_final['alpha'], df_final['lr'])]
ax1.barh(range(len(df_final)), df_final['val_loss'], color='steelblue')
ax1.set_yticks(range(len(df_final)))
ax1.set_yticklabels(x_labels, fontsize=8)
ax1.set_xlabel('Val Loss', fontweight='bold')
ax1.set_title('Val Loss por Combinación (Ordenado)', fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# BLEU-4 por combinación
ax2 = axes[0, 1]
ax2.barh(range(len(df_final)), df_final['bleu4'], color='green')
ax2.set_yticks(range(len(df_final)))
ax2.set_yticklabels(x_labels, fontsize=8)
ax2.set_xlabel('BLEU-4', fontweight='bold')
ax2.set_title('BLEU-4 por Combinación', fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

# ROUGE-L por combinación
ax3 = axes[1, 0]
ax3.barh(range(len(df_final)), df_final['rougeL'], color='orange')
ax3.set_yticks(range(len(df_final)))
ax3.set_yticklabels(x_labels, fontsize=8)
ax3.set_xlabel('ROUGE-L', fontweight='bold')
ax3.set_title('ROUGE-L por Combinación', fontweight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# Scatter: Val Loss vs BLEU-4
ax4 = axes[1, 1]
scatter = ax4.scatter(df_final['val_loss'], df_final['bleu4'], 
                      c=df_final['r'], cmap='viridis', s=100, alpha=0.7)
ax4.set_xlabel('Val Loss', fontweight='bold')
ax4.set_ylabel('BLEU-4', fontweight='bold')
ax4.set_title('Val Loss vs BLEU-4 (color = r)', fontweight='bold')
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='r value')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparison_summary.png", dpi=300, bbox_inches='tight')
print(f"   ✅ Gráfico comparativo: comparison_summary.png")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("🎉 GRID SEARCH FINALIZADO")
print("="*80)

best = df_final.iloc[0]
print(f"\n🏆 MEJOR COMBINACIÓN:")
print(f"   • r: {best['r']}")
print(f"   • alpha: {best['alpha']}")
print(f"   • lr: {best['lr']:.2e}")
print(f"   • Val Loss: {best['val_loss']:.4f}")
print(f"   • BLEU-4: {best['bleu4']:.4f}")
print(f"   • ROUGE-L: {best['rougeL']:.4f}")

print(f"\n📁 Todos los archivos en: {OUTPUT_DIR}")
print(f"   • final_results_ranked.csv")
print(f"   • all_epochs_results.csv")
print(f"   • plots/ ({total_combos} gráficos)")
print(f"   • comparison_summary.png")

print("\n" + "="*80 + "\n")

# ============================================================================
# AUTO-PAUSAR POD DE RUNPOD (si está disponible)
# ============================================================================

print("🛑 Intentando pausar pod de RunPod...")

try:
    # Método 1: Usando runpodctl (si está instalado)
    import subprocess
    result = subprocess.run(['runpodctl', 'stop', 'pod'], 
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print("✅ Pod pausado exitosamente con runpodctl")
    else:
        # Método 2: Usando variable de entorno RUNPOD_POD_ID y API
        import requests
        pod_id = os.environ.get('RUNPOD_POD_ID')
        api_key = os.environ.get('RUNPOD_API_KEY')
        
        if pod_id and api_key:
            url = f"https://api.runpod.io/graphql"
            headers = {"Content-Type": "application/json"}
            query = f"""
            mutation {{
              podStop(input: {{podId: "{pod_id}"}}) {{
                id
                desiredStatus
              }}
            }}
            """
            response = requests.post(url, 
                                   json={"query": query},
                                   headers={"Authorization": api_key})
            if response.status_code == 200:
                print("✅ Pod pausado exitosamente con API de RunPod")
            else:
                print(f"⚠️  No se pudo pausar automáticamente. Pausa manualmente desde RunPod UI.")
        else:
            print("⚠️  Variables RUNPOD_POD_ID o RUNPOD_API_KEY no encontradas.")
            print("💡 Pausa manualmente el pod desde: https://www.runpod.io/console/pods")
except Exception as e:
    print(f"⚠️  No se pudo pausar automáticamente: {e}")
    print("💡 Pausa manualmente el pod desde: https://www.runpod.io/console/pods")

print("\n✅ Grid search completado. Es seguro cerrar esta terminal.\n")
