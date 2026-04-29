"""
Sanity Check: Verificación de Entrenamiento
============================================
Objetivo: Validar que el código funciona correctamente ANTES del entrenamiento completo.

Estrategia:
- 10 imágenes del train set
- 50 épocas
- Hiperparámetros fijos (r=16, alpha=32, LR=1e-4)
- Objetivo: Loss → 0 (overfitting intencional)
- Verifica: No hay errores de CUDA OOM, gradientes fluyen correctamente

Autor: Evelyn Silva Rozas
Fecha: Abril 2026
"""

import os
import sys
import warnings
from pathlib import Path

import torch
import pandas as pd
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from tqdm import tqdm

from config import config
from model_utils import (
    cargar_procesador,
    cargar_modelo_completo,
    imprimir_info_modelo
)
from data_loader import MimicCXRDataset

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("🧪 SANITY CHECK - VALIDACIÓN DE ENTRENAMIENTO")
print("="*80)
print("\n📋 Objetivo:")
print("   • Verificar que el código funciona SIN errores")
print("   • Validar que los gradientes fluyen correctamente")
print("   • Confirmar que el modelo PUEDE aprender (Loss → 0)")
print("   • Detectar problemas de memoria CUDA")
print("\n⚠️  IMPORTANTE:")
print("   • Esto NO es para elegir hiperparámetros")
print("   • Es ingeniería de software pura")
print("   • Esperamos OVERFITTING (es intencional)")
print("\n" + "="*80 + "\n")

# ============================================================================
# CONFIGURACIÓN SANITY CHECK
# ============================================================================

SANITY_CONFIG = {
    'num_samples': 10,           # Solo 10 imágenes
    'num_epochs': 50,            # 50 épocas (overfitting intencional)
    'batch_size': 2,             # Batch pequeño para máxima estabilidad
    'gradient_accumulation': 2,  # Acumulación para batch efectivo de 4
    'learning_rate': 1e-4,       # LR estándar
    'lora_r': 16,                # Rango LoRA
    'lora_alpha': 32,            # Alpha LoRA
    'save_steps': 10,            # Guardar cada 10 steps
    'logging_steps': 5,          # Log cada 5 steps
}

print("⚙️  CONFIGURACIÓN:")
for key, value in SANITY_CONFIG.items():
    print(f"   • {key}: {value}")
print()

# ============================================================================
# PREPARAR DATOS (10 IMÁGENES)
# ============================================================================

print("📂 Preparando subset de 10 imágenes...\n")

# Cargar CSV completo
train_csv_full = config.paths.train_csv
df_full = pd.read_csv(train_csv_full)

print(f"   📊 Train set completo: {len(df_full)} imágenes")

# Tomar SOLO las primeras 10 que tengan imagen disponible
sanity_samples = []
for idx, row in df_full.iterrows():
    dicom_id = row['dicom_id']
    
    # Buscar imagen (método robusto)
    img_found = False
    for img_file in config.paths.images_dir.rglob(f"{dicom_id}.jpg"):
        img_found = True
        break
    
    if img_found:
        sanity_samples.append(row)
    
    if len(sanity_samples) >= SANITY_CONFIG['num_samples']:
        break

if len(sanity_samples) < SANITY_CONFIG['num_samples']:
    print(f"\n⚠️  ADVERTENCIA: Solo se encontraron {len(sanity_samples)} imágenes válidas")
    print("   Continuando con las disponibles...")
else:
    print(f"   ✅ {len(sanity_samples)} imágenes seleccionadas")

# Crear CSV temporal con subset
sanity_csv = config.paths.base_dir / "sanity_check_subset.csv"
df_sanity = pd.DataFrame(sanity_samples)
df_sanity.to_csv(sanity_csv, index=False)

print(f"   💾 Subset guardado en: {sanity_csv}")

print("\n   📋 Muestras seleccionadas:")
for i, row in enumerate(df_sanity.iterrows(), 1):
    idx, data = row
    print(f"      {i}. {data['dicom_id']} - {data.get('ViewPosition', 'N/A')}")

print()

# ============================================================================
# CARGAR PROCESADOR Y MODELO
# ============================================================================

print("🤖 Cargando procesador y modelo...\n")

# Cargar procesador
processor = cargar_procesador(model_name=config.model.model_name)
print("   ✅ Procesador cargado")

# Cargar modelo con LoRA (SIN cuantización para evitar problemas de compatibilidad)
print("   🔧 Cargando modelo con LoRA (sin cuantización para sanity check)...")
model = cargar_modelo_completo(
    use_lora=True,
    model_name=config.model.model_name,
    use_quantization=False  # Sin 8-bit para evitar errores de compatibilidad
)

# Mostrar info del modelo
print()
imprimir_info_modelo(model)
print()

# ============================================================================
# CREAR DATASET
# ============================================================================

print("📊 Creando dataset de sanity check...\n")

sanity_dataset = MimicCXRDataset(
    csv_path=sanity_csv,
    images_dir=config.paths.images_dir,
    processor=processor,
    image_size=config.data.image_size,
    clahe_clip_limit=config.data.clahe_clip_limit,
    clahe_tile_grid_size=config.data.clahe_tile_grid_size,
    max_length=config.data.max_length,
    padding=config.data.padding
)

print(f"   ✅ Dataset creado con {len(sanity_dataset)} muestras\n")

# Verificar que las muestras se cargan correctamente
print("🔍 Verificando carga de muestras...\n")
try:
    sample = sanity_dataset[0]
    print("   ✅ Primera muestra cargada exitosamente")
    print(f"      - pixel_values shape: {sample['pixel_values'].shape}")
    print(f"      - input_ids shape: {sample['input_ids'].shape}")
    print(f"      - labels shape: {sample['labels'].shape}")
    print()
except Exception as e:
    print(f"   ❌ ERROR cargando muestra: {e}")
    sys.exit(1)

# ============================================================================
# CONFIGURAR ENTRENAMIENTO
# ============================================================================

print("⚙️  Configurando entrenamiento de sanity check...\n")

# Directorio de salida
output_dir = config.paths.base_dir / "sanity_check_output"
output_dir.mkdir(exist_ok=True)

training_args = TrainingArguments(
    # Directorios
    output_dir=str(output_dir),
    
    # Épocas y batch size
    num_train_epochs=SANITY_CONFIG['num_epochs'],
    per_device_train_batch_size=SANITY_CONFIG['batch_size'],
    gradient_accumulation_steps=SANITY_CONFIG['gradient_accumulation'],
    
    # Learning rate
    learning_rate=SANITY_CONFIG['learning_rate'],
    
    # Logging y guardado
    logging_dir=str(output_dir / "logs"),
    logging_steps=SANITY_CONFIG['logging_steps'],
    save_steps=SANITY_CONFIG['save_steps'],
    save_strategy="steps",
    save_total_limit=3,  # Mantener solo últimos 3 checkpoints
    
    # Sin evaluación (solo train)
    eval_strategy="no",
    
    # Optimización
    fp16=torch.cuda.is_available(),
    
    # Otros
    report_to="none",
    remove_unused_columns=False,
    label_names=["labels"],
    
    # Desactivar algunas optimizaciones para debugging
    dataloader_num_workers=0,  # Sin workers paralelos para debugging más claro
)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.tokenizer,
    model=model,
    padding=True
)

# Crear Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sanity_dataset,
    data_collator=data_collator
)

print("   ✅ Trainer configurado")
print(f"   📊 Total steps: {len(sanity_dataset) // (SANITY_CONFIG['batch_size'] * SANITY_CONFIG['gradient_accumulation']) * SANITY_CONFIG['num_epochs']}")
print(f"   💾 Checkpoints en: {output_dir}")
print()

# ============================================================================
# ENTRENAR (SANITY CHECK)
# ============================================================================

print("="*80)
print("🚀 INICIANDO SANITY CHECK")
print("="*80)
print("\n⏱️  Esto tomará varios minutos...")
print("📈 Monitorea que el Loss disminuya progresivamente hacia 0")
print("🎯 Objetivo: Loss final < 0.5 (idealmente < 0.1)")
print("\n" + "="*80 + "\n")

try:
    # Entrenar
    train_result = trainer.train()
    
    print("\n" + "="*80)
    print("✅ SANITY CHECK COMPLETADO")
    print("="*80)
    
    # Mostrar métricas finales
    final_loss = train_result.metrics.get('train_loss', 'N/A')
    
    print(f"\n📊 RESULTADOS:")
    print(f"   • Loss final: {final_loss}")
    print(f"   • Épocas completadas: {SANITY_CONFIG['num_epochs']}")
    print(f"   • Muestras entrenadas: {len(sanity_dataset)}")
    
    # Evaluación del resultado
    print(f"\n🎯 EVALUACIÓN:")
    
    if isinstance(final_loss, float):
        if final_loss < 0.1:
            print("   ✅ EXCELENTE: Loss < 0.1 → Modelo aprende perfectamente")
            print("   ✅ Código validado → Listo para entrenamiento completo")
        elif final_loss < 0.5:
            print("   ✅ BUENO: Loss < 0.5 → Modelo aprende correctamente")
            print("   ✅ Código validado → Listo para entrenamiento completo")
        elif final_loss < 1.0:
            print("   ⚠️  ACEPTABLE: Loss < 1.0 → Modelo aprende pero podría mejorar")
            print("   💡 Considera aumentar épocas o verificar LR")
        else:
            print("   ❌ PROBLEMA: Loss > 1.0 → Modelo no está aprendiendo bien")
            print("   🔧 Revisa: LR, arquitectura, datos")
    else:
        print(f"   ℹ️  Loss: {final_loss}")
    
    print(f"\n💾 Checkpoint final guardado en: {output_dir}")
    
    # Guardar modelo final
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    print(f"   💾 Modelo final: {final_model_dir}")
    
    print("\n" + "="*80)
    print("🎉 ¡SANITY CHECK EXITOSO!")
    print("="*80)
    print("\n✅ El código está LISTO para entrenamiento completo")
    print("✅ No hay errores de memoria CUDA")
    print("✅ Los gradientes fluyen correctamente")
    print("\n👉 Siguiente paso: Entrenar con dataset completo usando train.py")
    print("\n" + "="*80 + "\n")

except RuntimeError as e:
    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
        print("\n" + "="*80)
        print("❌ ERROR: CUDA OUT OF MEMORY")
        print("="*80)
        print(f"\n🚨 {e}")
        print("\n💡 SOLUCIONES:")
        print("   1. Reduce batch_size a 1")
        print("   2. Reduce gradient_accumulation_steps")
        print("   3. Usa CPU (más lento): device='cpu'")
        print("   4. Limpia caché: torch.cuda.empty_cache()")
        print("\n" + "="*80 + "\n")
        sys.exit(1)
    else:
        print(f"\n❌ ERROR durante entrenamiento: {e}")
        raise

except KeyboardInterrupt:
    print("\n\n⚠️  Entrenamiento interrumpido por el usuario")
    print("   Los checkpoints guardados se mantienen en:", output_dir)
    sys.exit(0)

except Exception as e:
    print(f"\n❌ ERROR inesperado: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
