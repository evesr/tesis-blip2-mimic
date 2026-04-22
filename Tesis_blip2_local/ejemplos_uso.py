"""
Ejemplo de uso programático del modelo BLIP2.

Este script muestra cómo usar los módulos del proyecto de forma programática
(no desde línea de comandos) para integrar en otros sistemas.

Author: [Tu nombre]
Date: 2026-04-22
"""

from pathlib import Path
from typing import List, Dict

# Importar módulos del proyecto
from config import config
from model_utils import (
    cargar_procesador,
    cargar_modelo_completo,
    guardar_modelo,
    cargar_modelo_entrenado
)
from preprocessing import estandarizar_imagen
from data_loader import MimicCXRDataset, cargar_splits
from evaluate import GeneradorReportes, visualizar_prediccion


# ============================================================================
# EJEMPLO 1: Entrenar un modelo programáticamente
# ============================================================================

def ejemplo_entrenamiento_basico():
    """Ejemplo de cómo entrenar el modelo de forma programática."""
    
    print("\n" + "="*80)
    print("EJEMPLO 1: Entrenamiento Básico")
    print("="*80)
    
    from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
    
    # 1. Configurar rutas (opcional, usa config por defecto)
    config.training.num_train_epochs = 5  # Menos épocas para ejemplo
    config.paths.create_directories()
    
    # 2. Cargar procesador
    print("\n📥 Cargando procesador...")
    processor = cargar_procesador()
    
    # 3. Cargar modelo con LoRA
    print("\n🤖 Cargando modelo con LoRA...")
    model = cargar_modelo_completo(use_lora=True)
    
    # 4. Cargar datasets
    print("\n📂 Cargando datasets...")
    train_dataset, val_dataset, _ = cargar_splits(
        processor=processor,
        train_csv=config.paths.train_csv,
        val_csv=config.paths.val_csv
    )
    
    # 5. Configurar entrenamiento
    print("\n⚙️ Configurando entrenamiento...")
    training_args = TrainingArguments(
        output_dir=str(config.paths.checkpoints_dir),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        logging_steps=config.training.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=config.training.fp16,
        remove_unused_columns=False,
        label_names=["labels"]
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        padding=True
    )
    
    # 6. Crear Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    # 7. Entrenar (comentado para no ejecutar realmente)
    # print("\n🔥 Iniciando entrenamiento...")
    # trainer.train()
    
    # 8. Guardar modelo
    # guardar_modelo(trainer.model, processor, config.paths.output_dir)
    
    print("\n✅ Ejemplo de entrenamiento completado (sin ejecutar)")
    print("   Descomenta las líneas para entrenar realmente")


# ============================================================================
# EJEMPLO 2: Generar reportes para múltiples imágenes
# ============================================================================

def ejemplo_inferencia_batch(imagenes: List[Path], model_dir: Path):
    """
    Genera reportes para un lote de imágenes.
    
    Args:
        imagenes: Lista de rutas a imágenes
        model_dir: Directorio con el modelo entrenado
    """
    
    print("\n" + "="*80)
    print("EJEMPLO 2: Inferencia en Batch")
    print("="*80)
    
    # Inicializar generador
    print("\n🔄 Inicializando generador...")
    generador = GeneradorReportes(model_dir=model_dir)
    
    # Generar reportes
    resultados = []
    
    for i, imagen_path in enumerate(imagenes, 1):
        print(f"\n📸 Procesando imagen {i}/{len(imagenes)}: {imagen_path.name}")
        
        try:
            reporte = generador.generar_desde_imagen(imagen_path)
            
            resultados.append({
                'imagen': imagen_path.name,
                'reporte': reporte,
                'longitud': len(reporte)
            })
            
            print(f"   ✅ Generado: {len(reporte)} caracteres")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return resultados


# ============================================================================
# EJEMPLO 3: Comparar estrategias de generación
# ============================================================================

def ejemplo_comparar_estrategias(imagen_path: Path, model_dir: Path):
    """
    Compara diferentes estrategias de generación para una imagen.
    
    Args:
        imagen_path: Ruta a la imagen
        model_dir: Directorio con el modelo entrenado
    """
    
    print("\n" + "="*80)
    print("EJEMPLO 3: Comparación de Estrategias")
    print("="*80)
    
    # Inicializar generador
    generador = GeneradorReportes(model_dir=model_dir)
    
    # Definir estrategias
    estrategias = [
        {
            'nombre': 'Greedy (Rápido)',
            'params': {
                'do_sample': False,
                'num_beams': 1,
                'max_new_tokens': 150
            }
        },
        {
            'nombre': 'Beam Search (Mejor Calidad)',
            'params': {
                'do_sample': False,
                'num_beams': 5,
                'max_new_tokens': 150
            }
        },
        {
            'nombre': 'Sampling (Creativo)',
            'params': {
                'do_sample': True,
                'temperature': 0.8,
                'top_p': 0.95,
                'max_new_tokens': 150
            }
        }
    ]
    
    # Generar con cada estrategia
    resultados = {}
    
    for estrategia in estrategias:
        nombre = estrategia['nombre']
        params = estrategia['params']
        
        print(f"\n🎯 Probando: {nombre}")
        reporte = generador.generar_desde_imagen(imagen_path, **params)
        resultados[nombre] = reporte
        
        print(f"   Longitud: {len(reporte)} caracteres")
        print(f"   Preview: {reporte[:100]}...")
    
    return resultados


# ============================================================================
# EJEMPLO 4: Preprocesar imágenes manualmente
# ============================================================================

def ejemplo_preprocesamiento():
    """Ejemplo de preprocesamiento manual de imágenes."""
    
    print("\n" + "="*80)
    print("EJEMPLO 4: Preprocesamiento Manual")
    print("="*80)
    
    # Buscar una imagen de ejemplo
    imagenes = list(config.paths.images_dir.rglob("*.jpg"))
    
    if not imagenes:
        print("❌ No se encontraron imágenes")
        return
    
    imagen_path = imagenes[0]
    print(f"\n📸 Procesando: {imagen_path.name}")
    
    # Preprocesar con configuración personalizada
    imagen_procesada = estandarizar_imagen(
        imagen_path,
        target_size=(384, 384),  # Tamaño personalizado
        clahe_clip_limit=3.0,     # Mayor contraste
        clahe_tile_grid_size=(16, 16)  # Grid más fino
    )
    
    if imagen_procesada:
        print(f"   ✅ Imagen procesada: {imagen_procesada.size}, modo: {imagen_procesada.mode}")
        # Aquí podrías guardar o visualizar la imagen
        # imagen_procesada.save("imagen_procesada.jpg")
    else:
        print("   ❌ Error en preprocesamiento")


# ============================================================================
# EJEMPLO 5: Cargar dataset y explorar datos
# ============================================================================

def ejemplo_explorar_dataset():
    """Ejemplo de cómo explorar el dataset."""
    
    print("\n" + "="*80)
    print("EJEMPLO 5: Explorar Dataset")
    print("="*80)
    
    from transformers import Blip2Processor
    
    # Cargar procesador
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    # Crear dataset
    print("\n📂 Cargando dataset...")
    dataset = MimicCXRDataset(
        csv_path=config.paths.train_csv,
        images_dir=config.paths.images_dir,
        processor=processor
    )
    
    print(f"   ✅ Dataset: {len(dataset)} muestras")
    
    # Explorar algunas muestras
    print("\n🔍 Explorando muestras:")
    
    for i in range(min(3, len(dataset))):
        info = dataset.get_sample_info(i)
        
        print(f"\n   Muestra {i}:")
        print(f"      - DICOM ID: {info['dicom_id']}")
        print(f"      - Patient: {info['subject_id']}")
        print(f"      - Study: {info['study_id']}")
        print(f"      - Report length: {info['report_length']} chars")
        print(f"      - Preview: {info['report_preview'][:80]}...")


# ============================================================================
# EJEMPLO 6: Personalizar configuración dinámicamente
# ============================================================================

def ejemplo_configuracion_personalizada():
    """Ejemplo de cómo personalizar la configuración."""
    
    print("\n" + "="*80)
    print("EJEMPLO 6: Configuración Personalizada")
    print("="*80)
    
    # Ver configuración actual
    print("\n📋 Configuración actual:")
    print(f"   - Épocas: {config.training.num_train_epochs}")
    print(f"   - Batch size: {config.training.per_device_train_batch_size}")
    print(f"   - Learning rate: {config.training.learning_rate}")
    
    # Modificar configuración
    print("\n⚙️ Modificando configuración...")
    config.training.num_train_epochs = 20
    config.training.learning_rate = 5e-5
    config.lora.r = 32  # Mayor capacidad
    
    print("\n✅ Configuración modificada:")
    print(f"   - Épocas: {config.training.num_train_epochs}")
    print(f"   - Learning rate: {config.training.learning_rate}")
    print(f"   - LoRA rank: {config.lora.r}")
    
    # Restaurar valores por defecto si es necesario
    from config import Config
    config_nueva = Config()  # Crear nueva instancia con valores por defecto


# ============================================================================
# FUNCIÓN PRINCIPAL DE DEMOSTRACIÓN
# ============================================================================

def main():
    """Ejecuta todos los ejemplos (algunos comentados para no ejecutar realmente)."""
    
    print("\n" + "="*80)
    print("🎓 EJEMPLOS DE USO PROGRAMÁTICO - BLIP2 + LoRA")
    print("="*80)
    
    # Ejemplo 1: Entrenamiento (sin ejecutar realmente)
    # ejemplo_entrenamiento_basico()
    
    # Ejemplo 4: Preprocesamiento
    ejemplo_preprocesamiento()
    
    # Ejemplo 5: Explorar dataset
    # ejemplo_explorar_dataset()
    
    # Ejemplo 6: Configuración personalizada
    ejemplo_configuracion_personalizada()
    
    # Los siguientes ejemplos requieren un modelo entrenado:
    # model_dir = Path("modelo_entrenado")
    # 
    # if model_dir.exists():
    #     # Ejemplo 2: Inferencia batch
    #     imagenes = list(config.paths.images_dir.rglob("*.jpg"))[:3]
    #     resultados = ejemplo_inferencia_batch(imagenes, model_dir)
    #     
    #     # Ejemplo 3: Comparar estrategias
    #     if imagenes:
    #         comparacion = ejemplo_comparar_estrategias(imagenes[0], model_dir)
    # else:
    #     print("\n⚠️  Modelo no encontrado. Entrena primero con: python train.py")
    
    print("\n" + "="*80)
    print("✅ Ejemplos completados")
    print("="*80)
    print("\n💡 Tips:")
    print("   - Descomenta los ejemplos que quieras ejecutar")
    print("   - Modifica los parámetros según tus necesidades")
    print("   - Lee los docstrings para más detalles")
    print("   - Consulta README.md para documentación completa")


if __name__ == "__main__":
    main()
