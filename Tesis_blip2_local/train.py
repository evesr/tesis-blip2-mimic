"""
Script de entrenamiento del modelo BLIP2 con LoRA.

Este script orquesta el proceso completo de entrenamiento:
- Carga de datos (train/val splits)
- Configuración del modelo con LoRA
- Entrenamiento con Hugging Face Trainer
- Guardado de checkpoints y modelo final

Author: [Tu nombre]
Date: 2026-04-22
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional
import logging

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

from config import config
from model_utils import (
    cargar_procesador,
    cargar_modelo_completo,
    guardar_modelo,
    imprimir_info_modelo
)
from data_loader import cargar_splits

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def configurar_argumentos_entrenamiento(
    output_dir: Path,
    checkpoints_dir: Optional[Path] = None
) -> TrainingArguments:
    """
    Configura los argumentos para el entrenamiento con Hugging Face Trainer.
    
    Args:
        output_dir: Directorio principal de salida
        checkpoints_dir: Directorio específico para checkpoints (usa config si es None)
        
    Returns:
        TrainingArguments configurado
        
    Notes:
        - Usa parámetros de config.training
        - Configura estrategias de guardado y evaluación
        - Habilita FP16 para mayor eficiencia
    """
    if checkpoints_dir is None:
        checkpoints_dir = config.paths.checkpoints_dir
    
    logger.info("⚙️ Configurando argumentos de entrenamiento...")
    
    training_args = TrainingArguments(
        # Directorios
        output_dir=str(checkpoints_dir),
        
        # Batch sizes y acumulación
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        
        # Optimización
        learning_rate=config.training.learning_rate,
        num_train_epochs=config.training.num_train_epochs,
        warmup_steps=config.training.warmup_steps,
        
        # Logging y guardado
        logging_dir=str(output_dir / "logs"),
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        eval_strategy=config.training.eval_strategy,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        
        # Precisión y optimización
        fp16=config.training.fp16,
        
        # Métricas
        metric_for_best_model="loss",
        greater_is_better=False,
        
        # Otros
        report_to=config.training.report_to,
        remove_unused_columns=False,
        label_names=["labels"],
        seed=config.training.seed,
        
        # Evaluación
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
    )
    
    logger.info(f"   - Épocas: {config.training.num_train_epochs}")
    logger.info(f"   - Batch size efectivo: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
    logger.info(f"   - Learning rate: {config.training.learning_rate}")
    logger.info(f"   - FP16: {config.training.fp16}")
    
    return training_args


def entrenar_modelo(
    train_csv: Optional[Path] = None,
    val_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    resume_from_checkpoint: Optional[str] = None
) -> Trainer:
    """
    Función principal de entrenamiento.
    
    Args:
        train_csv: Ruta al CSV de entrenamiento (usa config si es None)
        val_csv: Ruta al CSV de validación (usa config si es None)
        output_dir: Directorio de salida (usa config si es None)
        resume_from_checkpoint: Ruta a checkpoint para continuar entrenamiento
        
    Returns:
        Trainer entrenado
        
    Example:
        >>> trainer = entrenar_modelo(
        ...     train_csv=Path("train_split_limpio.csv"),
        ...     val_csv=Path("val_split_limpio.csv")
        ... )
    """
    # Usar configuración por defecto si no se especifica
    if train_csv is None:
        train_csv = config.paths.train_csv
    if val_csv is None:
        val_csv = config.paths.val_csv
    if output_dir is None:
        output_dir = config.paths.output_dir
    
    # Crear directorios
    config.paths.create_directories()
    
    logger.info("\n" + "="*80)
    logger.info("🚀 INICIANDO ENTRENAMIENTO BLIP2 + LoRA")
    logger.info("="*80)
    
    # 1. Cargar procesador
    logger.info("\n📥 PASO 1/5: Cargando procesador...")
    processor = cargar_procesador(model_name=config.model.model_name)
    
    # 2. Cargar modelo
    logger.info("\n🤖 PASO 2/5: Cargando modelo con LoRA...")
    model = cargar_modelo_completo(
        use_lora=True,
        model_name=config.model.model_name,
        use_quantization=config.model.load_in_8bit
    )
    
    # Mostrar información del modelo
    imprimir_info_modelo(model)
    
    # 3. Cargar datasets
    logger.info("\n📂 PASO 3/5: Cargando datasets...")
    train_dataset, val_dataset, _ = cargar_splits(
        processor=processor,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=None
    )
    
    if train_dataset is None:
        raise ValueError(f"No se pudo cargar el dataset de entrenamiento: {train_csv}")
    
    logger.info(f"   ✅ Train: {len(train_dataset)} muestras")
    if val_dataset is not None:
        logger.info(f"   ✅ Val: {len(val_dataset)} muestras")
    else:
        logger.warning("   ⚠️ No hay dataset de validación")
    
    # 4. Configurar entrenamiento
    logger.info("\n⚙️ PASO 4/5: Configurando entrenamiento...")
    training_args = configurar_argumentos_entrenamiento(output_dir)
    
    # Data collator para manejo de batches
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        padding=True
    )
    
    # Crear Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    # 5. Entrenar
    logger.info("\n🔥 PASO 5/5: INICIANDO ENTRENAMIENTO...")
    logger.info(f"   - Duración estimada: ~{config.training.num_train_epochs * 10} minutos")
    logger.info(f"   - Checkpoints se guardarán en: {config.paths.checkpoints_dir}")
    logger.info("\n" + "-"*80)
    
    start_time = time.time()
    
    try:
        # Entrenar
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        end_time = time.time()
        duration_minutes = (end_time - start_time) / 60
        
        logger.info("\n" + "-"*80)
        logger.info(f"✅ ENTRENAMIENTO COMPLETADO")
        logger.info(f"   - Tiempo total: {duration_minutes:.2f} minutos")
        
        # Guardar modelo final
        logger.info("\n💾 Guardando modelo final...")
        guardar_modelo(
            model=trainer.model,
            processor=processor,
            output_dir=output_dir,
            save_full_model=False  # Solo guardar adaptadores LoRA
        )
        
        logger.info(f"\n🎉 ¡PROCESO COMPLETO!")
        logger.info(f"   - Modelo guardado en: {output_dir}")
        logger.info(f"   - Logs en: {output_dir / 'logs'}")
        
        return trainer
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Entrenamiento interrumpido por el usuario")
        logger.info("   - Los checkpoints parciales se han guardado")
        raise
    
    except Exception as e:
        logger.error(f"\n❌ Error durante el entrenamiento: {str(e)}")
        raise


def main():
    """
    Función principal para ejecutar desde línea de comandos.
    
    Example:
        $ python train.py
        $ python train.py --train-csv custom_train.csv --epochs 10
        $ python train.py --resume-from checkpoints/checkpoint-500
    """
    parser = argparse.ArgumentParser(
        description="Entrenar modelo BLIP2 con LoRA para generación de reportes médicos"
    )
    
    parser.add_argument(
        '--train-csv',
        type=str,
        default=None,
        help='Ruta al CSV de entrenamiento (default: usa config)'
    )
    
    parser.add_argument(
        '--val-csv',
        type=str,
        default=None,
        help='Ruta al CSV de validación (default: usa config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directorio de salida (default: usa config)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=f'Número de épocas (default: {config.training.num_train_epochs})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f'Batch size por dispositivo (default: {config.training.per_device_train_batch_size})'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help=f'Learning rate (default: {config.training.learning_rate})'
    )
    
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Ruta a checkpoint para continuar entrenamiento'
    )
    
    parser.add_argument(
        '--no-lora',
        action='store_true',
        help='No usar LoRA (entrenar modelo completo - requiere más memoria)'
    )
    
    args = parser.parse_args()
    
    # Sobrescribir configuración si se especifica
    if args.epochs is not None:
        config.training.num_train_epochs = args.epochs
        logger.info(f"📝 Épocas sobrescritas: {args.epochs}")
    
    if args.batch_size is not None:
        config.training.per_device_train_batch_size = args.batch_size
        logger.info(f"📝 Batch size sobrescrito: {args.batch_size}")
    
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
        logger.info(f"📝 Learning rate sobrescrito: {args.learning_rate}")
    
    # Convertir paths si se especificaron
    train_csv = Path(args.train_csv) if args.train_csv else None
    val_csv = Path(args.val_csv) if args.val_csv else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Entrenar
    try:
        trainer = entrenar_modelo(
            train_csv=train_csv,
            val_csv=val_csv,
            output_dir=output_dir,
            resume_from_checkpoint=args.resume_from
        )
        
        logger.info("\n✅ Script finalizado exitosamente")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Error fatal: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
