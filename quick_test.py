#!/usr/bin/env python3
"""
Script de prueba rápida del modelo entrenado.

Útil para verificar que el modelo funciona correctamente después del entrenamiento.

Usage:
    python quick_test.py --model-dir modelo_entrenado

Author: [Tu nombre]
Date: 2026-04-22
"""

import sys
import argparse
from pathlib import Path
import random

from evaluate import GeneradorReportes, visualizar_prediccion
from config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Prueba rápida del modelo entrenado"
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='modelo_entrenado',
        help='Directorio con el modelo entrenado'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=3,
        help='Número de imágenes a probar'
    )
    
    args = parser.parse_args()
    
    logger.info("🧪 PRUEBA RÁPIDA DEL MODELO")
    logger.info("="*60)
    
    # Inicializar generador
    try:
        generador = GeneradorReportes(model_dir=args.model_dir)
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
        logger.info("\n💡 Asegúrate de que el modelo esté entrenado:")
        logger.info("   python train.py")
        return 1
    
    # Buscar imágenes
    imagenes = list(config.paths.images_dir.rglob("*.jpg"))
    
    if not imagenes:
        logger.error(f"❌ No se encontraron imágenes en: {config.paths.images_dir}")
        return 1
    
    # Seleccionar aleatorias
    imagenes_prueba = random.sample(imagenes, min(args.num_samples, len(imagenes)))
    
    logger.info(f"\n📸 Probando con {len(imagenes_prueba)} imágenes aleatorias...\n")
    
    # Generar reportes
    for i, imagen_path in enumerate(imagenes_prueba, 1):
        print("\n" + "="*60)
        print(f"🔍 IMAGEN {i}/{len(imagenes_prueba)}")
        print(f"📁 {imagen_path.name}")
        print("="*60)
        
        try:
            reporte = generador.generar_desde_imagen(imagen_path)
            print("\n🤖 REPORTE GENERADO:")
            print("-"*60)
            print(reporte)
            print("-"*60)
            
        except Exception as e:
            logger.error(f"❌ Error generando reporte: {e}")
            continue
    
    print("\n" + "="*60)
    print("✅ Prueba completada")
    print("="*60)
    print("\n💡 Para evaluación completa, usa:")
    print(f"   python evaluate.py --model-dir {args.model_dir} --test-csv test_split_limpio.csv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
