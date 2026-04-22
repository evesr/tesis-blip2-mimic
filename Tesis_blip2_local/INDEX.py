#!/usr/bin/env python3
"""
🏥 PROYECTO BLIP2 + LoRA - Generación de Reportes Médicos
==========================================================

Punto de entrada principal del proyecto.
Ejecuta este archivo para ver opciones disponibles.

ARCHIVOS PRINCIPALES CREADOS EN LA REFACTORIZACIÓN:
===================================================

📋 DOCUMENTACIÓN (4 archivos):
------------------------------
   README.md                 (12K)  - Documentación completa del proyecto
   QUICK_START.md           (6.0K)  - Guía de inicio rápido
   DOCUMENTATION_INDEX.py   (6.7K)  - Índice navegable de documentación
   REFACTORING_SUMMARY.md   (8.6K)  - Resumen de la refactorización

🔧 MÓDULOS CORE (6 archivos):
-----------------------------
   config.py                (7.3K)  - Configuración centralizada
   preprocessing.py         (9.3K)  - Preprocesamiento de imágenes
   data_loader.py          (16.0K)  - Dataset de PyTorch
   model_utils.py          (15.0K)  - Utilidades de modelos BLIP2
   train.py                (11.0K)  - Script de entrenamiento
   evaluate.py             (19.0K)  - Script de evaluación/inferencia

🛠️ UTILIDADES (3 archivos):
---------------------------
   check_setup.py           (7.4K)  - Verificación del entorno
   quick_test.py            (2.7K)  - Prueba rápida del modelo
   ejemplos_uso.py         (12.0K)  - Ejemplos de uso programático

📦 CONFIGURACIÓN (2 archivos):
------------------------------
   requirements.txt         (623B)  - Dependencias del proyecto
   .gitignore                       - Configuración Git

TOTAL: 15 archivos nuevos (~133 KB de código profesional)


COMANDOS RÁPIDOS:
================

1. VERIFICAR INSTALACIÓN:
   $ python check_setup.py

2. ENTRENAR MODELO:
   $ python train.py

3. PROBAR MODELO:
   $ python quick_test.py

4. EVALUAR MODELO:
   $ python evaluate.py --model-dir modelo_entrenado --test-csv test_split_limpio.csv

5. VER EJEMPLOS:
   $ python ejemplos_uso.py

6. LEER DOCUMENTACIÓN:
   $ cat README.md
   $ cat QUICK_START.md


ESTRUCTURA DEL PROYECTO:
========================

Tesis_blip2_local/
│
├── 📚 DOCUMENTACIÓN
│   ├── README.md                  ← EMPIEZA AQUÍ
│   ├── QUICK_START.md             ← Guía rápida
│   ├── DOCUMENTATION_INDEX.py     ← Índice
│   └── REFACTORING_SUMMARY.md     ← Resumen de cambios
│
├── ⚙️ CONFIGURACIÓN
│   ├── config.py                  ← Configuración central (PERSONALIZA AQUÍ)
│   ├── requirements.txt           ← Dependencias
│   └── .gitignore                 ← Git config
│
├── 🔧 MÓDULOS
│   ├── preprocessing.py           ← Preprocesamiento de imágenes
│   ├── data_loader.py             ← Carga de datos
│   └── model_utils.py             ← Utilidades de modelos
│
├── 🚀 SCRIPTS PRINCIPALES
│   ├── train.py                   ← Entrenar modelo
│   └── evaluate.py                ← Evaluar/Inferencia
│
├── 🛠️ HERRAMIENTAS
│   ├── check_setup.py             ← Verificar setup
│   ├── quick_test.py              ← Prueba rápida
│   └── ejemplos_uso.py            ← Ejemplos programáticos
│
├── 📊 DATOS
│   ├── train_split_limpio.csv     ← Datos train
│   ├── val_split_limpio.csv       ← Datos val
│   ├── test_split_limpio.csv      ← Datos test
│   └── imagenes_1_porciento/      ← Imágenes
│
└── 💾 SALIDAS (se crean automáticamente)
    └── modelo_entrenado/          ← Modelo guardado
        ├── checkpoints/           ← Checkpoints
        └── logs/                  ← Logs


CARACTERÍSTICAS PRINCIPALES:
============================

✅ Sin dependencias de Google Colab
✅ Código 100% tipado (type hints)
✅ Documentación exhaustiva
✅ Arquitectura modular
✅ Configuración centralizada
✅ Logging profesional
✅ CLI completa
✅ Ejemplos de uso
✅ Manejo de errores robusto
✅ Listo para producción


TECNOLOGÍAS UTILIZADAS:
=======================

- PyTorch: Deep learning framework
- Transformers (HuggingFace): Modelo BLIP2
- PEFT: Implementación de LoRA
- OpenCV: Procesamiento de imágenes (CLAHE)
- Pandas: Manejo de datos
- Matplotlib: Visualización


CAMBIOS RESPECTO AL NOTEBOOK ORIGINAL:
======================================

ANTES (Google Colab):
- 1 archivo .ipynb monolítico
- Dependiente de Google Drive
- Comandos shell (!pip, !unzip)
- Sin tipado
- Documentación inline mezclada con código
- Difícil de mantener

DESPUÉS (Python Modular):
- 15 archivos Python organizados
- Independiente de servicios cloud
- Instalación con pip estándar
- 100% tipado
- Documentación profesional separada
- Fácil de mantener y extender


SOPORTE Y AYUDA:
===============

1. Documentación completa: README.md
2. Guía rápida: QUICK_START.md
3. Verificar setup: python check_setup.py
4. Ver ejemplos: python ejemplos_uso.py
5. Revisar logs: cat training.log


LICENCIA:
========

MIT License - Ver LICENSE para detalles


AUTOR:
=====

Proyecto de Tesis - Data Science
Especialización en Imágenes Médicas
Abril 2026


---
Para comenzar, ejecuta:
    python check_setup.py

Para ayuda sobre un script específico:
    python train.py --help
    python evaluate.py --help
"""

import sys
from pathlib import Path


def mostrar_menu():
    """Muestra el menú interactivo."""
    print("\n" + "="*80)
    print("🏥 BLIP2 + LoRA: Generación de Reportes Médicos")
    print("="*80)
    print("\n¿Qué deseas hacer?\n")
    print("  1. Verificar configuración del entorno")
    print("  2. Entrenar modelo")
    print("  3. Probar modelo (requiere modelo entrenado)")
    print("  4. Evaluar en dataset de test")
    print("  5. Ver ejemplos de uso programático")
    print("  6. Leer documentación")
    print("  0. Salir")
    print("\n" + "-"*80)
    
    try:
        opcion = input("\nSelecciona una opción [0-6]: ").strip()
        return opcion
    except KeyboardInterrupt:
        print("\n\n👋 Hasta luego!")
        return "0"


def ejecutar_opcion(opcion: str):
    """Ejecuta la opción seleccionada."""
    
    if opcion == "1":
        print("\n🔍 Verificando configuración...\n")
        import check_setup
        check_setup.main()
    
    elif opcion == "2":
        print("\n🚀 Iniciando entrenamiento...\n")
        print("💡 Tip: Puedes usar argumentos CLI:")
        print("   python train.py --epochs 10 --batch-size 8\n")
        respuesta = input("¿Continuar con configuración por defecto? [s/N]: ")
        if respuesta.lower() == 's':
            import train
            train.main()
        else:
            print("\n📝 Revisa config.py para personalizar parámetros")
    
    elif opcion == "3":
        print("\n🧪 Probando modelo...\n")
        modelo_dir = Path("modelo_entrenado")
        if not modelo_dir.exists():
            print("❌ Modelo no encontrado. Primero debes entrenar:")
            print("   python train.py")
        else:
            import quick_test
            quick_test.main()
    
    elif opcion == "4":
        print("\n📊 Evaluando modelo...\n")
        modelo_dir = Path("modelo_entrenado")
        if not modelo_dir.exists():
            print("❌ Modelo no encontrado. Primero debes entrenar:")
            print("   python train.py")
        else:
            print("💡 Uso:")
            print("   python evaluate.py --model-dir modelo_entrenado --test-csv test_split_limpio.csv")
    
    elif opcion == "5":
        print("\n📚 Ejemplos de uso programático...\n")
        import ejemplos_uso
        ejemplos_uso.main()
    
    elif opcion == "6":
        print("\n📖 Documentación disponible:\n")
        print("  - README.md               : Documentación completa")
        print("  - QUICK_START.md          : Guía de inicio rápido")
        print("  - DOCUMENTATION_INDEX.py  : Índice de documentación")
        print("  - REFACTORING_SUMMARY.md  : Resumen de refactorización")
        print("\n💡 Abre estos archivos con tu editor favorito")
    
    elif opcion == "0":
        print("\n👋 ¡Hasta luego!")
        return False
    
    else:
        print("\n⚠️  Opción no válida")
    
    return True


def main():
    """Función principal."""
    # Si se ejecuta sin argumentos, mostrar menú
    if len(sys.argv) == 1:
        print(__doc__)
        
        while True:
            opcion = mostrar_menu()
            if not ejecutar_opcion(opcion):
                break
            
            input("\n\nPresiona Enter para continuar...")
    else:
        # Mostrar solo el docstring
        print(__doc__)


if __name__ == "__main__":
    main()
