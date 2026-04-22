"""
Índice de Documentación del Proyecto BLIP2 + LoRA

Este archivo sirve como punto de entrada para navegar la documentación
del proyecto.

ESTRUCTURA DE ARCHIVOS
=======================

DOCUMENTACIÓN PRINCIPAL:
------------------------
- README.md          : Documentación completa del proyecto
- QUICK_START.md     : Guía rápida de inicio
- requirements.txt   : Lista de dependencias

MÓDULOS CORE:
-------------
- config.py          : Configuración centralizada (EMPIEZA AQUÍ para personalizar)
- preprocessing.py   : Preprocesamiento de imágenes médicas
- data_loader.py     : Dataset de PyTorch para MIMIC-CXR
- model_utils.py     : Utilidades para manejo de modelos BLIP2

SCRIPTS EJECUTABLES:
--------------------
- train.py          : Script principal de entrenamiento
- evaluate.py       : Script de evaluación e inferencia
- check_setup.py    : Verificación de configuración del entorno
- quick_test.py     : Prueba rápida del modelo entrenado

DATOS:
------
- train_split_limpio.csv  : Datos de entrenamiento
- val_split_limpio.csv    : Datos de validación
- test_split_limpio.csv   : Datos de prueba
- imagenes_1_porciento/   : Imágenes de radiografías


DONDE EMPEZAR
=============

1. INSTALACIÓN:
   $ pip install -r requirements.txt

2. VERIFICAR SETUP:
   $ python check_setup.py

3. LEER DOCUMENTACIÓN:
   - QUICK_START.md para comenzar rápido
   - README.md para documentación completa
   - config.py para ver todas las opciones de configuración

4. ENTRENAR:
   $ python train.py

5. EVALUAR:
   $ python quick_test.py
   $ python evaluate.py --model-dir modelo_entrenado


PERSONALIZACIÓN
===============

CAMBIAR HIPERPARÁMETROS:
- Editar config.py (clase TrainingConfig)
- O usar argumentos CLI: python train.py --epochs 10 --batch-size 8

CAMBIAR CONFIGURACIÓN DE LORA:
- Editar config.py (clase LoRAConfig)
  - r: rango de matrices (16 es bueno por defecto)
  - lora_alpha: típicamente 2*r
  - target_modules: capas donde aplicar LoRA

CAMBIAR PREPROCESAMIENTO:
- Editar config.py (clase DataConfig)
  - image_size: tamaño de salida
  - clahe_clip_limit: intensidad de mejora de contraste

CAMBIAR PARÁMETROS DE GENERACIÓN:
- Editar config.py (clase InferenceConfig)
  - max_new_tokens: longitud del reporte
  - num_beams: calidad vs velocidad
  - temperature: aleatoriedad del texto


ARQUITECTURA DEL CÓDIGO
========================

FILOSOFÍA:
- Configuración centralizada en config.py
- Separación de responsabilidades (un módulo = una función)
- Type hints en todo el código
- Documentación completa con docstrings
- Sin dependencias de Google Colab

MÓDULOS:

1. config.py
   └── Define todas las configuraciones usando dataclasses
   └── PathConfig, ModelConfig, LoRAConfig, TrainingConfig, etc.

2. preprocessing.py
   └── estandarizar_imagen(): Pipeline completo
       ├── aplicar_clahe(): Mejora de contraste
       ├── convertir_a_rgb(): Conversión de formato
       └── aplicar_padding(): Redimensionamiento

3. data_loader.py
   └── MimicCXRDataset(Dataset)
       ├── __init__(): Carga CSV y configura dataset
       ├── __getitem__(): Retorna muestra procesada
       └── _buscar_imagen(): Localiza imagen por dicom_id

4. model_utils.py
   └── Funciones de alto nivel para modelos
       ├── cargar_procesador()
       ├── cargar_modelo_base()
       ├── aplicar_lora_a_modelo()
       ├── guardar_modelo()
       └── cargar_modelo_entrenado()

5. train.py
   └── entrenar_modelo()
       ├── Carga procesador y modelo
       ├── Carga datasets
       ├── Configura Trainer de HuggingFace
       └── Entrena y guarda checkpoints

6. evaluate.py
   └── GeneradorReportes
       ├── generar_desde_imagen(): Genera reporte individual
       ├── evaluar_en_dataset(): Evaluación masiva
       └── generar_con_multiples_estrategias(): Comparación


FLUJO DE DATOS
==============

ENTRENAMIENTO:
CSV → MimicCXRDataset → Preprocessor → BLIP2 Processor → Trainer → Modelo Guardado
                ↓
         Busca imagen
                ↓
         CLAHE + Padding
                ↓
         Tokenización

INFERENCIA:
Imagen → Preprocessor → BLIP2 Processor → Modelo → Reporte Generado
           ↓
    CLAHE + Padding
           ↓
    Tokenización
           ↓
    Beam Search


DEPENDENCIAS CLAVE
==================

CORE:
- torch: Framework de deep learning
- transformers: Modelos y procesadores de HuggingFace
- peft: Implementación de LoRA
- accelerate: Aceleración de entrenamiento
- bitsandbytes: Cuantización 8-bit

PROCESAMIENTO:
- opencv-python: CLAHE y manipulación de imágenes
- pillow: Carga y transformación de imágenes
- pandas: Manejo de CSV

VISUALIZACIÓN:
- matplotlib: Gráficos y visualizaciones
- tqdm: Barras de progreso


PREGUNTAS FRECUENTES
====================

P: ¿Dónde cambio el número de épocas?
R: config.py → TrainingConfig.num_train_epochs
   O: python train.py --epochs 10

P: ¿Cómo reduzco el uso de memoria?
R: config.py → TrainingConfig.per_device_train_batch_size = 2
   Y → TrainingConfig.gradient_accumulation_steps = 8

P: ¿Dónde están los checkpoints guardados?
R: modelo_entrenado/checkpoints/

P: ¿Cómo genero reportes más largos?
R: config.py → InferenceConfig.max_new_tokens = 200

P: ¿Puedo usar CPU en lugar de GPU?
R: Sí, pero será muy lento. El código detecta automáticamente.

P: ¿Cómo sé si el modelo está aprendiendo?
R: Observa training.log, la loss debe disminuir con las épocas.


REFERENCIAS ÚTILES
==================

MODELOS:
- BLIP2: https://github.com/salesforce/LAVIS
- HuggingFace: https://huggingface.co/docs/transformers

TÉCNICAS:
- LoRA Paper: https://arxiv.org/abs/2106.09685
- PEFT: https://github.com/huggingface/peft

DATASET:
- MIMIC-CXR: https://physionet.org/content/mimic-cxr/2.0.0/


MANTENIMIENTO
=============

ACTUALIZAR DEPENDENCIAS:
$ pip install -r requirements.txt --upgrade

VERIFICAR CÓDIGO:
$ python -m pytest  # Si tienes tests
$ python check_setup.py

LIMPIAR CACHE:
$ rm -rf __pycache__
$ rm -rf .cache


CONTRIBUIR
==========

Si quieres extender el proyecto:

1. Mantén la filosofía de separación de módulos
2. Usa type hints en todas las funciones
3. Documenta con docstrings (formato Google)
4. Actualiza README.md si añades funcionalidad
5. Añade logging apropiado


CONTACTO Y SOPORTE
==================

Para problemas técnicos:
1. Ejecuta check_setup.py
2. Revisa training.log
3. Consulta README.md
4. Revisa los docstrings del código

---

Última actualización: 22 de abril de 2026
Versión del proyecto: 1.0
"""

# Este archivo es solo documentación, no es ejecutable
print(__doc__)
