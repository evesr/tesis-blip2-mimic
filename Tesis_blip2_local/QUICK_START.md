# 📝 GUÍA RÁPIDA DE USO

## ⚡ Inicio Rápido (5 minutos)

### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 2. Verificar Configuración
```bash
python check_setup.py
```

### 3. Entrenar Modelo
```bash
python train.py
```

### 4. Probar Modelo
```bash
python quick_test.py
```

---

## 📚 Comandos Esenciales

### Entrenamiento

```bash
# Entrenamiento básico (15 épocas, configuración por defecto)
python train.py

# Personalizado
python train.py --epochs 10 --batch-size 8 --learning-rate 5e-5

# Continuar desde checkpoint
python train.py --resume-from modelo_entrenado/checkpoints/checkpoint-500
```

### Evaluación

```bash
# Generar reporte para una imagen
python evaluate.py --model-dir modelo_entrenado --image path/to/xray.jpg

# Evaluar dataset completo
python evaluate.py --model-dir modelo_entrenado --test-csv test_split_limpio.csv

# Probar con imágenes aleatorias
python evaluate.py --model-dir modelo_entrenado --random 5 --visualize
```

### Diagnóstico

```bash
# Verificar que todo esté configurado correctamente
python check_setup.py

# Prueba rápida del modelo entrenado
python quick_test.py --model-dir modelo_entrenado
```

---

## 🗂️ Estructura de Archivos

```
Tesis_blip2_local/
│
├── 📄 ARCHIVOS PRINCIPALES
│   ├── train.py              # Script de entrenamiento
│   ├── evaluate.py           # Script de evaluación/inferencia
│   ├── config.py             # Configuración central
│   └── requirements.txt      # Dependencias
│
├── 🔧 MÓDULOS
│   ├── data_loader.py        # Carga de datos
│   ├── preprocessing.py      # Preprocesamiento de imágenes
│   └── model_utils.py        # Utilidades de modelo
│
├── 🛠️ UTILIDADES
│   ├── check_setup.py        # Verificar configuración
│   └── quick_test.py         # Prueba rápida
│
├── 📊 DATOS
│   ├── train_split_limpio.csv
│   ├── val_split_limpio.csv
│   ├── test_split_limpio.csv
│   └── imagenes_1_porciento/
│
└── 📁 SALIDAS (se crean automáticamente)
    └── modelo_entrenado/
        ├── checkpoints/
        └── logs/
```

---

## ⚙️ Configuración Importante

### Ubicación
Todas las configuraciones están en [`config.py`](config.py)

### Parámetros Clave

```python
# Entrenamiento
num_train_epochs = 15           # Número de épocas
per_device_train_batch_size = 4 # Batch size
learning_rate = 1e-4            # Learning rate

# LoRA
r = 16                          # Rango de LoRA
lora_alpha = 32                 # Factor de escalado

# Inferencia
max_new_tokens = 150            # Tokens a generar
num_beams = 5                   # Beams para generación
temperature = 0.7               # Temperatura de sampling
```

---

## 🎯 Flujo de Trabajo Típico

### 1️⃣ Preparación
```bash
# Verificar que todo esté listo
python check_setup.py
```

### 2️⃣ Entrenamiento
```bash
# Entrenar modelo (puede tardar varias horas)
python train.py --epochs 15

# Monitorear progreso en training.log
tail -f training.log
```

### 3️⃣ Evaluación
```bash
# Prueba rápida
python quick_test.py

# Evaluación completa en test set
python evaluate.py --model-dir modelo_entrenado --test-csv test_split_limpio.csv
```

### 4️⃣ Inferencia
```bash
# Generar reportes para nuevas imágenes
python evaluate.py --model-dir modelo_entrenado --image nueva_radiografia.jpg
```

---

## 💡 Tips y Mejores Prácticas

### Optimización de Memoria

Si encuentras errores de memoria (CUDA OOM):

```python
# En config.py, reducir:
per_device_train_batch_size = 2  # en lugar de 4
gradient_accumulation_steps = 8  # en lugar de 4
```

### Monitoreo del Entrenamiento

El entrenamiento guarda:
- **Checkpoints**: `modelo_entrenado/checkpoints/`
- **Logs**: `training.log`
- **Modelo final**: `modelo_entrenado/`

### Mejores Parámetros de Generación

Para reportes más detallados:
```python
# En evaluate.py o config.py
max_new_tokens = 200           # Más largo
num_beams = 8                  # Más exhaustivo
repetition_penalty = 1.8       # Menos repetición
```

Para reportes más concisos:
```python
max_new_tokens = 100
num_beams = 3
temperature = 0.5              # Más determinístico
```

---

## 🔍 Troubleshooting Rápido

| Problema | Solución |
|----------|----------|
| CUDA Out of Memory | Reducir `batch_size` en config.py |
| Imagen no encontrada | Verificar que `dicom_id` coincida con nombre de archivo |
| Modelo genera basura | Verificar que el modelo esté entrenado correctamente |
| Import errors | Ejecutar `pip install -r requirements.txt` |
| Entrenamiento muy lento | Verificar que CUDA esté disponible (`check_setup.py`) |

---

## 📊 Métricas de Rendimiento Esperadas

### Recursos Durante Entrenamiento
- **GPU VRAM**: ~8-10 GB
- **RAM**: ~16 GB
- **Tiempo por época**: ~5-10 minutos (depende de GPU)
- **Tiempo total (15 épocas)**: ~2-3 horas

### Tamaño de Archivos
- **Modelo base descargado**: ~5 GB (en cache)
- **Adaptadores LoRA guardados**: ~100-200 MB
- **Checkpoints por época**: ~100-200 MB cada uno

---

## 🎓 Conceptos Clave

### BLIP2
Modelo de visión-lenguaje que combina un vision encoder (ViT) con un language model (OPT-2.7B) mediante un Q-Former.

### LoRA (Low-Rank Adaptation)
Técnica de fine-tuning que solo entrena ~1% de los parámetros, reduciendo memoria y tiempo mientras mantiene calidad.

### CLAHE
Contrast Limited Adaptive Histogram Equalization - mejora el contraste local en radiografías.

### Beam Search
Estrategia de generación que mantiene múltiples hipótesis simultáneas para mejor calidad de texto.

---

## 📞 Soporte

Para problemas o preguntas:

1. **Revisar README.md**: Documentación completa
2. **Ejecutar check_setup.py**: Verificar configuración
3. **Revisar training.log**: Logs detallados del entrenamiento
4. **Consultar código**: Todo está documentado con docstrings

---

**Última actualización**: 22 de abril de 2026

¡Buena suerte con tu proyecto de tesis! 🚀
