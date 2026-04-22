# 🏥 BLIP2 + LoRA: Generación de Reportes Médicos

Sistema de generación automática de reportes radiológicos usando BLIP2 con fine-tuning eficiente mediante LoRA (Low-Rank Adaptation).

## 📋 Descripción del Proyecto

Este proyecto implementa un modelo de visión-lenguaje (BLIP2) entrenado con LoRA para generar reportes médicos a partir de radiografías de tórax del dataset MIMIC-CXR. El código ha sido refactorizado desde un notebook de Google Colab a una estructura modular profesional en Python.

### ✨ Características Principales

- ✅ **Modelo BLIP2** de Salesforce (2.7B parámetros)
- ✅ **Fine-tuning eficiente con LoRA** (~1% de parámetros entrenables)
- ✅ **Cuantización 8-bit** para reducir uso de memoria
- ✅ **Preprocesamiento avanzado** (CLAHE, normalización)
- ✅ **Código completamente tipado** con type hints
- ✅ **Documentación completa** en español
- ✅ **Sin dependencias de Google Colab**

## 🗂️ Estructura del Proyecto

```
Tesis_blip2_local/
├── config.py              # Configuración central del proyecto
├── preprocessing.py       # Preprocesamiento de imágenes (CLAHE, padding)
├── data_loader.py         # Dataset personalizado para MIMIC-CXR
├── model_utils.py         # Utilidades para manejo de modelos BLIP2
├── train.py              # Script principal de entrenamiento
├── evaluate.py           # Script de evaluación e inferencia
├── requirements.txt      # Dependencias del proyecto
│
├── train_split_limpio.csv    # Datos de entrenamiento
├── val_split_limpio.csv      # Datos de validación
├── test_split_limpio.csv     # Datos de prueba
│
├── imagenes_1_porciento/     # Imágenes de radiografías
│   └── files/
│       ├── p10/
│       ├── p11/
│       └── ...
│
└── modelo_entrenado/         # Modelo y checkpoints (se crea durante entrenamiento)
    ├── checkpoints/
    └── logs/
```

## 📦 Instalación

### Requisitos Previos

- Python 3.8 o superior
- CUDA 11.7+ (para entrenamiento en GPU)
- Al menos 16GB de RAM
- GPU con al menos 8GB VRAM (recomendado: 12GB+)

### Paso 1: Clonar/Descargar el Proyecto

```bash
cd /workspace/Tesis_blip2_local
```

### Paso 2: Crear Entorno Virtual

```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# o
venv\Scripts\activate  # En Windows
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

## 🚀 Uso

### 1. Entrenamiento

#### Entrenamiento Básico

```bash
python train.py
```

Este comando:
- Carga el modelo BLIP2 base
- Aplica LoRA para fine-tuning eficiente
- Entrena durante 15 épocas (configurable)
- Guarda checkpoints automáticamente
- Guarda el modelo final en `modelo_entrenado/`

#### Opciones Avanzadas

```bash
# Personalizar número de épocas
python train.py --epochs 10

# Cambiar batch size
python train.py --batch-size 8

# Modificar learning rate
python train.py --learning-rate 5e-5

# Usar CSV personalizados
python train.py --train-csv mi_train.csv --val-csv mi_val.csv

# Continuar desde checkpoint
python train.py --resume-from modelo_entrenado/checkpoints/checkpoint-500
```

#### Monitoreo del Entrenamiento

El entrenamiento genera logs en:
- **Consola**: Progreso en tiempo real
- **training.log**: Log completo del entrenamiento
- **modelo_entrenado/logs/**: Logs de TensorBoard (opcional)

### 2. Evaluación e Inferencia

#### Generar Reporte para una Imagen

```bash
python evaluate.py --model-dir modelo_entrenado --image imagenes_1_porciento/files/p10/p10000032/s50414267/12345.jpg
```

#### Evaluar en Dataset de Test

```bash
python evaluate.py --model-dir modelo_entrenado --test-csv test_split_limpio.csv
```

Esto genera:
- Reportes para todas las imágenes del test set
- Archivo CSV con predicciones vs ground truth
- Estadísticas de evaluación

#### Generar Reportes Aleatorios con Visualización

```bash
python evaluate.py --model-dir modelo_entrenado --random 5 --visualize
```

#### Prompt Personalizado

```bash
python evaluate.py --model-dir modelo_entrenado \
  --image radiografia.jpg \
  --prompt "Question: Describe the chest x-ray findings in detail. Answer:"
```

## ⚙️ Configuración

Toda la configuración del proyecto está centralizada en [`config.py`](config.py). Puedes modificar:

### Rutas

```python
class PathConfig:
    base_dir: Path = Path("/workspace/Tesis_blip2_local")
    images_dir: Path = base_dir / "imagenes_1_porciento"
    train_csv: Path = base_dir / "train_split_limpio.csv"
    # ... más rutas
```

### Hiperparámetros de Entrenamiento

```python
class TrainingConfig:
    per_device_train_batch_size: int = 4
    learning_rate: float = 1e-4
    num_train_epochs: int = 15
    # ... más parámetros
```

### Configuración de LoRA

```python
class LoRAConfig:
    r: int = 16              # Rango de matrices LoRA
    lora_alpha: int = 32     # Factor de escalado
    lora_dropout: float = 0.05
    target_modules: List[str] = ["q_proj", "v_proj"]
```

### Parámetros de Inferencia

```python
class InferenceConfig:
    max_new_tokens: int = 150
    num_beams: int = 5
    temperature: float = 0.7
    repetition_penalty: float = 1.5
    # ... más parámetros
```

## 📊 Datos

### Formato de los CSV

Los archivos CSV deben contener las siguientes columnas:

- `dicom_id`: Identificador único de la imagen (nombre del archivo sin extensión)
- `report_text`: Texto completo del reporte médico
- `subject_id`: ID del paciente (opcional)
- `study_id`: ID del estudio (opcional)

Ejemplo:

```csv
dicom_id,subject_id,study_id,report_text
814b8f30-06ba54a9-e8c58ff8-542b6930-da88ebd1,10007920,53949165,"CHEST, TWO VIEWS: Normal chest radiograph..."
```

### Estructura de Imágenes

Las imágenes deben estar organizadas en la carpeta `imagenes_1_porciento/` con cualquier estructura de subdirectorios. El sistema buscará recursivamente archivos `.jpg` que coincidan con el `dicom_id`.

```
imagenes_1_porciento/
└── files/
    └── p10/
        └── p10000032/
            └── s50414267/
                └── 814b8f30-06ba54a9-e8c58ff8-542b6930-da88ebd1.jpg
```

## 🔬 Pipeline de Preprocesamiento

El preprocesamiento de imágenes incluye:

1. **Carga en escala de grises**: Lectura con OpenCV
2. **CLAHE**: Mejora de contraste adaptativo
   - Clip limit: 2.0
   - Tile grid size: 8x8
3. **Conversión a RGB**: Para compatibilidad con BLIP2
4. **Padding**: Redimensionamiento a 224x224 con aspecto preservado

Ver [`preprocessing.py`](preprocessing.py) para más detalles.

## 📈 Entrenamiento con LoRA

### ¿Qué es LoRA?

LoRA (Low-Rank Adaptation) permite entrenar modelos grandes de manera eficiente:

- **Solo ~1% de parámetros entrenables** (vs 100% en fine-tuning tradicional)
- **Menor uso de memoria** (~2.7GB vs ~5.4GB)
- **Entrenamiento más rápido**
- **Misma calidad** que full fine-tuning

### Parámetros LoRA

El proyecto usa por defecto:

```python
r = 16                    # Rango de matrices (mayor = más capacidad)
lora_alpha = 32           # Factor de escalado (típicamente 2*r)
lora_dropout = 0.05       # Regularización
target_modules = ["q_proj", "v_proj"]  # Capas de atención
```

## 🎯 Arquitectura del Modelo

```
BLIP2 (2.7B parámetros)
├── Vision Encoder (ViT)          [CONGELADO]
├── Q-Former                       [CONGELADO]
└── Language Model (OPT-2.7B)     [LoRA aplicado aquí]
    ├── q_proj [LoRA ✓]
    ├── v_proj [LoRA ✓]
    └── otras capas [CONGELADO]
```

## 📝 Módulos Principales

### [`config.py`](config.py)
Configuración centralizada con dataclasses:
- `PathConfig`: Rutas del proyecto
- `ModelConfig`: Configuración del modelo base
- `LoRAConfig`: Parámetros de LoRA
- `TrainingConfig`: Hiperparámetros de entrenamiento
- `DataConfig`: Configuración de datos
- `InferenceConfig`: Parámetros de generación

### [`preprocessing.py`](preprocessing.py)
Funciones de preprocesamiento de imágenes:
- `estandarizar_imagen()`: Pipeline completo
- `aplicar_clahe()`: Mejora de contraste
- `convertir_a_rgb()`: Conversión de formato
- `aplicar_padding()`: Redimensionamiento

### [`data_loader.py`](data_loader.py)
Dataset personalizado de PyTorch:
- `MimicCXRDataset`: Clase principal del dataset
- `cargar_splits()`: Carga train/val/test
- `crear_dataloader()`: Factory de DataLoaders

### [`model_utils.py`](model_utils.py)
Utilidades para manejo de modelos:
- `cargar_procesador()`: Carga procesador BLIP2
- `cargar_modelo_base()`: Carga modelo con cuantización
- `aplicar_lora_a_modelo()`: Configura LoRA
- `guardar_modelo()`: Guarda modelo y procesador
- `cargar_modelo_entrenado()`: Carga modelo fine-tuned

### [`train.py`](train.py)
Script principal de entrenamiento:
- Orquesta todo el proceso de entrenamiento
- Maneja argumentos de línea de comandos
- Integra con Hugging Face Trainer
- Guarda checkpoints automáticamente

### [`evaluate.py`](evaluate.py)
Script de evaluación e inferencia:
- `GeneradorReportes`: Clase para generación
- Evaluación en datasets completos
- Visualización de predicciones
- Múltiples estrategias de generación

## 🛠️ Desarrollo

### Type Hints

Todo el código está completamente tipado:

```python
def estandarizar_imagen(
    ruta_imagen: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224),
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
) -> Optional[Image.Image]:
    ...
```

### Logging

El proyecto usa logging estándar de Python:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("✅ Proceso completado")
logger.warning("⚠️ Advertencia importante")
logger.error("❌ Error encontrado")
```

### Testing

```bash
# Probar dataset
python data_loader.py

# Probar modelo
python model_utils.py

# Probar preprocesamiento
python preprocessing.py imagen.jpg
```

## 📊 Resultados Esperados

### Durante el Entrenamiento

```
Epoch 1/15
  Step 50: loss=2.456, lr=0.0001
  Step 100: loss=2.123, lr=0.0001
  ...
Validation loss: 1.987

Epoch 2/15
  ...
```

### Evaluación

```
📊 ESTADÍSTICAS:
Muestras evaluadas: 100
Longitud promedio generada: 145.3
Longitud promedio real: 152.7
```

### Ejemplo de Reporte Generado

```
FINDINGS: The lungs are clear without focal consolidation,
pleural effusion, or pneumothorax. The cardiac silhouette
is normal in size. The mediastinal contours are unremarkable.

IMPRESSION: No acute cardiopulmonary process.
```

## ⚠️ Troubleshooting

### Error: CUDA Out of Memory

**Solución**: Reducir batch size o gradient accumulation steps

```python
# En config.py
config.training.per_device_train_batch_size = 2
config.training.gradient_accumulation_steps = 2
```

### Error: Imagen no encontrada

**Solución**: Verificar que el `dicom_id` en el CSV coincida con el nombre del archivo

```bash
# Verificar estructura
ls imagenes_1_porciento/files/ -R | grep "dicom_id"
```

### Modelo genera texto repetitivo

**Solución**: Ajustar `repetition_penalty` en config

```python
config.inference.repetition_penalty = 2.0  # Aumentar penalización
```

## 📚 Referencias

- **BLIP2**: [Salesforce/LAVIS](https://github.com/salesforce/LAVIS)
- **LoRA**: [Paper](https://arxiv.org/abs/2106.09685) | [PEFT Library](https://github.com/huggingface/peft)
- **MIMIC-CXR**: [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/)
- **Transformers**: [Hugging Face](https://huggingface.co/docs/transformers)

## 👨‍💻 Autor

**[Tu nombre]**
- Tesis de [Grado/Magister] en Data Science
- Especialización en Imágenes Médicas

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- Dataset MIMIC-CXR de PhysioNet
- Modelo BLIP2 de Salesforce Research
- Biblioteca PEFT de Hugging Face
- Comunidad de código abierto

---

**Última actualización**: 22 de abril de 2026

Para preguntas o problemas, por favor abre un issue en el repositorio.
