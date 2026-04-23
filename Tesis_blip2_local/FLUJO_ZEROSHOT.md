# 🔄 Flujo de Trabajo Zero-Shot - Pipeline Completo

## 📋 Resumen Ejecutivo

Para cada imagen de rayos X, el sistema:
1. **Busca** la imagen por `dicom_id`
2. **Preprocesa** la imagen (CLAHE + padding)
3. **Genera** reporte con BLIP2 base
4. **Calcula** 4 tipos de métricas
5. **Guarda** resultados + visualizaciones

---

## 🎯 Pipeline Detallado para UNA Imagen

```
┌─────────────────────────────────────────────────────────────────┐
│  ENTRADA: dicom_id (ej: "a1b2c3d4-e5f6g7h8")                   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  PASO 1: BÚSQUEDA DE IMAGEN                                     │
│  ───────────────────────────────────────────────────────────────│
│  Script:    zeroshot_evaluation.py                              │
│  Función:   _buscar_imagen(dicom_id)                            │
│  ───────────────────────────────────────────────────────────────│
│  Lógica:                                                         │
│    • Busca recursivamente en imagenes_1_porciento/              │
│    • Primero intenta: {dicom_id}.jpg                            │
│    • Si no encuentra, prueba: .png, .jpeg, .JPG, etc.           │
│    • Retorna: Path("/path/to/a1b2c3d4-e5f6g7h8.jpg")            │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  PASO 2: PREPROCESAMIENTO DE IMAGEN                             │
│  ───────────────────────────────────────────────────────────────│
│  Script:    preprocessing.py                                    │
│  Función:   estandarizar_imagen(ruta_imagen)                    │
│  ───────────────────────────────────────────────────────────────│
│  Sub-pasos:                                                      │
│                                                                  │
│  2.1 Cargar en Escala de Grises                                 │
│      cv2.imread(path, IMREAD_GRAYSCALE)                         │
│      → Array numpy (H, W), valores [0-255]                      │
│                                                                  │
│  2.2 Aplicar CLAHE (Mejora de Contraste)                        │
│      aplicar_clahe(img, clip_limit=2.0, tile_grid=(8,8))       │
│      → Mejora contraste local en radiografías                   │
│      → Resalta detalles en pulmones, huesos, tejidos            │
│                                                                  │
│  2.3 Convertir a RGB                                            │
│      convertir_a_rgb(img_gray)                                  │
│      → Duplica canal: (H, W) → (H, W, 3)                        │
│      → Requerido por BLIP2 (espera RGB)                         │
│                                                                  │
│  2.4 Aplicar Padding                                            │
│      aplicar_padding(img_rgb, target_size=(224,224))            │
│      → Mantiene aspect ratio                                    │
│      → Añade bordes negros                                      │
│      → Tamaño final: exactamente 224×224                        │
│                                                                  │
│  Salida: PIL.Image RGB de 224×224                               │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  PASO 3: GENERACIÓN DE REPORTE                                  │
│  ───────────────────────────────────────────────────────────────│
│  Script:    zeroshot_evaluation.py                              │
│  Función:   generate_report(image_path)                         │
│  Modelo:    BLIP2 base (Salesforce/blip2-opt-2.7b)              │
│  ───────────────────────────────────────────────────────────────│
│  Configuración:                                                  │
│                                                                  │
│  3.1 Prompt Utilizado (Prompt B del ab_test):                   │
│      "Question: Describe the clinical findings, anatomical      │
│       structures, and abnormalities visible in this chest       │
│       radiograph. Answer:"                                      │
│                                                                  │
│  3.2 Preparar Inputs                                            │
│      processor(image, text=prompt, return_tensors="pt")         │
│      → Tokeniza prompt                                          │
│      → Normaliza imagen para BLIP2                              │
│      → Crea tensores PyTorch                                    │
│                                                                  │
│  3.3 Generar con Modelo                                         │
│      model.generate(                                            │
│          **inputs,                                              │
│          max_new_tokens=150,                                    │
│          num_beams=5,           # Beam search                   │
│          temperature=0.7,       # Creatividad moderada          │
│          repetition_penalty=1.5 # Evita repeticiones            │
│      )                                                           │
│                                                                  │
│  3.4 Decodificar y Limpiar                                      │
│      report = processor.decode(outputs[0])                      │
│      report = report.replace(prompt, "").strip()                │
│                                                                  │
│  Salida: Texto del reporte generado                             │
│  Ejemplo: "The heart size is normal. The lungs are clear.       │
│            No pleural effusion or pneumothorax."                │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  PASO 4: EXTRACCIÓN DE LABELS CHEXPERT                          │
│  ───────────────────────────────────────────────────────────────│
│  Script:    zeroshot_evaluation.py                              │
│  Función:   evaluate_test_set() - inline                        │
│  ───────────────────────────────────────────────────────────────│
│  Proceso:                                                        │
│    • Lee 14 columnas del CSV de test:                           │
│      - No Finding, Cardiomegaly, Edema, etc.                    │
│    • Convierte valores:                                         │
│      - 1.0  → Positivo (presente)                               │
│      - 0.0  → Negativo (ausente)                                │
│      - -1.0 → Incierto → 0.5 (crédito parcial)                  │
│                                                                  │
│  Salida: Dict {patología: valor}                                │
│  Ejemplo: {'Cardiomegaly': 1.0, 'Edema': 0.0,                   │
│            'Pneumonia': 0.5, ...}                               │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  PASO 5: CÁLCULO DE MÉTRICAS                                    │
│  ───────────────────────────────────────────────────────────────│
│  Script:    zeroshot_evaluation.py                              │
│  Clase:     MedicalMetrics                                      │
│  Función:   calculate_all(reference, generated, chexpert_labels)│
│  ───────────────────────────────────────────────────────────────│
│  5.1 BLEU (calculate_bleu)                                      │
│      • Tokeniza: reference.split() y generated.split()          │
│      • Calcula precisión de n-gramas (n=1,2,3,4)                │
│      • Usa smoothing para evitar 0s                             │
│      Librería: nltk.translate.bleu_score                        │
│      Retorna: {bleu_1: 0.45, bleu_2: 0.32, ...}                 │
│                                                                  │
│  5.2 ROUGE (calculate_rouge)                                    │
│      • Calcula recall de n-gramas                               │
│      • ROUGE-1, ROUGE-2, ROUGE-L                                │
│      • Retorna precision, recall, fmeasure para cada uno        │
│      Librería: rouge_score.rouge_scorer                         │
│      Retorna: {rouge1_fmeasure: 0.52, rouge2_fmeasure: 0.38, ...}│
│                                                                  │
│  5.3 CheXpert REAL (calculate_chexpert_real)                    │
│      a) Detectar patologías en texto generado:                  │
│         detect_chexpert_from_text(generated)                    │
│         • Busca keywords médicos (14 patologías)                │
│         • "pneumonia", "effusion", "cardiomegaly", etc.         │
│         Retorna: {patología: 1.0 si detectada, 0.0 si no}       │
│                                                                  │
│      b) Comparar con labels reales:                             │
│         • Manejo de inciertos (0.5):                            │
│           - Da crédito parcial                                  │
│           - No penaliza tanto como error completo               │
│         • Calcula: accuracy, precision, recall, F1              │
│         Retorna: {chexpert_accuracy: 0.71,                      │
│                   chexpert_precision: 0.68, ...}                │
│                                                                  │
│  5.4 RadGraph REAL (calculate_radgraph_real)                    │
│      a) Si RadGraph disponible (modelo DyGIE++):                │
│         • Extrae grafos de conocimiento:                        │
│           - Entidades: anatomía, observaciones                  │
│           - Relaciones: (entidad1, tipo, entidad2)              │
│         • Compara grafo_ref vs grafo_gen                        │
│         • Calcula F1 de entidades + F1 de relaciones            │
│         Librería: radgraph (pip package oficial)                │
│         Retorna: {radgraph_simple: 0.65,                        │
│                   radgraph_partial: 0.52,                       │
│                   radgraph_complete: 0.39,                      │
│                   radgraph_score: 0.52}                         │
│                                                                  │
│      b) Si RadGraph NO disponible (fallback):                   │
│         _radgraph_proxy_fallback(reference, generated)          │
│         • Extrae anatomía: lung, heart, pleura, etc.            │
│         • Extrae observaciones: opacity, edema, etc.            │
│         • Calcula F1 simple basado en términos                  │
│         Retorna: mismo formato pero con aproximación            │
│                                                                  │
│  Salida: Dict con ~18 métricas combinadas                       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  PASO 6: ALMACENAMIENTO DE RESULTADOS                           │
│  ───────────────────────────────────────────────────────────────│
│  Script:    zeroshot_evaluation.py                              │
│  Función:   evaluate_test_set() - return results_df             │
│  ───────────────────────────────────────────────────────────────│
│  Estructura del resultado por imagen:                           │
│  {                                                               │
│    'dicom_id': 'a1b2c3d4-e5f6g7h8',                             │
│    'reference': 'Texto del reporte real',                       │
│    'generated': 'Texto del reporte generado',                   │
│    'img_path': '/path/to/image.jpg',                            │
│    'bleu_1': 0.45, 'bleu_2': 0.32, ...,                         │
│    'rouge1_fmeasure': 0.52, ...,                                │
│    'chexpert_accuracy': 0.71, ...,                              │
│    'radgraph_score': 0.52, ...                                  │
│  }                                                               │
│                                                                  │
│  Se repite para TODAS las imágenes del test set (~397)          │
│  Almacena en lista → convierte a DataFrame                      │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  SALIDA: DataFrame con resultados completos                     │
│          + Archivos guardados en resultados_zeroshot/           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Después del Loop: Post-Procesamiento

Una vez procesadas TODAS las imágenes:

```
┌─────────────────────────────────────────────────────────────────┐
│  POST-PROCESAMIENTO                                             │
│  ───────────────────────────────────────────────────────────────│
│  Script:    zeroshot_evaluation.py - main()                     │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  1. GUARDAR CSV COMPLETO                                        │
│     resultados_zeroshot/resultados_completos.csv                │
│     • Todas las filas (una por imagen)                          │
│     • Todas las columnas (métricas + textos)                    │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. CALCULAR ESTADÍSTICAS                                       │
│     • Promedios de cada métrica                                 │
│     • Desviaciones estándar                                     │
│     • Min, max, mediana                                         │
│     Guardar: resultados_zeroshot/estadisticas.csv               │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. VISUALIZACIONES                                             │
│     Función: plot_metrics_summary(results_df)                   │
│     • Gráficos de barras por métrica                            │
│     • 4 subplots: BLEU, ROUGE, CheXpert, RadGraph               │
│     • Barras de error (desviación estándar)                     │
│     Guardar: resultados_zeroshot/metricas_resumen.png           │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. SELECCIONAR EJEMPLOS REPRESENTATIVOS                        │
│     Función: select_representative_examples(results_df, n=10)   │
│     • 1 mejor BLEU                                              │
│     • 1 peor BLEU                                               │
│     • 1 mejor RadGraph                                          │
│     • 1 peor RadGraph                                           │
│     • 3 aleatorios con scores medios                            │
│     • 3 aleatorios adicionales                                  │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. CREAR VISUALIZACIONES DE EJEMPLOS                           │
│     Función: create_example_visualization(row)                  │
│     Para cada uno de los 10 ejemplos:                           │
│     • Cargar imagen original                                    │
│     • Mostrar métricas en panel lateral                         │
│     • Mostrar reporte de referencia                             │
│     • Mostrar reporte generado                                  │
│     • Guardar: resultados_zeroshot/ejemplos/ejemplo_01.png      │
│     Layout: Imagen | Métricas | Textos                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Estructura de Archivos Generados

```
resultados_zeroshot/
├── resultados_completos.csv       # 397 filas × ~22 columnas
├── estadisticas.csv                # Resumen estadístico
├── metricas_resumen.png            # Gráficos de barras
└── ejemplos/
    ├── ejemplo_01_best_bleu.png
    ├── ejemplo_02_worst_bleu.png
    ├── ejemplo_03_best_radgraph.png
    ├── ejemplo_04_worst_radgraph.png
    ├── ejemplo_05_medium.png
    ├── ejemplo_06_medium.png
    ├── ejemplo_07_medium.png
    ├── ejemplo_08_random.png
    ├── ejemplo_09_random.png
    └── ejemplo_10_random.png
```

---

## 🔧 Dependencias Entre Scripts

### zeroshot_evaluation.py **DEPENDE DE**:

1. **preprocessing.py**
   - `estandarizar_imagen()` → Pipeline completo de preprocesamiento
   - `aplicar_clahe()` → Mejora de contraste
   - `convertir_a_rgb()` → Conversión de canal
   - `aplicar_padding()` → Padding a 224×224

2. **config.py**
   - `config.paths.images_dir` → Directorio de imágenes
   - `config.paths.test_csv` → CSV de test
   - `config.data.image_size` → Tamaño objetivo (224, 224)
   - `config.data.clahe_clip_limit` → Parámetro CLAHE (2.0)
   - `config.model.model_name` → Nombre del modelo BLIP2

3. **data_loader.py** (indirectamente)
   - La lógica de búsqueda de imágenes está REPLICADA
   - `_buscar_imagen()` en zeroshot es copia de data_loader

### Librerías Externas Clave:

- **transformers** (HuggingFace)
  - `Blip2Processor` → Preprocesa imagen + prompt
  - `Blip2ForConditionalGeneration` → Modelo generativo

- **nltk**
  - `sentence_bleu` → Métrica BLEU
  - `SmoothingFunction` → Evita BLEU=0

- **rouge_score**
  - `RougeScorer` → Métricas ROUGE-1, 2, L

- **radgraph** (opcional)
  - `RadGraph` → Modelo DyGIE++ para grafos médicos
  - Si no está: usa fallback interno

- **cv2** (OpenCV)
  - `imread` → Cargar imágenes
  - `createCLAHE` → Algoritmo CLAHE

- **PIL** (Pillow)
  - `Image` → Manipulación de imágenes
  - `ImageOps.pad` → Padding preservando aspect ratio

---

## ⚙️ Configuración del Modelo

### BLIP2 Base (Sin LoRA):
```python
Model: Salesforce/blip2-opt-2.7b
Params: 2.7B total (0% entrenados - zero-shot)
Device: GPU si disponible, CPU fallback
Dtype: float16 (GPU) / float32 (CPU)
```

### Parámetros de Generación:
```python
max_new_tokens = 150        # Longitud máxima del reporte
num_beams = 5               # Beam search (mejor calidad)
temperature = 0.7           # Creatividad moderada
repetition_penalty = 1.5    # Evita repeticiones
```

### Prompt Utilizado (Prompt B):
```
"Question: Describe the clinical findings, anatomical structures, 
 and abnormalities visible in this chest radiograph. Answer:"
```

Este es el mismo prompt que dio mejores resultados en el `ab_test.py`.

---

## 📈 Métricas Calculadas (18 total)

### BLEU (4 métricas):
- `bleu_1` → Unigram precision
- `bleu_2` → Bigram precision
- `bleu_3` → Trigram precision
- `bleu_4` → 4-gram precision

### ROUGE (3 métricas principales):
- `rouge1_fmeasure` → Unigram F1
- `rouge2_fmeasure` → Bigram F1
- `rougeL_fmeasure` → Longest common subsequence F1

(También retorna precision y recall para cada uno → 9 métricas ROUGE en total)

### CheXpert (4 métricas):
- `chexpert_accuracy` → % clasificaciones correctas
- `chexpert_precision` → VP / (VP + FP)
- `chexpert_recall` → VP / (VP + FN)
- `chexpert_f1` → Media armónica P-R

### RadGraph (4 métricas):
- `radgraph_simple` → F1 simple matches
- `radgraph_partial` → F1 partial matches
- `radgraph_complete` → F1 complete matches
- `radgraph_score` → Score global

---

## 🎯 Ejemplo Completo: De Imagen a Resultado

### Entrada:
```
dicom_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

### Búsqueda:
```python
img_path = Path("imagenes_1_porciento/p10/p10123456/s51234567/a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg")
```

### Preprocesamiento:
```python
# 1. Cargar grayscale
img_gray = cv2.imread(img_path, IMREAD_GRAYSCALE)  # (2048, 2048)

# 2. CLAHE
img_clahe = aplicar_clahe(img_gray)  # Contraste mejorado

# 3. RGB
img_rgb = convertir_a_rgb(img_clahe)  # (2048, 2048, 3)

# 4. Padding
img_final = aplicar_padding(img_rgb, (224, 224))  # (224, 224, 3)
```

### Generación:
```python
prompt = "Question: Describe the clinical findings, anatomical structures, and abnormalities visible in this chest radiograph. Answer:"

generated_report = "The heart size is within normal limits. The lungs are clear without focal consolidation. No pleural effusion or pneumothorax. The mediastinal and hilar contours are unremarkable."
```

### Métricas:
```python
{
  'bleu_1': 0.52,
  'bleu_2': 0.38,
  'bleu_3': 0.29,
  'bleu_4': 0.22,
  'rouge1_fmeasure': 0.61,
  'rouge2_fmeasure': 0.45,
  'rougeL_fmeasure': 0.58,
  'chexpert_accuracy': 0.86,
  'chexpert_precision': 0.80,
  'chexpert_recall': 0.75,
  'chexpert_f1': 0.77,
  'radgraph_simple': 0.68,
  'radgraph_partial': 0.54,
  'radgraph_complete': 0.41,
  'radgraph_score': 0.55
}
```

### Resultado guardado:
```python
# En resultados_completos.csv fila 123:
dicom_id,reference,generated,img_path,bleu_1,bleu_2,...
a1b2c3d4-...,The lungs are...,The heart size...,/path/to/img.jpg,0.52,0.38,...

# Visualización en ejemplos/ si es seleccionado
```

---

## 🚀 Cómo Ejecutar

```bash
python zeroshot_evaluation.py
```

**Duración estimada**: 
- ~397 imágenes
- ~5-10 segundos por imagen (con GPU)
- Total: ~30-60 minutos

**Salida en pantalla**:
```
🔄 Inicializando evaluador zero-shot...
   📍 Dispositivo: cuda
   📥 Cargando procesador...
   🤖 Cargando modelo base BLIP2...
   🔬 Inicializando RadGraph (modelo DyGIE++)...
   ✅ RadGraph listo
✅ Evaluador listo

📊 EVALUACIÓN EN CONJUNTO DE PRUEBA
====================================

📖 Cargando datos de: splits/test.csv
   Total de muestras: 397

🔍 Verificando existencia de imágenes...
   Imágenes válidas: 397 / 397

🚀 Generando reportes y calculando métricas...

Evaluando: 100%|████████████| 397/397 [35:21<00:00,  5.34s/it]

✅ Evaluación completada

📊 Calculando estadísticas...
💾 Guardando resultados...
   ✅ resultados_zeroshot/resultados_completos.csv
   ✅ resultados_zeroshot/estadisticas.csv

📈 Generando visualizaciones...
   ✅ resultados_zeroshot/metricas_resumen.png

🎯 Seleccionando ejemplos representativos...
   ✅ 10 ejemplos guardados en resultados_zeroshot/ejemplos/

✨ EVALUACIÓN ZERO-SHOT COMPLETADA ✨
```

---

## 📌 Resumen de Funciones por Script

### **zeroshot_evaluation.py** (script principal):
- `MedicalMetrics.__init__()` → Inicializa calculadoras de métricas
- `MedicalMetrics.detect_chexpert_from_text()` → Detecta patologías en texto
- `MedicalMetrics.calculate_chexpert_real()` → Compara labels vs texto
- `MedicalMetrics.calculate_bleu()` → BLEU 1-4
- `MedicalMetrics.calculate_rouge()` → ROUGE 1,2,L
- `MedicalMetrics.calculate_radgraph_real()` → RadGraph con DyGIE++
- `MedicalMetrics._radgraph_proxy_fallback()` → Fallback si RadGraph falla
- `MedicalMetrics.calculate_all()` → Todas las métricas combinadas
- `ZeroShotEvaluator.__init__()` → Carga modelo BLIP2 base
- `ZeroShotEvaluator._buscar_imagen()` → Busca imagen por dicom_id
- `ZeroShotEvaluator.generate_report()` → Genera reporte con BLIP2
- `ZeroShotEvaluator.evaluate_test_set()` → Loop principal de evaluación
- `plot_metrics_summary()` → Gráficos de resumen
- `select_representative_examples()` → Selecciona 10 ejemplos
- `create_example_visualization()` → Crea PNG de ejemplo
- `main()` → Orquesta todo el flujo

### **preprocessing.py** (preprocesamiento):
- `aplicar_clahe()` → Mejora de contraste adaptativo
- `convertir_a_rgb()` → Grayscale → RGB
- `aplicar_padding()` → Padding manteniendo aspect ratio
- `estandarizar_imagen()` → **Pipeline completo** (usado por zeroshot)

### **config.py** (configuración):
- `config.paths.*` → Rutas a datos
- `config.data.*` → Parámetros de preprocesamiento
- `config.model.*` → Configuración del modelo

---

## 🎓 Para tu Tesis

Este flujo demuestra:

1. **Reproducibilidad**: Pipeline completo documentado
2. **Robustez**: Manejo de errores en cada paso
3. **Métricas Rigurosas**: BLEU, ROUGE, CheXpert REAL, RadGraph REAL
4. **Preprocesamiento Médico**: CLAHE específico para radiografías
5. **Zero-shot Baseline**: Evaluación del modelo base sin fine-tuning

Los resultados de este script son tu **baseline** para comparar contra el modelo fine-tuneado con LoRA.

---

**Fecha**: Abril 2026  
**Autora**: Evelyn Silva Rozas  
**Propósito**: Evaluación zero-shot para tesis de grado
