# 🔄 DIAGRAMA DE FLUJO COMPLETO - PROYECTO TESIS BLIP2

## 📊 FLUJO GENERAL DEL PROYECTO

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FASE 1: PREPARACIÓN DE DATOS                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────┐
                    │   MIMIC-CXR Dataset Completo     │
                    │   (377,110 radiografías)         │
                    └──────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────┐
                    │   Extracción 1% (~6,400 imgs)    │
                    │   dataset_1_porciento_final.csv  │
                    └──────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────┐
                    │   División Train/Val/Test        │
                    │   (70% / 20% / 10%)              │
                    └──────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┬─────────────────┐
                    ▼                 ▼                 ▼                 
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │ train_split     │ │ val_split       │ │ test_split      │
          │ 2,660 imgs      │ │ 736 imgs        │ │ 398 imgs        │
          └─────────────────┘ └─────────────────┘ └─────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    FASE 2: EXPERIMENTACIÓN PRELIMINAR                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          ┌──────────────────────┐          ┌──────────────────────┐
          │  Zero-Shot           │          │  A/B Test Prompts    │
          │  Evaluation          │          │  (10 imágenes)       │
          │  (Baseline)          │          │                      │
          └──────────────────────┘          └──────────────────────┘
                    │                                   │
                    ▼                                   ▼
          Métricas sin fine-tune          Prompt A vs Prompt B
          BLEU, ROUGE, CheXpert                    │
                                                   ▼
                                          ✅ Prompt B Ganador
                                          "Describe clinical findings..."


┌─────────────────────────────────────────────────────────────────────────────┐
│                 FASE 3: TORNEO DE HIPERPARÁMETROS (GRID SEARCH)             │
│                              ✅ COMPLETADO                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────┐
                    │   Espacio de Búsqueda            │
                    │   • r: [8, 16, 32]               │
                    │   • alpha: [r, 2r]               │
                    │   • lr: [1e-4]                   │
                    │   • epochs: 4                    │
                    │   ────────────────               │
                    │   Total: 6 combinaciones         │
                    └──────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │   Entrenar 6 modelos en paralelo  │
                    │   (cada uno 4 épocas)              │
                    └────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────────────────────┐
                    │                                                   │
          ┌─────────▼──────────┐                        ┌──────────────▼────────┐
          │ Por cada época:    │                        │ Por cada modelo:      │
          │ • Train loss       │                        │ • 4 checkpoints       │
          │ • Val loss         │                        │ • Métricas guardadas  │
          │ • BLEU-4 (50 imgs) │                        │ • Resultados CSV      │
          │ • ROUGE-L (50 imgs)│                        │                       │
          └────────────────────┘                        └───────────────────────┘
                    │
                    ▼
          ┌──────────────────────────────────────────────────────────┐
          │          RESULTADOS FINALES (Abril 25, 2026)             │
          ├──────────────────────────────────────────────────────────┤
          │ 🥇 r=32, alpha=64  → Val Loss: 0.4213 (MEJOR)            │
          │ 🥈 r=32, alpha=32  → ROUGE-L: 0.2257 (MEJOR)             │
          │ 🥉 r=16, alpha=32  → BLEU-4: 0.0463 (MEJOR)              │
          └──────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    FASE 4: ENTRENAMIENTO FINAL (PENDIENTE)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │   Seleccionar Top 3 del torneo    │
                    └───────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┬─────────────────┐
                    ▼                 ▼                 ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │ Modelo 1        │ │ Modelo 2        │ │ Modelo 3        │
          │ r=32, α=64      │ │ r=32, α=32      │ │ r=16, α=32      │
          │ 15 ÉPOCAS       │ │ 15 ÉPOCAS       │ │ 15 ÉPOCAS       │
          └─────────────────┘ └─────────────────┘ └─────────────────┘
                    │                 │                 │
                    └─────────────────┴─────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────┐
                    │   3 modelos completamente        │
                    │   entrenados guardados           │
                    └──────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    FASE 5: EVALUACIÓN EXHAUSTIVA (PENDIENTE)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │   Test Set (398 imágenes)         │
                    │   Nunca vistas durante training   │
                    └───────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────────────────┐
                    │   Generar reportes para todas las imágenes   │
                    └───────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │   Calcular Métricas Completas     │
                    │   • BLEU-1, 2, 3, 4               │
                    │   • ROUGE-1, 2, L                 │
                    │   • CheXpert accuracy             │
                    │   • RadGraph F1                   │
                    └───────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │   Análisis Cualitativo            │
                    │   • Ejemplos buenos/malos         │
                    │   • Errores comunes               │
                    │   • Comparación con humanos       │
                    └───────────────────────────────────┘
                                      │
                                      ▼
                    ┌──────────────────────────────────┐
                    │   Seleccionar MEJOR modelo       │
                    │   para despliegue                │
                    └──────────────────────────────────┘
```

---

## 🔄 FLUJO DETALLADO DE ENTRENAMIENTO

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENTRENAMIENTO DE UN MODELO COMPLETO                      │
└─────────────────────────────────────────────────────────────────────────────┘

INICIO: python train.py --r 32 --alpha 64 --epochs 15
   │
   ▼
┌────────────────────────────────────────┐
│ 1. CARGA DE CONFIGURACIÓN              │
│    └─ config.py (PathConfig, etc.)     │
└────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────┐
│ 2. CARGA DE DATOS                      │
│    ├─ train_split_limpio.csv (2,660)   │
│    └─ val_split_limpio.csv (736)       │
└────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────┐
│ 3. CREAR DATASET PYTORCH               │
│    └─ MimicCXRDataset                  │
│       ├─ Buscar imágenes por dicom_id  │
│       └─ Preprocesar automáticamente   │
└────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────────┐
│ 4. PREPROCESAMIENTO DE CADA IMAGEN (automático)           │
│    ┌──────────────────────────────────────────┐           │
│    │ a) Cargar en escala de grises            │           │
│    │    • cv2.imread(IMREAD_GRAYSCALE)        │           │
│    │    • Array (H, W), valores [0-255]       │           │
│    └──────────────────────────────────────────┘           │
│                     ▼                                      │
│    ┌──────────────────────────────────────────┐           │
│    │ b) Aplicar CLAHE                         │           │
│    │    • clip_limit = 2.0                    │           │
│    │    • tile_grid = (8, 8)                  │           │
│    │    • Mejora contraste local              │           │
│    │    • ¡CRUCIAL para radiografías!         │           │
│    └──────────────────────────────────────────┘           │
│                     ▼                                      │
│    ┌──────────────────────────────────────────┐           │
│    │ c) Convertir a RGB                       │           │
│    │    • (H, W) → (H, W, 3)                  │           │
│    │    • Requerido por BLIP2                 │           │
│    └──────────────────────────────────────────┘           │
│                     ▼                                      │
│    ┌──────────────────────────────────────────┐           │
│    │ d) Padding a 224×224                     │           │
│    │    • Mantiene aspect ratio               │           │
│    │    • Bordes negros                       │           │
│    │    • Tamaño fijo para batch              │           │
│    └──────────────────────────────────────────┘           │
│                     ▼                                      │
│    ┌──────────────────────────────────────────┐           │
│    │ e) Normalización BLIP2Processor          │           │
│    │    • Valores a rango [0, 1]              │           │
│    │    • Mean/std de ImageNet                │           │
│    └──────────────────────────────────────────┘           │
└────────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────┐
│ 5. CARGAR MODELO BASE                  │
│    └─ BLIP2 (Salesforce/blip2-opt-2.7b)│
│       • 2.7B parámetros                │
│       • Pre-entrenado en natural imgs  │
└────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────┐
│ 6. APLICAR LoRA                        │
│    ├─ r = 32 (del grid search)         │
│    ├─ alpha = 64 (del grid search)     │
│    ├─ dropout = 0.05                   │
│    ├─ target = ["q_proj", "v_proj"]    │
│    └─ ~1% params entrenables           │
└────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────────┐
│ 7. LOOP DE ENTRENAMIENTO (15 épocas)                      │
│    ┌────────────────────────────────────────┐             │
│    │ POR CADA ÉPOCA:                        │             │
│    ├────────────────────────────────────────┤             │
│    │ • Iterar sobre batches (batch_size=4)  │             │
│    │ • Forward pass                         │             │
│    │   ├─ Imagen preprocesada → BLIP2      │             │
│    │   ├─ Texto tokenizado → labels         │             │
│    │   └─ Calcular loss (cross-entropy)     │             │
│    │                                         │             │
│    │ • Backward pass                        │             │
│    │   ├─ loss.backward()                   │             │
│    │   ├─ Actualizar SOLO params LoRA      │             │
│    │   └─ Base model congelado              │             │
│    │                                         │             │
│    │ • Cada 4 batches (gradient_accum)     │             │
│    │   └─ optimizer.step()                  │             │
│    │                                         │             │
│    │ • Cada 50 steps                        │             │
│    │   └─ Log train_loss                    │             │
│    │                                         │             │
│    │ • Al final de época                    │             │
│    │   ├─ Evaluar en val set                │             │
│    │   ├─ Calcular val_loss                 │             │
│    │   ├─ Guardar checkpoint (si es mejor)  │             │
│    │   └─ torch.cuda.empty_cache()          │             │
│    └────────────────────────────────────────┘             │
└────────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────┐
│ 8. GUARDAR MODELO FINAL                │
│    ├─ modelo_entrenado/                │
│    │  ├─ adapter_config.json            │
│    │  ├─ adapter_model.safetensors      │
│    │  └─ tokenizer files                │
│    └─ checkpoints/ (mejores épocas)     │
└────────────────────────────────────────┘
   │
   ▼
FIN: Modelo listo para evaluación
```

---

## 🔄 FLUJO DE INFERENCIA (GENERAR REPORTE)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GENERAR REPORTE PARA UNA RADIOGRAFÍA                     │
└─────────────────────────────────────────────────────────────────────────────┘

INICIO: python evaluate.py --model-dir modelo_entrenado --image xray.jpg
   │
   ▼
┌────────────────────────────────────────┐
│ 1. CARGAR MODELO ENTRENADO             │
│    ├─ BLIP2 base                       │
│    ├─ Adaptadores LoRA                 │
│    └─ Procesador (tokenizer)           │
└────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────┐
│ 2. CARGAR Y PREPROCESAR IMAGEN         │
│    └─ Mismo pipeline que entrenamiento │
│       (CLAHE → RGB → Padding)          │
└────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────────┐
│ 3. PREPARAR PROMPT                                         │
│    Prompt B: "Question: Describe the clinical findings,    │
│               anatomical structures, and abnormalities     │
│               visible in this chest radiograph. Answer:"   │
└────────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────┐
│ 4. TOKENIZAR PROMPT E IMAGEN           │
│    └─ processor(image, text, ...)      │
│       ├─ pixel_values: imagen tensor   │
│       ├─ input_ids: prompt tokens      │
│       └─ attention_mask: máscara       │
└────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────────┐
│ 5. GENERAR REPORTE                                         │
│    model.generate(                                         │
│        pixel_values=...,                                   │
│        input_ids=...,                                      │
│        max_new_tokens=150,      # Longitud máxima          │
│        num_beams=5,             # Beam search              │
│        temperature=0.7,         # Creatividad              │
│        repetition_penalty=1.5,  # Evitar repetición        │
│        early_stopping=True      # Detener al <EOS>         │
│    )                                                        │
└────────────────────────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────┐
│ 6. DECODIFICAR TOKENS → TEXTO          │
│    └─ processor.decode(outputs)        │
└────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────┐
│ 7. LIMPIAR OUTPUT                      │
│    └─ Remover prompt del texto         │
└────────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────────────────────────────┐
│ OUTPUT: REPORTE MÉDICO GENERADO                            │
│ Ejemplo:                                                   │
│ "The heart size is within normal limits. The lungs are    │
│  clear without focal consolidation, pleural effusion, or  │
│  pneumothorax. The mediastinal and hilar contours are     │
│  unremarkable. No acute osseous abnormalities."           │
└────────────────────────────────────────────────────────────┘
```

---

## 📊 FLUJO DE DATOS EN EL PROYECTO

```
CSV Files                Dataset PyTorch           Modelo               Output
─────────                ─────────────────         ──────               ──────

train_split_limpio.csv ─┐
                        │
                        ├─→ MimicCXRDataset ──┐
                        │   ├─ Busca imágenes │
val_split_limpio.csv ───┤   ├─ Preprocesa     ├──→ BLIP2 + LoRA ──→ Reportes
                        │   └─ Tokeniza       │   (entrenamiento)    generados
                        │                     │
test_split_limpio.csv ──┘                     │
                                              │
                                              └──→ Evaluación
                                                   (métricas)


Directorio Imágenes       Preprocesamiento        Tensores PyTorch
───────────────────       ────────────────        ────────────────

imagenes_1_porciento/ ──→ CLAHE ──→ RGB ──→ Padding ──→ Normalización ──→ Batch
  ├─ p10/                  ↓         ↓        ↓           ↓                Tensors
  ├─ p11/              Contraste  3 canales  224×224   [0,1] range        (4, 3, 224, 224)
  └─ .../              mejorado                                            + labels
```

---

## 🎯 DEPENDENCIAS ENTRE COMPONENTES

```
                                config.py
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            preprocessing.py  model_utils.py   paths/params
                    │               │               
                    └───────┬───────┘               
                            │                       
                            ▼                       
                    data_loader.py                  
                    (MimicCXRDataset)               
                            │                       
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
        train.py      evaluate.py    grid_search.py
            │               │               │
            └───────────────┴───────────────┘
                            │
                            ▼
                    Modelos entrenados
                    + Resultados
```

**Orden de dependencias** (de más básico a más complejo):
1. `config.py` - Sin dependencias internas
2. `preprocessing.py` - Depende solo de config
3. `model_utils.py` - Depende solo de config
4. `data_loader.py` - Depende de config + preprocessing
5. `train.py`, `evaluate.py`, `grid_search.py` - Dependen de todo lo anterior

---

## 💾 FLUJO DE GUARDADO DE RESULTADOS

```
Grid Search Results                     Training Outputs
───────────────────                     ────────────────

grid_search_results/                    modelo_entrenado/
├── progress.txt                        ├── adapter_config.json
│   (actualizado cada época)            ├── adapter_model.safetensors
│                                       ├── tokenizer files
├── intermediate_results.csv            │
│   (checkpoints automáticos)           └── checkpoints/
│                                           ├── checkpoint-epoch-1/
├── final_results_ranked.csv                ├── checkpoint-epoch-2/
│   (ranking final)                         └── ... (best epochs)
│
├── all_epochs_results.csv
│   (todas las métricas)
│
└── plots/
    ├── r8_alpha8_metrics.png
    ├── r32_alpha64_metrics.png
    └── comparison_all_models.png
```

---

Este diagrama de flujo completo te ayudará a visualizar cómo interactúan todos los componentes del proyecto desde la preparación de datos hasta la evaluación final. 🎯
