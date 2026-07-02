# Fase 3.2 — Escalamiento Final al 10%

## 📋 Resumen de la Refactorización

Se han creado exitosamente **3 archivos nuevos** para la Fase 3.2, adaptando la infraestructura del 1% al dataset del 10% con las configuraciones ganadoras y scheduler dinámico.

---

## 📁 Archivos Creados

### 1. `trainer_utils_phase3_10pct.py` (816 líneas)

**Utilidades de entrenamiento para el 10%**

#### Cambios principales respecto a `trainer_utils_phase3.py`:

- ✅ **Datasets actualizados al 10%**:
  - `TRAIN_CSV_10PCT = "train_split_10pct_limpio.csv"` (26,335 muestras)
  - `VAL_CSV_10PCT = "val_split_10pct_limpio.csv"` (7,477 muestras)
  - `TEST_CSV_10PCT = "test_split_10pct_limpio.csv"` (3,796 muestras)
  - `IMAGES_DIR_10PCT = "imagenes_10_porciento"` (37,608 imágenes)

- ✅ **Directorio de salida**:
  - `OUTPUT_DIR = "phase3_results-10pct"`

- ✅ **Scheduler dinámico cosine**:
  - `lr_scheduler_type="cosine"` (en TrainingArguments)
  - `warmup_ratio=0.1` (reemplaza warmup_steps absolutos)
  - `logging_steps=TRAINING_CONFIG["eval_steps"]` (sincronizado para evitar escaleras)

- ✅ **Hiperparámetros finales**:
  - `lr: 1e-4`
  - `eval_steps: 125`
  - `warmup_ratio: 0.1`
  - `batch_size: 4`, `gradient_accumulation: 16` (efectivo = 64)

- ✅ **Código protegido (mantenido intacto)**:
  - Monkey patch global de `Blip2QFormerModel.forward` con whitelist
  - Lógica de `inject_qformer_lora()` con `merge_and_unload()` para LLM y ViT
  - Congelamiento completo del modelo base (`requires_grad=False`)

### 2. `run_phase3_10pct.py` (685 líneas)

**Script ejecutable del torneo final**

#### Cambios principales respecto a `run_tournament_phase3.py`:

- ✅ **2 configuraciones finalistas** (reducido de 4):
  - `Config_Final_A`: Base A + Q-Former LoRA (r=32, α=64)
  - `Config_Final_B`: Base B + Q-Former LoRA (r=32, α=64)

- ✅ **Imports actualizados**:
  - Importa desde `trainer_utils_phase3_10pct` (no `trainer_utils_phase3`)

- ✅ **Datasets del 10%**:
  - `TEST_CSV_10PCT`, `IMAGES_DIR_10PCT` automáticamente referenciados

- ✅ **Infraestructura completa**:
  - `fase1_inferencia_phase3()`: Carga 3 adaptadores (LLM + ViT + Q-Former)
  - `run_tournament()`: Pipeline completo (train → inference → evaluation)
  - CLI con flags: `--skip-train`, `--skip-inference`, `--configs`, `--num-samples`

### 3. `verify_phase3_10pct.py` (script de verificación)

**Validación pre-ejecución**

Verifica:
1. Scripts creados correctamente
2. Datasets del 10% disponibles
3. Pesos de Fase 2 (LLM + ViT duales)
4. 2 configuraciones finalistas
5. Hiperparámetros actualizados (cosine, warmup_ratio, logging_steps)
6. Monkey patch activo
7. Dependencias Python
8. GPU CUDA disponible
9. Módulos del proyecto importables

---

## ✅ Verificación Ejecutada

```bash
$ python verify_phase3_10pct.py
```

**Resultado**: ✅ TODAS LAS VERIFICACIONES PASARON

### Resumen de verificación:
- **Scripts**: ✓ 2 archivos presentes
- **Datasets**: ✓ 26,335 train / 7,477 val / 3,796 test / 37,608 imágenes
- **Pesos Fase 2**: ✓ 4 adaptadores (2 LLM + 2 ViT)
- **Configs**: ✓ 2 finalistas (Config_Final_A, Config_Final_B)
- **Hiperparámetros**: ✓ lr=1e-4, cosine scheduler, warmup_ratio=0.1, logging_steps sincronizado
- **Monkey Patch**: ✓ Activo con 10 parámetros en whitelist
- **GPU**: ✓ NVIDIA L40S, 44.4 GB VRAM

---

## 🎯 Configuraciones Finalistas

| Config | Base | LLM (Fase 2) | ViT (Fase 2) | Q-Former (Fase 3.2) |
|--------|------|--------------|--------------|---------------------|
| **Config_Final_A** | A | r=16, α=32 | r=16, α=32 | **r=32, α=64** |
| **Config_Final_B** | B | r=32, α=64 | r=8, α=16 | **r=32, α=64** |

Ambas configs usan **r=32** en el Q-Former (parámetro ganador del torneo 1%).

---

## 🚀 Comandos de Ejecución

### Ejecución completa (ambas configs)
```bash
python run_phase3_10pct.py
```

### Ejecutar solo una config
```bash
python run_phase3_10pct.py --configs Config_Final_A
```

### Modo rescate (skip entrenamiento, solo inferencia)
```bash
python run_phase3_10pct.py --skip-train
```

### Reutilizar predicciones existentes
```bash
python run_phase3_10pct.py --skip-inference
```

### Evaluar solo N muestras del test set
```bash
python run_phase3_10pct.py --num-samples 500
```

---

## 📊 Outputs Esperados

```
phase3_results-10pct/
├── 📁 checkpoints/
│   ├── best_model_Config_Final_A/
│   │   └── qformer/                         ← Adaptador Q-Former (r=32)
│   └── best_model_Config_Final_B/
│       └── qformer/                         ← Adaptador Q-Former (r=32)
├── 📊 history_phase3_Config_Final_*.csv     ← Curvas de entrenamiento
├── 📝 audit_phase3_Config_Final_*.txt       ← Predicciones de auditoría
├── 🏆 tournament_train_ranked.csv           ← Ranking final por val_loss
├── 📁 evaluation/
│   ├── predicciones_Config_Final_*.csv      ← Reportes generados (test set)
│   ├── resultados_granulares_Config_Final_*.csv  ← 38 columnas de métricas
│   └── metricas_globales_comparacion.csv    ← Resumen comparativo
└── 📊 plots/
    ├── 01_val_loss_phase3_10pct.png         ← Curvas comparativas
    ├── 02_bleu4_phase3_10pct.png
    ├── 03_rougeL_phase3_10pct.png
    └── 04_train_val_loss_Config_Final_*.png ← Curvas individuales (train vs val)
```

---

## 🔧 Detalles Técnicos

### Scheduler Cosine con Warmup

```python
# En TrainingArguments (trainer_utils_phase3_10pct.py)
lr_scheduler_type="cosine",
warmup_ratio=0.1,  # 10% de los pasos totales para warmup
logging_steps=TRAINING_CONFIG["eval_steps"],  # Sincronizado
```

**Cálculo de warmup**:
- Dataset 10%: ~26,335 muestras de entrenamiento
- Batch efectivo: 64
- Steps por época: ~411
- Total epochs (max): 50
- Total steps (max): ~20,550
- **Warmup steps**: ~2,055 (10% del total)

### Protecciones de Código Mantenidas

#### 1. Monkey Patch Global del Q-Former
```python
# Whitelist que filtra kwargs inválidos antes de llegar al Q-Former
_QFORMER_VALID_KWARGS = frozenset({
    "query_embeds", "attention_mask", "head_mask", 
    "encoder_hidden_states", "encoder_attention_mask", 
    "past_key_values", "use_cache", "output_attentions", 
    "output_hidden_states", "return_dict"
})

def _patched_qformer_forward(self, *args, **kwargs):
    filtered = {k: v for k, v in kwargs.items() if k in _QFORMER_VALID_KWARGS}
    return _original_qformer_forward(self, *args, **filtered)

Blip2QFormerModel.forward = _patched_qformer_forward
```

#### 2. Merge and Unload en `inject_qformer_lora()`
```python
# LLM: cargar y fusionar
llm_peft = PeftModel.from_pretrained(base_model.language_model, str(llm_weights_dir))
base_model.language_model = llm_peft.merge_and_unload()

# ViT: cargar y fusionar
vit_peft = PeftModel.from_pretrained(base_model.vision_model, str(vit_weights_dir))
base_model.vision_model = vit_peft.merge_and_unload().to(device=..., dtype=...)

# Congelar TODO
for param in base_model.parameters():
    param.requires_grad = False

# Inyectar LoRA SOLO en Q-Former
base_model.qformer = get_peft_model(base_model.qformer, qformer_lora_cfg)
```

---

## 📝 Notas de Implementación

### Diferencias clave con Fase 3 (1%)

| Aspecto | Fase 3 (1%) | Fase 3.2 (10%) |
|---------|-------------|----------------|
| **Dataset** | 2,659 train | 26,335 train |
| **Configs** | 4 (exploración) | 2 (finalistas) |
| **Scheduler** | constant_with_warmup | **cosine** |
| **Warmup** | warmup_steps=12 | **warmup_ratio=0.1** |
| **logging_steps** | 50 (fijo) | **125 (sincronizado con eval_steps)** |
| **eval_steps** | 12 | **125** |
| **r Q-Former** | 16, 32 (torneo) | **32 (ganador)** |
| **Tiempo estimado** | ~2-3h/config | **~10-15h/config** |

### ¿Por qué logging_steps=eval_steps?

El problema de "escaleras" en los CSVs de historial ocurre cuando:
- `logging_steps` (cada 50) registra train_loss
- `eval_steps` (cada 125) registra val_loss
- Al unirlos en el CSV, hay filas donde `val_loss=NaN` (solo train) o `train_loss=NaN` (solo eval)

**Solución**: Sincronizar ambos parámetros para que cada registro tenga ambas métricas.

---

## 🎓 Contexto de Tesis

Esta Fase 3.2 representa el **experimento final** de la tesis:

1. **Fase 1**: LoRA solo en LLM → Config_A ganadora (r=16)
2. **Fase 2**: Dual LoRA (LLM + ViT) → Config_A2 y Config_B1 (escalamiento 10%)
3. **Fase 3** (1%): Q-Former LoRA exploratorio → r=32 ganador
4. **Fase 3.2** (10%): **Escalamiento final con r=32** → modelo de producción

Los resultados de esta fase determinarán la configuración óptima para el sistema de generación automática de reportes radiológicos.

---

## ⏱️ Tiempo Estimado de Ejecución

### Por configuración:
- **Entrenamiento**: ~10-15 horas (con early stopping, puede ser menos)
- **Inferencia**: ~1.5 horas (3,796 muestras × 4 batch_size)
- **Evaluación**: ~2 horas (RadGraph, CheXbert, BLEU, ROUGE)

### Total para 2 configs: **~25-35 horas**

---

## 📚 Archivos de Referencia

- ✅ `trainer_utils_phase3_10pct.py` — Utilidades de entrenamiento (10%)
- ✅ `run_phase3_10pct.py` — Ejecutable del torneo final
- ✅ `verify_phase3_10pct.py` — Verificación de pre-requisitos
- 📖 `FASE_3.2_RESUMEN.md` — Este archivo

---

**Última actualización**: Mayo 30, 2026  
**Autor**: Evelyn Silva Rozas  
**Magíster en Ciencia de Datos** — FCFM, Universidad de Chile
