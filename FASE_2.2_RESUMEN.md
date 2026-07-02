# FASE 2.2 — ESCALAMIENTO AL 10% (Config_A2 vs Config_B1)

## ✅ CAMBIOS IMPLEMENTADOS

### 1. ACTUALIZACIÓN DE RUTAS (1% → 10%)

**Archivo**: `trainer_utils.py`

```python
# ANTES (Fase 2.1 - 1%)
TRAIN_CSV_1PCT  = BASE_DIR / "train_split_limpio.csv"
VAL_CSV_1PCT    = BASE_DIR / "val_split_limpio.csv"
TEST_CSV_1PCT   = BASE_DIR / "test_split_limpio.csv"
IMAGES_DIR_1PCT = BASE_DIR / "imagenes_1_porciento"

# AHORA (Fase 2.2 - 10%)
TRAIN_CSV_10PCT  = BASE_DIR / "train_split_10pct_limpio.csv"
VAL_CSV_10PCT    = BASE_DIR / "val_split_10pct_limpio.csv"
TEST_CSV_10PCT   = BASE_DIR / "test_split_10pct_limpio.csv"
IMAGES_DIR_10PCT = BASE_DIR / "imagenes_10_porciento"
```

**Función actualizada**: `load_phase2_datasets()` ahora carga desde los archivos del 10%.

---

### 2. DESACTIVACIÓN DE GENERACIÓN DE TEXTO DURANTE ENTRENAMIENTO

**Archivo**: `trainer_utils.py`

#### Nuevo parámetro en `Phase2AuditCallback`:
```python
def __init__(
    self,
    ...
    skip_text_generation: bool = False,  # ← NUEVO FLAG
):
```

#### Lógica optimizada en `on_evaluate()`:
```python
if self.skip_text_generation:
    # Modo fast-train: solo track val_loss
    mean_bleu4  = 0.0
    mean_rougeL = 0.0
    audit_lines = [f"Step={step} | Text generation SKIPPED (modo fast-train)\n"]
else:
    # Modo completo: genera texto y calcula métricas (solo Fase 2.1)
    n = min(self.audit_samples, len(self.val_dataset))
    # ... generación normal ...
```

#### Activación automática en `train_single_config_phase2()`:
```python
audit_cb = Phase2AuditCallback(
    ...
    skip_text_generation=True,  # ← Activado para Fase 2.2
)
```

**Ganancia esperada**: ~70% de reducción en tiempo de evaluación (cada eval ya no genera reportes).

---

### 3. FILTRADO DE CONFIGURACIONES (4 → 2 finalistas)

**Archivo**: `run_tournament.py`

```python
# ANTES (Fase 2.1 - 4 configs)
TOURNAMENT_CONFIGS: List[Dict] = [
    {"name": "Config_A1", ...},  # LLM r=16 + ViT r=8
    {"name": "Config_A2", ...},  # LLM r=16 + ViT r=16
    {"name": "Config_B1", ...},  # LLM r=32 + ViT r=8
    {"name": "Config_B2", ...},  # LLM r=32 + ViT r=16
]

# AHORA (Fase 2.2 - 2 finalistas)
TOURNAMENT_CONFIGS: List[Dict] = [
    {"name": "Config_A2", "llm_r": 16, "vit_r": 16, ...},  # Finalista A
    {"name": "Config_B1", "llm_r": 32, "vit_r": 8,  ...},  # Finalista B
]
```

---

### 4. ACTUALIZACIÓN DE BANNERS Y DOCUMENTACIÓN

- **`trainer_utils.py`**: Docstring actualizado a "Fase 2.2 - Escalamiento de finalistas al 10%"
- **`run_tournament.py`**: Docstring y print banner actualizados con:
  - Dataset: "10% MIMIC-CXR (10x más datos que Fase 2.1)"
  - Optimización: "skip_text_generation=True (ahorro ~70% en eval)"
  - Fecha: "Mayo 2026 (actualizado para Fase 2.2)"

---

## 📋 VERIFICACIÓN ANTES DE EJECUTAR

### Asegúrate de que existan estos archivos en `/workspace/tesis-blip2-mimic/`:

```bash
ls -lh train_split_10pct_limpio.csv
ls -lh val_split_10pct_limpio.csv
ls -lh test_split_10pct_limpio.csv
ls -d imagenes_10_porciento/
```

Si **NO** existen, ejecuta primero:
```bash
# Generar splits del 10% (si aún no lo hiciste)
python muestreo_10_porciento.py  # o el script que uses para crear el 10%
```

---

## 
### Entrenamiento completo de ambos finalistas:
```bash
python run_tournament.py
```

### Entrenamiento con TMux (recomendado, para persistencia):
```bash
tmux new-session -d -s fase2_10pct "python run_tournament.py 2>&1 | tee phase2_results/tournament_10pct.log"
tmux attach -t fase2_10pct
```

### Solo una config (si Config_A2 ya terminó):
```bash
python run_tournament.py --configs Config_B1
```

### Solo inferencia + evaluación (checkpoints ya listos):
```bash
python run_tournament.py --skip-train
```

---

## 📊 OUTPUTS ESPERADOS

Al finalizar, encontrarás en `phase2_results/`:

### Checkpoints:
```
checkpoints/
  ├── best_model_Config_A2/
  │   ├── llm/  (adaptador LLM LoRA r=16)
  │   └── vit/  (adaptador ViT LoRA r=16)
  └── best_model_Config_B1/
      ├── llm/  (adaptador LLM LoRA r=32)
      └── vit/  (adaptador ViT LoRA r=8)
```

### Historiales de entrenamiento:
```
history_phase2_Config_A2.csv  (Step | Val_Loss | BLEU-4=0 | ROUGE-L=0)
history_phase2_Config_B1.csv  (BLEU/ROUGE en 0 por skip_text_generation=True)
```

### Logs de auditoría:
```
audit_phase2_Config_A2.txt  (mostrará "Text generation SKIPPED")
audit_phase2_Config_B1.txt
```

### Evaluación (Test Set 10%):
```
evaluation/
  ├── predicciones_Config_A2.csv
  ├── predicciones_Config_B1.csv
  ├── resultados_granulares_Config_A2.csv  (38 columnas con todas las métricas)
  ├── resultados_granulares_Config_B1.csv
  └── metricas_globales_comparacion.csv    (tabla resumen comparativo)
```

### Plots:
```
plots/
  ├── 01_val_loss_phase2.png   (curvas de loss comparativas)
  ├── 02_bleu4_phase2.png      (curvas BLEU-4, ambas en 0 por optimización)
  └── 03_rougeL_phase2.png     (curvas ROUGE-L, ambas en 0 por optimización)
```

### Ranking final:
```
tournament_train_ranked.csv  (ordenado por best_val_loss)
```

---

## ⚙️ CONFIGURACIÓN DE BATCH (ACTUAL)

Mantiene los valores estables para evitar OOM en L40S:

```python
TRAINING_CONFIG: Dict = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    # Batch efectivo = 4 × 4 = 16
    "max_epochs": 20,
    "eval_steps": 100,
    "early_stopping_patience": 5,
    ...
}
```

**NO** modificar batch/accumulation sin monitorear VRAM durante ejecución.

---

## 🔍 VALIDACIÓN POST-EJECUCIÓN

Compara métricas Fase 2.1 (1%) vs Fase 2.2 (10%):

```python
# Cargar resultados de ambas fases
df_1pct = pd.read_csv("phase2_results_fase21/evaluation/metricas_globales_comparacion.csv")
df_10pct = pd.read_csv("phase2_results/evaluation/metricas_globales_comparacion.csv")

# Comparar BLEU-4, ROUGE-L, RadGraph F1, CheXpert F1
print(df_1pct[["config", "bleu_4", "rouge_l", "radgraph_f1", "chexpert_f1"]])
print(df_10pct[["config", "bleu_4", "rouge_l", "radgraph_f1", "chexpert_f1"]])
```

**Hipótesis**: Las métricas deberían **mejorar** con 10% (más datos de entrenamiento).

---

## ⚠️ TROUBLESHOOTING

### Error: "FileNotFoundError: train_split_10pct_limpio.csv"
**Solución**: Genera primero los splits del 10% con tu script de muestreo.

### Error: "CUDA out of memory"
**Solución**: Reduce `per_device_train_batch_size` de 4 a 2 en `TRAINING_CONFIG`.

### Warning: "CSV CheXpert no encontrado"
**Solución**: Verifica que exista `mimic-cxr-2.0.0-chexpert.csv` (o `.csv.gz`) en `/workspace/tesis-blip2-mimic/`.

---

**Autor**: Evelyn Silva Rozas  
**Fecha**: Mayo 28, 2026  
**Versión**: Fase 2.2 - Escalamiento al 10%
