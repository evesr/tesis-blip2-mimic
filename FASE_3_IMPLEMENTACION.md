# 📋 FASE 3 — RESUMEN DE IMPLEMENTACIÓN

## ✅ Archivos Creados

### Scripts Principales
1. **`trainer_utils_phase3.py`** (892 líneas)
   - `inject_qformer_lora()`: Carga adaptadores LLM+ViT de Fase 2, fusiona, congela todo, inyecta LoRA solo en Q-Former
   - `Phase3AuditCallback`: Guarda checkpoints Q-Former cuando val_loss mejora
   - `train_single_config_phase3()`: Función principal de entrenamiento por config
   - `plot_comparative_curves()`: Genera 3 PNGs comparativos (Val Loss / BLEU-4 / ROUGE-L)
   - `load_phase3_datasets()`: Carga datasets del 1% MIMIC-CXR

2. **`run_tournament_phase3.py`** (560 líneas)
   - `fase1_inferencia_phase3()`: Inferencia con 3 adaptadores (LLM + ViT + Q-Former)
   - `run_tournament()`: Bucle principal del torneo (4 configs secuenciales)
   - CLI completo con `--skip-train`, `--skip-inference`, `--configs`, `--num-samples`

3. **`run_tournament_phase3_tmux.sh`** (ejecutable)
   - Launcher con tmux para sesiones persistentes
   - Manejo de sesiones duplicadas
   - Prompts interactivos para conectarse o dejar en segundo plano

4. **`verify_phase3.py`**
   - Verificación automática de:
     - Scripts de Fase 3 (existencia y permisos)
     - Datasets del 1% (CSVs + imágenes)
     - Pesos de Fase 2 (4 adaptadores: 2 LLM + 2 ViT)
     - Dependencias Python (torch, transformers, peft, radgraph, etc.)
     - Hardware (GPU CUDA disponible)
     - Módulos del proyecto
   - Output con colores ANSI para fácil lectura

### Documentación
5. **`FASE_3_README.md`**
   - Descripción completa del objetivo
   - Diagrama ASCII de arquitectura triple
   - Tabla de las 4 configuraciones
   - Hiperparámetros detallados
   - Estructura completa de outputs
   - Comandos útiles de tmux
   - Métricas de evaluación
   - Notas importantes
   - Contexto de tesis

6. **`FASE_3_QUICKSTART.md`**
   - Guía de inicio rápido
   - Comandos de ejecución (normal + tmux)
   - Monitoreo en tiempo real
   - Tabla de configuraciones con tiempos estimados
   - Outputs clave
   - Opciones avanzadas
   - Troubleshooting
   - Interpretación de resultados
   - Pasos post-completion

---

## 🎯 Las 4 Configuraciones del Torneo

```python
TOURNAMENT_CONFIGS = [
    {
        "name":             "Config_3A1",
        "qformer_r":        16,
        "qformer_alpha":    32,
        "base_weights_key": "A",  # Config_A2 de Fase 2
    },
    {
        "name":             "Config_3A2",
        "qformer_r":        32,
        "qformer_alpha":    64,
        "base_weights_key": "A",
    },
    {
        "name":             "Config_3B1",
        "qformer_r":        16,
        "qformer_alpha":    32,
        "base_weights_key": "B",  # Config_B1 de Fase 2
    },
    {
        "name":             "Config_3B2",
        "qformer_r":        32,
        "qformer_alpha":    64,
        "base_weights_key": "B",
    },
]
```

---

## 🏗️ Arquitectura de Inyección Triple

### Estrategia Implementada en `inject_qformer_lora()`

1. **Carga modelo BLIP-2 base** (bfloat16, sin cuantización)

2. **LLM (OPT 2.7B)**:
   - Carga adaptador LoRA de Fase 2 (`PeftModel.from_pretrained`)
   - **Fusiona pesos**: `llm_peft.merge_and_unload()`
   - Resultado: LLM con pesos optimizados integrados

3. **ViT (Vision Transformer)**:
   - Carga adaptador LoRA de Fase 2 (`PeftModel.from_pretrained`)
   - **Fusiona pesos**: `vit_peft.merge_and_unload()`
   - Fuerza dtype/device consistente usando Q-Former como referencia
   - Resultado: ViT con pesos optimizados integrados

4. **Congelamiento absoluto**:
   ```python
   for param in base_model.parameters():
       param.requires_grad = False
   ```

5. **Q-Former LoRA** (¡único entrenable!):
   ```python
   qformer_lora_cfg = LoraConfig(
       r=qformer_r,
       lora_alpha=qformer_alpha,
       lora_dropout=0.05,
       bias="none",
       task_type="FEATURE_EXTRACTION",
       target_modules=["query", "value"],
   )
   base_model.qformer = get_peft_model(base_model.qformer, qformer_lora_cfg)
   ```

**Parámetros entrenables**: ~0.01% del modelo total (solo Q-Former LoRA)

---

## 🔄 Diferencias con Fase 2

| Aspecto                | Fase 2                          | Fase 3                          |
|------------------------|----------------------------------|----------------------------------|
| **Dataset**            | 10% MIMIC-CXR                   | 1% MIMIC-CXR                    |
| **Componente entrenable** | LLM + ViT (dual LoRA)        | Q-Former (LoRA único)           |
| **LLM**                | LoRA nuevo                      | Fusionado y congelado           |
| **ViT**                | LoRA nuevo                      | Fusionado y congelado           |
| **Q-Former**           | Congelado                       | LoRA NUEVO ✨                   |
| **Optimizador**        | Diferencial (2 LRs: 5e-5 y 1e-4) | Simple (1 LR: 1e-4 fijo)       |
| **Scheduler**          | Cosine decay + warmup           | Constant (sin decay)            |
| **Checkpoints**        | `llm/` + `vit/`                 | `qformer/`                      |
| **Tiempo estimado**    | ~6-8h por config                | ~2-3h por config                |

---

## 📊 Pipeline de Evaluación

### Fase 1: Inferencia (`fase1_inferencia_phase3`)

Carga **3 adaptadores**:
1. LLM LoRA (Fase 2) → `merge_and_unload()`
2. ViT LoRA (Fase 2) → `merge_and_unload()`
3. Q-Former LoRA (Fase 3) → `merge_and_unload()`

Genera predicciones para test set (1%):
- Batch size: 4
- Parámetros de generación: `num_beams=5`, `do_sample=False`, `max_new_tokens=400`
- Output: `predicciones_{config_name}.csv`

### Fase 2: Evaluación (`evaluate_final.fase2_evaluacion`)

Reutiliza pipeline completo de Fase 2:
- **RadGraph F1**: Consistencia de grafos médicos
- **CheXpert**: Precision/Recall/F1 sobre 14 categorías clínicas
- **BLEU-4**: Precisión de n-gramas
- **ROUGE-L**: Similitud de secuencias

Output:
- `resultados_granulares_{config_name}.csv` (38 columnas)
- `metricas_globales_comparacion.csv`

---

## 🎨 Callbacks y Auditoría

### `Phase3AuditCallback`

Ejecuta después de cada `eval()`:

1. **Checkpoint condicional**:
   ```python
   if val_loss < self.best_val_loss:
       qformer_save = best_model_dir / "qformer"
       model.qformer.save_pretrained(str(qformer_save))
   ```

2. **Historial CSV**:
   ```
   Step,Train_Loss,Val_Loss,BLEU-4,ROUGE-L
   125,0.2819,0.3244,0.0,0.0  # BLEU/ROUGE en 0 por skip_text_generation
   250,0.2680,0.3216,0.0,0.0
   ...
   ```

3. **Audit log TXT** (predicciones sin truncar):
   ```
   Step=125 | Text generation SKIPPED (modo fast-train)
   Step=250 | Text generation SKIPPED (modo fast-train)
   ...
   ```

4. **VRAM cleanup**:
   ```python
   gc.collect()
   torch.cuda.empty_cache()
   ```

**Optimización**: `skip_text_generation=True` ahorra ~70% del tiempo en evaluaciones intermedias.

---

## 🔧 Hiperparámetros Clave

```python
TRAINING_CONFIG = {
    "lr":                      1e-4,    # Fijo (sin scheduler dinámico)
    "max_epochs":              50,
    "early_stopping_patience": 10,      # Evals consecutivas sin mejora
    "batch_size":              4,
    "gradient_accumulation":   16,      # Batch efectivo = 64
    "eval_steps":              125,
    "warmup_steps":            50,
    "audit_samples":           10,
    "max_new_tokens":          512,
    "lora_dropout":            0.05,
}

QFORMER_TARGET_MODULES = ["query", "value"]
```

**Rationale de LR fijo**: 
- En torneo de exploración rápida (1% datos), scheduler dinámico puede causar inestabilidad
- LR constante permite convergencia limpia en pocas épocas

---

## 📈 Outputs Generados

```
phase3_results/
├── Config_3A1/               # Checkpoints internos por config
├── Config_3A2/
├── Config_3B1/
├── Config_3B2/
├── checkpoints/
│   ├── best_model_Config_3A1/qformer/  # Solo adaptador Q-Former
│   ├── best_model_Config_3A2/qformer/
│   ├── best_model_Config_3B1/qformer/
│   └── best_model_Config_3B2/qformer/
├── history_phase3_Config_3*.csv        # 4 archivos (1 por config)
├── audit_phase3_Config_3*.txt          # 4 archivos
├── intermediate_train_results.csv      # Progreso incremental
├── tournament_train_ranked.csv         # Ranking final
├── tournament_phase3.log               # Log completo
├── evaluation/
│   ├── predicciones_Config_3*.csv          # 4 archivos
│   ├── resultados_granulares_Config_3*.csv # 4 archivos (38 cols)
│   └── metricas_globales_comparacion.csv
└── plots/
    ├── 01_val_loss_phase3.png          # Comparativo
    ├── 02_bleu4_phase3.png             # Comparativo
    ├── 03_rougeL_phase3.png            # Comparativo
    └── 04_train_val_loss_Config_3*.png # 4 individuales
```

Total: **~22 archivos principales** (sin contar checkpoints internos de HF Trainer)

---

## ✅ Verificación Pre-ejecución

```bash
python verify_phase3.py
```

Verifica:
- ✓ Scripts de Fase 3 (3 archivos)
- ✓ Datasets del 1% (4 elementos: 3 CSVs + 1 directorio)
- ✓ Pesos de Fase 2 (4 adaptadores)
- ✓ Dependencias Python (7 librerías)
- ✓ GPU CUDA
- ✓ Módulos del proyecto (6 módulos)

**Output ejemplo**:
```
======================================================================
  VERIFICACIÓN DE PRE-REQUISITOS — FASE 3
======================================================================

1. Scripts de Fase 3:
  ✓ trainer_utils_phase3.py
  ✓ run_tournament_phase3.py
  ✓ run_tournament_phase3_tmux.sh (ejecutable)

2. Datasets del 1% (MIMIC-CXR):
  ✓ train_split_limpio.csv
    → Train: 2,659 muestras
  ✓ val_split_limpio.csv
    → Val:   735 muestras
  ✓ test_split_limpio.csv
    → Test:  397 muestras
  ✓ imagenes_1_porciento/

3. Pesos de Fase 2 (bases duales LLM+ViT):
  ✓ Base A — LLM LoRA (Config_A2)
  ✓ Base A — ViT LoRA (Config_A2)
  ✓ Base B — LLM LoRA (Config_B1)
  ✓ Base B — ViT LoRA (Config_B1)

...

✅ TODAS LAS VERIFICACIONES PASARON
```

---

## 🚀 Comandos de Ejecución

### Verificar setup
```bash
python verify_phase3.py
```

### Lanzar torneo (tmux)
```bash
bash run_tournament_phase3_tmux.sh
```

### Lanzar torneo (terminal normal)
```bash
python run_tournament_phase3.py 2>&1 | tee phase3_results/tournament_phase3.log
```

### Solo una config
```bash
python run_tournament_phase3.py --configs Config_3A1
```

### Skip entrenamiento
```bash
python run_tournament_phase3.py --skip-train
```

---

## 📚 Documentación Asociada

1. **`FASE_3_README.md`**: Documentación completa y detallada
2. **`FASE_3_QUICKSTART.md`**: Guía de inicio rápido
3. Este archivo: Resumen de implementación

---

## 🎓 Contexto en la Tesis

**Fase 1**: LoRA en LLM únicamente (ViT y Q-Former congelados)  
**Fase 2**: LoRA dual en LLM + ViT (Q-Former congelado)  
**Fase 3**: LoRA en Q-Former únicamente (LLM y ViT fusionados y congelados) ← **NUEVA**

**Objetivo**: Identificar si optimizar el puente semántico (Q-Former) mejora la calidad de los reportes sin alterar las capacidades ya aprendidas en fases previas.

**Hipótesis**: El Q-Former es el cuello de botella semántico. Optimizarlo con LoRA sobre bases ya entrenadas debería mejorar la coherencia entre visión y lenguaje sin sobreajustar.

---

## ⏱️ Tiempo Estimado de Ejecución

| Config      | Tiempo     |
|-------------|------------|
| Config_3A1  | ~2-3h      |
| Config_3A2  | ~2-3h      |
| Config_3B1  | ~2-3h      |
| Config_3B2  | ~2-3h      |
| **TOTAL**   | **~8-12h** |

**Hardware**: NVIDIA L40S (44.4 GB VRAM)  
**Dataset**: 1% MIMIC-CXR (~2,659 train samples)

---

## ✨ Características Implementadas

1. ✅ **Inyección triple de adaptadores** (LLM + ViT + Q-Former)
2. ✅ **Fusión de pesos** (`merge_and_unload()` para LLM y ViT)
3. ✅ **Congelamiento selectivo** (solo Q-Former entrenable)
4. ✅ **Optimizador simple** (AdamW con LR fijo)
5. ✅ **Callbacks de auditoría** (checkpoints + historial + audit log)
6. ✅ **Pipeline de evaluación completo** (RadGraph + CheXpert + BLEU + ROUGE)
7. ✅ **Plots comparativos** (3 PNGs multi-config + 4 PNGs individuales)
8. ✅ **Rescate automático** (reanudación desde checkpoints)
9. ✅ **Verificación pre-ejecución** (`verify_phase3.py`)
10. ✅ **Launcher tmux** (sesiones persistentes)
11. ✅ **Documentación completa** (2 README + este resumen)

---

**Implementado por**: Claude (Anthropic)  
**Solicitado por**: Evelyn Silva Rozas  
**Fecha**: Mayo 30, 2026  
**Contexto**: Tesis de Magíster en Ciencia de Datos, FCFM, Universidad de Chile
