# 🔬 FASE 3 — OPTIMIZACIÓN SEMÁNTICA Q-FORMER

## 📋 Descripción

Fase 3 de la tesis de generación automática de reportes médicos: **entrenamiento exclusivo del Q-Former** con bases LLM y ViT congeladas (fusionadas con adaptadores de Fase 2).

### ✨ Objetivo

Optimizar el **puente semántico** entre la representación visual (ViT) y el modelo de lenguaje (LLM) mediante LoRA en el Q-Former, manteniendo las capacidades ya aprendidas en fases previas.

---

## 🏗️ Arquitectura de Inyección Triple

```
┌─────────────────────────────────────────────────────────────┐
│  BLIP-2 Fase 3: Q-Former LoRA con LLM+ViT Congelados       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ViT (Vision Transformer)                                   │
│  ├─ Carga adaptador LoRA de Fase 2                         │
│  ├─ merge_and_unload() → pesos fusionados                  │
│  └─ requires_grad = False (CONGELADO)                      │
│                                                             │
│  Q-Former (Querying Transformer)                            │
│  ├─ Inyecta LoRA NUEVO                                     │
│  ├─ target_modules = ["query", "value"]                    │
│  ├─ r ∈ {16, 32}, alpha ∈ {32, 64}                        │
│  └─ requires_grad = True (ENTRENABLE) ✅                   │
│                                                             │
│  LLM (OPT 2.7B)                                            │
│  ├─ Carga adaptador LoRA de Fase 2                         │
│  ├─ merge_and_unload() → pesos fusionados                  │
│  └─ requires_grad = False (CONGELADO)                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Configuraciones del Torneo (4 corridas)

| Config    | Base                          | Q-Former LoRA      | Descripción                                    |
|-----------|-------------------------------|--------------------|------------------------------------------------|
| **Config_3A1** | A (LLM r=16 + ViT r=16) | r=16, α=32        | Menor capacidad Q-Former sobre base equilibrada |
| **Config_3A2** | A (LLM r=16 + ViT r=16) | r=32, α=64        | Mayor capacidad Q-Former sobre base equilibrada |
| **Config_3B1** | B (LLM r=32 + ViT r=8)  | r=16, α=32        | Menor capacidad Q-Former sobre base asimétrica  |
| **Config_3B2** | B (LLM r=32 + ViT r=8)  | r=32, α=64        | Mayor capacidad Q-Former sobre base asimétrica  |

**Base A**: Adaptadores de `best_model_Config_A2` (Fase 2)  
**Base B**: Adaptadores de `best_model_Config_B1` (Fase 2)

---

## ⚙️ Hiperparámetros

```python
TRAINING_CONFIG = {
    "lr":                      1e-4,    # Learning rate FIJO (sin scheduler)
    "max_epochs":              50,
    "early_stopping_patience": 10,      # Evaluaciones consecutivas sin mejora
    "batch_size":              4,
    "gradient_accumulation":   16,      # Batch efectivo = 64
    "eval_steps":              125,
    "audit_samples":           10,
    "max_new_tokens":          512,
    "lora_dropout":            0.05,
}

QFORMER_TARGET_MODULES = ["query", "value"]
```

**Dataset**: 1% MIMIC-CXR (~2,660 train / ~736 val / ~398 test)

---

## 🚀 Ejecución

### Opción 1: Ejecución Normal

```bash
python run_tournament_phase3.py
```

### Opción 2: Con TMux (Recomendado)

```bash
bash run_tournament_phase3_tmux.sh
```

**Ventajas de tmux**:
- ✅ Sobrevive a desconexiones SSH
- ✅ Sobrevive a cierre de VS Code
- ✅ Corre en segundo plano
- ✅ Reconexión en cualquier momento

### Comandos útiles

```bash
# Ver sesiones activas
tmux ls

# Conectarse a la sesión
tmux attach -t fase3

# Desconectarse SIN cerrar (desde dentro de tmux)
Ctrl+B, luego D

# Detener el torneo
tmux kill-session -t fase3

# Monitorear log en tiempo real
tail -f phase3_results/tournament_phase3.log

# Ver progreso de entrenamiento
watch -n 10 'tail -20 phase3_results/history_phase3_*.csv'
```

### Opciones CLI

```bash
# Solo una configuración
python run_tournament_phase3.py --configs Config_3A1

# Solo inferencia + evaluación (entrenamiento ya completado)
python run_tournament_phase3.py --skip-train

# Reutilizar predicciones existentes
python run_tournament_phase3.py --skip-inference

# Evaluar solo N muestras del test set
python run_tournament_phase3.py --num-samples 200
```

---

## 📊 Outputs Generados

```
phase3_results/
├── Config_3A1/                           # Logs y checkpoints internos
├── Config_3A2/
├── Config_3B1/
├── Config_3B2/
├── checkpoints/
│   ├── best_model_Config_3A1/
│   │   └── qformer/                      # Adaptador Q-Former LoRA
│   ├── best_model_Config_3A2/
│   ├── best_model_Config_3B1/
│   └── best_model_Config_3B2/
├── history_phase3_Config_3A1.csv         # Step | Train_Loss | Val_Loss | BLEU-4 | ROUGE-L
├── history_phase3_Config_3A2.csv
├── history_phase3_Config_3B1.csv
├── history_phase3_Config_3B2.csv
├── audit_phase3_Config_3A1.txt           # Predicciones de auditoría (sin truncar)
├── audit_phase3_Config_3A2.txt
├── audit_phase3_Config_3B1.txt
├── audit_phase3_Config_3B2.txt
├── intermediate_train_results.csv        # Progreso incremental del torneo
├── tournament_train_ranked.csv           # Ranking final por val_loss
├── tournament_phase3.log                 # Log completo del torneo
├── evaluation/
│   ├── predicciones_Config_3A1.csv       # dicom_id | study_id | reference | generated
│   ├── predicciones_Config_3A2.csv
│   ├── predicciones_Config_3B1.csv
│   ├── predicciones_Config_3B2.csv
│   ├── resultados_granulares_Config_3A1.csv  # 38 columnas (métricas completas)
│   ├── resultados_granulares_Config_3A2.csv
│   ├── resultados_granulares_Config_3B1.csv
│   ├── resultados_granulares_Config_3B2.csv
│   └── metricas_globales_comparacion.csv     # Resumen comparativo de las 4 configs
└── plots/
    ├── 01_val_loss_phase3.png            # Curvas comparativas Val Loss
    ├── 02_bleu4_phase3.png               # Curvas comparativas BLEU-4
    ├── 03_rougeL_phase3.png              # Curvas comparativas ROUGE-L
    ├── 04_train_val_loss_Config_3A1.png # Curvas individuales por config
    ├── 04_train_val_loss_Config_3A2.png
    ├── 04_train_val_loss_Config_3B1.png
    └── 04_train_val_loss_Config_3B2.png
```

---

## 📐 Métricas de Evaluación

### NLP
- **BLEU-4**: Precisión de n-gramas (0-1, mayor es mejor)
- **ROUGE-L**: Similitud de secuencias (0-1, mayor es mejor)

### Clínicas
- **RadGraph F1**: Consistencia de grafos médicos (0-1, mayor es mejor)
- **CheXpert Precision/Recall/F1**: Etiquetas clínicas binarias (0-1, mayor es mejor)

### Comparación con Ground Truth
- 14 categorías CheXpert: `No Finding`, `Cardiomegaly`, `Edema`, `Pneumonia`, etc.
- Match clase por clase + micro-averaged F1

---

## 🔍 Callback de Auditoría

`Phase3AuditCallback` ejecuta después de cada evaluación:

1. **Checkpoint Q-Former**: Guarda `qformer/` solo si `val_loss` mejora
2. **Historial**: Appends a `history_phase3_{name}.csv`
3. **Audit Log**: Guarda predicciones completas (sin truncar) en `audit_phase3_{name}.txt`
4. **Fast-train mode**: `skip_text_generation=True` (ahorra ~70% tiempo en eval)

---

## 🧪 Verificación Pre-ejecución

```bash
# Verificar que existen los pesos de Fase 2
ls -lh phase2_results-10pct/checkpoints/best_model_Config_A2/llm/
ls -lh phase2_results-10pct/checkpoints/best_model_Config_A2/vit/
ls -lh phase2_results-10pct/checkpoints/best_model_Config_B1/llm/
ls -lh phase2_results-10pct/checkpoints/best_model_Config_B1/vit/

# Verificar datasets del 1%
ls -lh train_split_limpio.csv
ls -lh val_split_limpio.csv
ls -lh test_split_limpio.csv
ls -d imagenes_1_porciento/

# Verificar dependencias
python -c "from peft import PeftModel; from radgraph import F1RadGraph; print('OK')"
```

---

## 📝 Notas Importantes

1. **BLEU-4 y ROUGE-L en 0 durante entrenamiento**: `skip_text_generation=True` desactiva generación de texto en evaluaciones intermedias para acelerar el proceso. Las métricas **reales** se calculan al final en el test set.

2. **Checkpoints duales de Fase 2**: Los adaptadores LLM y ViT se cargan desde `phase2_results-10pct/checkpoints/` y se fusionan (`merge_and_unload()`) antes de congelar.

3. **Batch efectivo 64**: `batch_size=4 × gradient_accumulation=16`. No modificar sin monitorear VRAM.

4. **Learning rate fijo**: A diferencia de Fase 2, aquí usamos `lr_scheduler_type="constant"` (sin decaimiento coseno) para simplificar el torneo de exploración.

5. **Device/dtype consistency**: Se usa `qformer` como referencia para garantizar que todos los componentes fusionados estén en el mismo device/dtype.

---

## 🎓 Contexto de Tesis

**Fase 1**: LoRA en LLM únicamente (ViT y Q-Former congelados)  
**Fase 2**: LoRA dual en LLM + ViT (Q-Former congelado)  
**Fase 3**: LoRA en Q-Former únicamente (LLM y ViT fusionados y congelados) ← **ESTÁS AQUÍ**

**Objetivo final**: Identificar la estrategia óptima de fine-tuning modular para generación de reportes médicos con BLIP-2 en MIMIC-CXR.

---

## 👤 Autor

**Evelyn Silva Rozas**  
Magíster en Ciencia de Datos — FCFM, Universidad de Chile  
Mayo 2026

---

## 📚 Referencias

- **BLIP-2**: Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models", ICML 2023
- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- **MIMIC-CXR**: Johnson et al., "MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs", Nature Scientific Data 2019
- **RadGraph**: Jain et al., "RadGraph: Extracting Clinical Entities and Relations from Radiology Reports", NeurIPS 2021
- **CheXbert**: Smit et al., "CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT", EMNLP 2020
