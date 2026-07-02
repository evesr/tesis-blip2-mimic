# 🚀 FASE 2.2 QUICKSTART

## ✅ CAMBIOS REALIZADOS

### 1. **Rutas actualizadas al 10%**
   - `train_split_limpio.csv` → `train_split_10pct_limpio.csv`
   - `imagenes_1_porciento/` → `imagenes_10_porciento/`

### 2. **Generación de texto desactivada durante entrenamiento**
   - `Phase2AuditCallback(..., skip_text_generation=True)`
   - **Ahorro: ~70% en tiempo de evaluación**

### 3. **Torneo filtrado: 2 finalistas (de 4)**
   - ✅ Config_A2 (LLM r=16 + ViT r=16)
   - ✅ Config_B1 (LLM r=32 + ViT r=8)

---

## ⚡ EJECUCIÓN

### Verificar setup:
```bash
python verify_fase2_2.py
```

### Entrenar ambos finalistas:
```bash
python run_tournament.py
```

### Con TMux (recomendado):
```bash
tmux new-session -d -s fase2 "python run_tournament.py 2>&1 | tee phase2_results/tournament_10pct.log"
tmux attach -t fase2
```

### Solo Config_B1:
```bash
python run_tournament.py --configs Config_B1
```

---

## 📊 OUTPUTS

```
phase2_results/
 checkpoints/
   ├── best_model_Config_A2/  (llm/ + vit/)
   └── best_model_Config_B1/  (llm/ + vit/)
 history_phase2_Config_A2.csv
 history_phase2_Config_B1.csv
 evaluation/
   ├── predicciones_Config_A2.csv
   ├── predicciones_Config_B1.csv
   ├── resultados_granulares_Config_A2.csv  (38 columnas)
   ├── resultados_granulares_Config_B1.csv
   └── metricas_globales_comparacion.csv
 plots/
   ├── 01_val_loss_phase2.png
   ├── 02_bleu4_phase2.png
   └── 03_rougeL_phase2.png
 tournament_train_ranked.csv
```

---

## ⚙️ CONFIG ACTUAL

- Batch size: 4
- Gradient accumulation: 4
- **Batch efectivo: 16**
- Max epochs: 20
- Early stopping: patience=5 evals
- Dataset: **26,336 train** / **7,478 val** / **3,797 test** (10%)

---

## 📝 NOTAS

1. **BLEU-4 y ROUGE-L en history_phase2_*.csv estarán en 0** (skip_text_generation=True)
2. Las métricas **reales** se calculan al final en `evaluation/`
3. Los checkpoints se guardan **solo** si val_loss mejora
4. **NO** modificar batch size sin monitorear VRAM

---

**Ver detalles completos**: [FASE_2.2_RESUMEN.md](FASE_2.2_RESUMEN.md)

---

**Autor**: Evelyn Silva Rozas  
**Fecha**: Mayo 28, 2026
