# 📚 FASE 3 — ÍNDICE DE NAVEGACIÓN

## 🚀 Inicio Rápido

1. **Verificar setup**: `python verify_phase3.py`
2. **Lanzar torneo**: `bash run_tournament_phase3_tmux.sh`
3. **Monitorear**: `tail -f phase3_results/tournament_phase3.log`

---

## 📖 Documentación

### Para Empezar
- **[FASE_3_QUICKSTART.md](FASE_3_QUICKSTART.md)** ← **COMIENZA AQUÍ**
  - Ejecución en 2 minutos
  - Comandos esenciales
  - Troubleshooting rápido

### Documentación Completa
- **[FASE_3_README.md](FASE_3_README.md)**
  - Arquitectura detallada
  - Hiperparámetros explicados
  - Estructura completa de outputs
  - Contexto de tesis

### Detalles Técnicos
- **[FASE_3_IMPLEMENTACION.md](FASE_3_IMPLEMENTACION.md)**
  - Archivos creados (resumen)
  - Configuraciones del torneo
  - Diferencias con Fase 2
  - Pipeline de evaluación
  - Callbacks y auditoría

---

## 🔧 Scripts Ejecutables

| Script | Descripción | Uso |
|--------|-------------|-----|
| `verify_phase3.py` | Verificación de pre-requisitos | `python verify_phase3.py` |
| `run_tournament_phase3.py` | Torneo principal (4 configs) | `python run_tournament_phase3.py` |
| `run_tournament_phase3_tmux.sh` | Launcher con tmux | `bash run_tournament_phase3_tmux.sh` |

---

## 📂 Módulos de Código

| Módulo | Descripción |
|--------|-------------|
| `trainer_utils_phase3.py` | Utilidades de entrenamiento Fase 3 |
| ├─ `inject_qformer_lora()` | Inyección triple de adaptadores |
| ├─ `Phase3AuditCallback` | Checkpoints + auditoría |
| ├─ `train_single_config_phase3()` | Loop de entrenamiento por config |
| ├─ `plot_comparative_curves()` | Generación de PNGs comparativos |
| └─ `load_phase3_datasets()` | Carga de datasets 1% MIMIC-CXR |

---

## 🎯 Las 4 Configuraciones

| Config | Base | Q-Former r | Q-Former α | LLM Fase 2 | ViT Fase 2 |
|--------|------|------------|------------|------------|------------|
| Config_3A1 | A | 16 | 32 | r=16 | r=16 |
| Config_3A2 | A | 32 | 64 | r=16 | r=16 |
| Config_3B1 | B | 16 | 32 | r=32 | r=8 |
| Config_3B2 | B | 32 | 64 | r=32 | r=8 |

**Base A** = `best_model_Config_A2` (Fase 2)  
**Base B** = `best_model_Config_B1` (Fase 2)

---

## 📊 Outputs Esperados

```
phase3_results/
├── 📁 checkpoints/
│   └── best_model_Config_3*/qformer/        ← Adaptadores Q-Former
├── 📊 history_phase3_Config_3*.csv          ← Curvas de entrenamiento
├── 📝 audit_phase3_Config_3*.txt            ← Predicciones de auditoría
├── 🏆 tournament_train_ranked.csv           ← Ranking final
├── 📁 evaluation/
│   ├── predicciones_Config_3*.csv           ← Reportes generados
│   ├── resultados_granulares_Config_3*.csv  ← 38 columnas de métricas
│   └── metricas_globales_comparacion.csv    ← Resumen comparativo
└── 📊 plots/
    ├── 01_val_loss_phase3.png               ← Curvas comparativas
    ├── 02_bleu4_phase3.png
    ├── 03_rougeL_phase3.png
    └── 04_train_val_loss_Config_3*.png      ← Curvas individuales
```

---

## 🔍 Comandos Útiles

### Durante Entrenamiento
```bash
# Ver progreso en tiempo real
tail -f phase3_results/tournament_phase3.log

# Ver historiales
tail -20 phase3_results/history_phase3_Config_3*.csv

# Ver progreso del torneo
cat phase3_results/intermediate_train_results.csv
```

### TMux
```bash
# Ver sesiones activas
tmux ls

# Conectarse
tmux attach -t fase3

# Desconectarse (desde dentro)
Ctrl+B, luego D

# Matar sesión
tmux kill-session -t fase3
```

### Post-completion
```bash
# Ver ranking final
cat phase3_results/tournament_train_ranked.csv

# Ver métricas globales
cat phase3_results/evaluation/metricas_globales_comparacion.csv

# Ver curvas
ls phase3_results/plots/*.png
```

---

## 🎓 Flujo de Trabajo Completo

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Verificación                                             │
│    python verify_phase3.py                                  │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Lanzamiento                                              │
│    bash run_tournament_phase3_tmux.sh                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Monitoreo (paralelo)                                     │
│    tail -f phase3_results/tournament_phase3.log             │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼ (8-12 horas después)
┌─────────────────────────────────────────────────────────────┐
│ 4. Análisis de Resultados                                   │
│    cat phase3_results/tournament_train_ranked.csv           │
│    cat phase3_results/evaluation/metricas_globales_*.csv    │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Visualización                                            │
│    xdg-open phase3_results/plots/*.png                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Backup                                                   │
│    tar -czf phase3_backup.tar.gz phase3_results/            │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚠️ Troubleshooting

| Problema | Solución |
|----------|----------|
| Error: "Adaptador no encontrado" | Verificar que Fase 2 completó Config_A2 y Config_B1 |
| OOM (Out of Memory) | Reducir `batch_size` de 4 a 2 en `trainer_utils_phase3.py` |
| Sesión tmux colgada | `tmux kill-session -t fase3` y reiniciar |
| BLEU/ROUGE en 0 durante train | Normal (skip_text_generation=True) |
| Checkpoint no se guarda | Val_loss no está mejorando (early stopping) |

---

## 📞 Preguntas Frecuentes

**P: ¿Cuánto tiempo toma el torneo completo?**  
R: ~8-12 horas (4 configs × 2-3h cada una)

**P: ¿Puedo ejecutar solo una config?**  
R: Sí: `python run_tournament_phase3.py --configs Config_3A1`

**P: ¿BLEU-4 y ROUGE-L están en 0 durante entrenamiento?**  
R: Sí, es normal. `skip_text_generation=True` desactiva generación en evals intermedias. Las métricas reales se calculan al final en el test set.

**P: ¿Qué config debería ganar?**  
R: Depende de las métricas:
- Val Loss más bajo → mejor convergencia
- RadGraph F1 más alto → mejor consistencia clínica
- CheXpert F1 más alto → mejores etiquetas

**P: ¿Cómo comparo con Fase 2?**  
R: Compara `metricas_globales_comparacion.csv` de ambas fases.

**P: ¿Puedo pausar y reanudar?**  
R: Sí, usa `Ctrl+B, D` en tmux. Los checkpoints permiten reanudación automática.

---

## 🎯 Next Steps (Post Fase 3)

1. **Identificar ganador** del torneo
2. **Comparar métricas** con Fase 1 y Fase 2
3. **Análisis de ablation**: ¿Q-Former LoRA mejora las métricas?
4. **Seleccionar modelo final** para producción
5. **Escribir capítulo de resultados** de tesis

---

## 📚 Archivos de Documentación (Todo en el directorio raíz)

- ✅ `FASE_3_QUICKSTART.md` ← **Inicio rápido**
- ✅ `FASE_3_README.md` ← Documentación completa
- ✅ `FASE_3_IMPLEMENTACION.md` ← Detalles técnicos
- ✅ `FASE_3_INDEX.md` ← Este archivo (navegación)

---

**Última actualización**: Mayo 30, 2026  
**Autor**: Evelyn Silva Rozas  
**Magíster en Ciencia de Datos** — FCFM, Universidad de Chile
