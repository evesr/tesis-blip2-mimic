# 🚀 FASE 3 QUICKSTART

## ✅ PRE-REQUISITOS VERIFICADOS

Ejecuta antes de empezar:
```bash
python verify_phase3.py
```

Si ves `✅ TODAS LAS VERIFICACIONES PASARON`, estás listo.

---

## ⚡ EJECUCIÓN RÁPIDA

### Opción 1: TMux (Recomendado)

```bash
bash run_tournament_phase3_tmux.sh
```

Luego:
- Opción 1: Conectarte ahora → `s` + Enter
- Opción 2: Dejarlo corriendo en segundo plano → `n` + Enter

### Opción 2: Terminal Normal

```bash
python run_tournament_phase3.py 2>&1 | tee phase3_results/tournament_phase3.log
```

---

## 📊 MONITOREO EN TIEMPO REAL

### Ver log completo en vivo
```bash
tail -f phase3_results/tournament_phase3.log
```

### Ver historiales de entrenamiento
```bash
# Última config activa
tail -20 phase3_results/history_phase3_Config_3*.csv

# Todas las configs (actualización cada 10 segundos)
watch -n 10 'tail -10 phase3_results/history_phase3_Config_3*.csv'
```

### Ver progreso del torneo
```bash
cat phase3_results/intermediate_train_results.csv
```

### Conectarse a tmux (si dejaste en segundo plano)
```bash
tmux attach -t fase3
# Desconectarse sin cerrar: Ctrl+B, luego D
```

---

## 🎯 LAS 4 CONFIGURACIONES

| Config      | Base | Q-Former r | Q-Former α | Tiempo est. |
|-------------|------|------------|------------|-------------|
| Config_3A1  | A    | 16         | 32         | ~2-3h       |
| Config_3A2  | A    | 32         | 64         | ~2-3h       |
| Config_3B1  | B    | 16         | 32         | ~2-3h       |
| Config_3B2  | B    | 32         | 64         | ~2-3h       |

**Tiempo total estimado**: ~8-12 horas (4 configs secuenciales)

**Base A**: LLM r=16 + ViT r=16 (de Config_A2 Fase 2)  
**Base B**: LLM r=32 + ViT r=8  (de Config_B1 Fase 2)

---

## 📁 OUTPUTS CLAVE

```bash
# Ranking final
cat phase3_results/tournament_train_ranked.csv

# Métricas globales
cat phase3_results/evaluation/metricas_globales_comparacion.csv

# Curvas de aprendizaje
ls phase3_results/plots/*.png
```

---

## 🔧 OPCIONES AVANZADAS

### Solo una config
```bash
python run_tournament_phase3.py --configs Config_3A1
```

### Skip entrenamiento (solo inferencia + eval)
```bash
python run_tournament_phase3.py --skip-train
```

### Skip inferencia (reutilizar predicciones)
```bash
python run_tournament_phase3.py --skip-inference
```

### Evaluar solo N muestras
```bash
python run_tournament_phase3.py --num-samples 200
```

---

## ⚠️ TROUBLESHOOTING

### Error: "Adaptador LLM/ViT no encontrado"
```bash
# Verificar que Fase 2 completó Config_A2 y Config_B1
ls -lh phase2_results-10pct/checkpoints/best_model_Config_A2/
ls -lh phase2_results-10pct/checkpoints/best_model_Config_B1/
```

### OOM (Out of Memory)
```python
# Reducir batch size en trainer_utils_phase3.py
TRAINING_CONFIG = {
    "batch_size": 2,              # Cambiar de 4 a 2
    "gradient_accumulation": 32,  # Cambiar de 16 a 32
    # Batch efectivo sigue siendo 64
}
```

### Sesión tmux colgada
```bash
# Matar y reiniciar
tmux kill-session -t fase3
bash run_tournament_phase3_tmux.sh
```

### Ver errores específicos
```bash
# En el log
grep -i "error\|exception" phase3_results/tournament_phase3.log

# En auditoría
tail -100 phase3_results/audit_phase3_Config_3*.txt
```

---

## 🎓 INTERPRETACIÓN DE RESULTADOS

### Ranking por Val Loss (menor es mejor)
```bash
cat phase3_results/tournament_train_ranked.csv | column -t -s,
```

### Métricas NLP
- **BLEU-4**: 0.04-0.06 es típico (mayor es mejor)
- **ROUGE-L**: 0.14-0.20 es razonable (mayor es mejor)

### Métricas Clínicas
- **RadGraph F1**: 0.10-0.15 es aceptable (mayor es mejor)
- **CheXpert F1**: 0.25-0.35 es común (mayor es mejor)

### ¿Cuál config ganó?
La que tenga:
1. **Menor Val Loss** (convergencia estable)
2. **Mejor RadGraph F1** (consistencia clínica)
3. **Mejor CheXpert F1** (etiquetas precisas)

---

## 📸 VISUALIZACIONES

```bash
# Ver curvas comparativas
xdg-open phase3_results/plots/01_val_loss_phase3.png
xdg-open phase3_results/plots/02_bleu4_phase3.png
xdg-open phase3_results/plots/03_rougeL_phase3.png

# O con VS Code
code phase3_results/plots/
```

---

## 🔄 AFTER COMPLETION

Una vez finalizado el torneo:

1. **Analizar ganador**
   ```bash
   head -2 phase3_results/tournament_train_ranked.csv
   ```

2. **Revisar predicciones del ganador**
   ```bash
   head -20 phase3_results/evaluation/predicciones_Config_3*.csv
   ```

3. **Comparar con Fase 2**
   ```bash
   # ¿Q-Former LoRA mejoró las métricas respecto a Fase 2?
   diff phase2_results-10pct/evaluation/metricas_globales_comparacion.csv \
        phase3_results/evaluation/metricas_globales_comparacion.csv
   ```

4. **Backup de resultados**
   ```bash
   tar -czf phase3_results_backup_$(date +%Y%m%d_%H%M%S).tar.gz phase3_results/
   ```

---

## 📚 DOCUMENTACIÓN COMPLETA

Ver [FASE_3_README.md](FASE_3_README.md) para:
- Arquitectura detallada de inyección triple
- Explicación de hiperparámetros
- Estructura completa de outputs
- Contexto de tesis

---

**Fecha**: Mayo 2026  
**Autor**: Evelyn Silva Rozas  
**Magíster en Ciencia de Datos** — FCFM, Universidad de Chile
