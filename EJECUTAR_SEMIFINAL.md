# 🚀 GUÍA RÁPIDA - EJECUTAR ENTRENAMIENTO SEMIFINAL

## ⚡ Inicio Inmediato (3 pasos)

### 1. Verificar que estés en el directorio correcto

```bash
cd /workspace/tesis-blip2-mimic/Tesis_blip2_local
```

### 2. Ejecutar el script

```bash
python train_semifinal.py
```

### 3. Monitorear progreso

El script mostrará en consola:
- Progreso de entrenamiento por época
- Val loss cada época
- Early stopping cuando se active
- Mejor modelo y época al finalizar

---

## ⏱️ Tiempos Estimados

| Configuración | Épocas Esperadas (con early stopping) | Tiempo Estimado |
|---------------|---------------------------------------|-----------------|
| Config 4 (r=16, α=32) | 10-25 épocas | 2-6 horas |
| Config 5 (r=16, α=16) | 10-25 épocas | 2-6 horas |
| Config 6 (r=32, α=64) | 10-25 épocas | 2-6 horas |
| **TOTAL** | - | **6-18 horas** |

> **Nota**: Early stopping reduce significativamente el tiempo vs 50 épocas completas

---

## 📊 Resultados Generados

Después de la ejecución, encontrarás:

```
semifinal_results/
├── final_results_ranked.csv      ← Ranking de las 3 configuraciones
├── intermediate_results.csv      ← Checkpoints durante ejecución
│
├── Config_4/                     ← Logs y checkpoints
│   └── logs/
├── Config_5/
│   └── logs/
├── Config_6/
│   └── logs/
│
└── plots/                        ← Gráficos comparativos
    ├── comparison_val_loss.png
    ├── comparison_bleu4.png
    ├── comparison_rougeL.png
    ├── comparison_epochs.png
    └── dashboard_complete.png
```

---

## 📈 Monitoreo Durante la Ejecución

### Opción 1: Ver consola directamente
El script muestra progreso en tiempo real.

### Opción 2: Ver resultados intermedios
```bash
# En otra terminal
cat semifinal_results/intermediate_results.csv
```

### Opción 3: Ejecutar en segundo plano (recomendado para sesiones largas)

```bash
# Con nohup
nohup python train_semifinal.py > semifinal_training.log 2>&1 &

# Ver progreso
tail -f semifinal_training.log
```

O con tmux:
```bash
# Crear sesión
tmux new -s semifinal

# Ejecutar
python train_semifinal.py

# Desconectar (Ctrl+B, luego D)
# Reconectar después
tmux attach -t semifinal
```

---

## 🔍 Interpretar Resultados

### Archivo: `final_results_ranked.csv`

Columnas importantes:
- `config_name`: Nombre de la configuración
- `r`, `alpha`: Hiperparámetros de LoRA
- `total_epochs_trained`: Épocas que entrenó (early stopping puede detener antes de 50)
- `best_epoch`: Época donde se alcanzó el mejor val_loss
- `best_val_loss`: Mejor val_loss alcanzado ⭐ (menor es mejor)
- `bleu4`: Métrica BLEU-4 (mayor es mejor)
- `rougeL`: Métrica ROUGE-L (mayor es mejor)

**El archivo ya está ordenado por `best_val_loss` (mejor primero)**

### Ejemplo de salida:

```csv
config_name,r,alpha,best_epoch,best_val_loss,bleu4,rougeL,total_epochs_trained
Config_6,32,64,15,0.3956,0.0512,0.2341,18
Config_4,16,32,12,0.4102,0.0487,0.2298,15
Config_5,16,16,10,0.4215,0.0441,0.2156,13
```

**Interpretación**: Config_6 (r=32, α=64) ganó con val_loss de 0.3956 en la época 15.

---

## 🎨 Gráficos Generados

### 1. `comparison_val_loss.png`
- Compara el mejor val_loss de cada configuración
- **Menor = Mejor**

### 2. `comparison_bleu4.png`
- Compara BLEU-4 (precisión de n-gramas)
- **Mayor = Mejor**

### 3. `comparison_rougeL.png`
- Compara ROUGE-L (similitud de secuencias)
- **Mayor = Mejor**

### 4. `comparison_epochs.png`
- Muestra cuántas épocas entrenó cada configuración
- Línea roja punteada = máximo (50 épocas)
- Si está por debajo = early stopping activado ✅

### 5. `dashboard_complete.png`
- Dashboard con las 4 métricas juntas
- Vista rápida de comparación

---

## ❓ Preguntas Frecuentes

### ¿Puedo detener el entrenamiento y continuarlo después?
No directamente. El script entrena las 3 configuraciones secuencialmente. Si lo detienes:
- Las configuraciones completadas están guardadas en `intermediate_results.csv`
- Puedes modificar `SEMIFINAL_CONFIGS` en el script para ejecutar solo las faltantes

### ¿Por qué una configuración entrenó menos épocas que otra?
**Early stopping**. Si val_loss no mejora en 3 épocas consecutivas, el entrenamiento se detiene automáticamente. Esto es **bueno** porque:
- Previene overfitting
- Ahorra tiempo de GPU
- Encuentra el punto óptimo

### ¿Qué configuración debo elegir para el modelo final?
La que tenga:
1. **Mejor val_loss** (más importante)
2. **Buenas métricas BLEU-4 y ROUGE-L**
3. **Número razonable de épocas** (no demasiado pocas = underfit)

**Recomendación**: Usa la configuración #1 del ranking (primera fila de `final_results_ranked.csv`)

### ¿Qué significa el soft metadata tagging?
Cada imagen ahora tiene contexto de vista en el prompt:
- Imagen frontal (PA/AP) → `[Context: Frontal view] Question: ...`
- Imagen lateral (LL/LATERAL) → `[Context: Lateral view] Question: ...`

Esto ayuda al modelo a generar reportes más específicos según la vista.

---

## 🛠️ Troubleshooting

### Error: "CUDA out of memory"
```python
# Editar train_semifinal.py, línea ~140
TRAINING_CONFIG = {
    'batch_size': 2,  # ← Cambiar de 4 a 2
    'gradient_accumulation': 8,  # ← Cambiar de 4 a 8
    # ...
}
```

### Error: "Image not found"
Verifica que `imagenes_1_porciento/` exista y contenga las imágenes.

### El entrenamiento se detiene muy rápido (pocas épocas)
Es normal con early stopping. Si quieres entrenar más épocas:
```python
# Editar train_semifinal.py, línea ~290
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5  # ← Cambiar de 3 a 5
)
```

### Error: "Deadlock detected"
El código ya previene esto con `cv2.setNumThreads(0)`. Si persiste, reduce `dataloader_num_workers`:
```python
# Editar train_semifinal.py, línea ~290
training_args = TrainingArguments(
    # ...
    dataloader_num_workers=2,  # ← Cambiar de 4 a 2
)
```

---

## 📞 Próximos Pasos Después del Entrenamiento

1. **Revisar `final_results_ranked.csv`** para identificar el mejor modelo
2. **Ver gráficos en `plots/`** para análisis visual
3. **Seleccionar configuración ganadora**
4. **Entrenar modelo final completo** con la mejor configuración (15-20 épocas completas)
5. **Evaluar en test set** con el modelo final

---

## 🎯 Comando Único (Copiar y Pegar)

```bash
cd /workspace/tesis-blip2-mimic/Tesis_blip2_local && python train_semifinal.py
```

---

**¡Listo para ejecutar!** 🚀

_Para más detalles técnicos, consulta `CAMBIOS_SEMIFINAL.md`_
