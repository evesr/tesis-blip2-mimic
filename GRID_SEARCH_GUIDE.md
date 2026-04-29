# 🔍 Grid Search - Guía de Uso

## 📋 Archivos Creados

1. **`grid_search.py`** - Script principal del grid search
2. **`monitor_grid_search.py`** - Monitor de progreso en tiempo real
3. **`run_grid_search_tmux.sh`** - Ejecutor con tmux (persistente)

---

## 🚀 Opción 1: Ejecución Normal (bloqueante)

```bash
python grid_search.py
```

**Pros:**
- Simple, directo
- Ver output en consola

**Contras:**
- ❌ Si cierras la terminal, se detiene
- ❌ Si se apaga la pantalla, puede interrumpirse
- ❌ Bloquea la terminal

---

## 🎯 Opción 2: Ejecución con TMUX (recomendado)

### Paso 1: Instalar tmux (si no está instalado)

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y tmux

# Con conda
conda install -c conda-forge tmux
```

### Paso 2: Ejecutar con tmux

```bash
bash run_grid_search_tmux.sh
```

El script te preguntará si quieres conectarte a la sesión o dejarla en segundo plano.

**Pros:**
- ✅ Corre en segundo plano
- ✅ Sobrevive si cierras VS Code
- ✅ Sobrevive si se apaga la pantalla
- ✅ Puedes desconectarte y reconectarte cuando quieras

---

## 📊 Monitorear el Progreso

### Opción A: Monitor automático (recomendado)

```bash
# Actualización cada 10 segundos
watch -n 10 python monitor_grid_search.py
```

### Opción B: Monitor manual

```bash
python monitor_grid_search.py
```

### Opción C: Ver logs en vivo

```bash
tail -f grid_search_results/grid_search.log
```

### Opción D: Ver archivos de checkpoint

```bash
# Progreso general
cat grid_search_results/progress.txt

# Resultados intermedios
head -20 grid_search_results/intermediate_results.csv
```

---

## 🎮 Comandos TMUX Útiles

### Ver sesiones activas
```bash
tmux ls
```

### Conectar a la sesión (ver en vivo)
```bash
tmux attach -t grid_search
```

### Desconectar SIN cerrar (desde dentro de tmux)
```
Presiona: Ctrl+B, luego D
```

### Matar la sesión (detener grid search)
```bash
tmux kill-session -t grid_search
```

### Navegación dentro de tmux
- `Ctrl+B` luego `[` - Modo scroll (navegar con flechas)
- `q` - Salir del modo scroll

---

## 📁 Archivos Generados

Durante la ejecución:

```
grid_search_results/
├── progress.txt                    # Progreso actual (actualizado continuamente)
├── intermediate_results.csv        # Resultados parciales (checkpoint)
├── grid_search.log                 # Log completo (solo si usas tmux)
└── (al finalizar)
    ├── final_results_ranked.csv    # Mejores combinaciones ordenadas
    ├── all_epochs_results.csv      # Todas las épocas de todas las combinaciones
    └── plots/                      # 24 gráficos individuales + 1 comparativo
```

---

## 🔍 Checkpoints y Recuperación

### ¿Qué se guarda automáticamente?

1. **Cada época completada** → `intermediate_results.csv` actualizado
2. **Cada combinación completa** → `progress.txt` actualizado
3. **Timestamp de última actualización** → Para saber si sigue vivo

### Si se interrumpe el grid search:

```bash
# Ver hasta dónde llegó
python monitor_grid_search.py

# Ver resultados parciales
cat grid_search_results/progress.txt
```

Los resultados en `intermediate_results.csv` están completos hasta la última época exitosa.

---

## 🎯 Workflow Recomendado

### Terminal 1: Ejecutar grid search
```bash
cd /workspace/tesis-blip2-mimic/Tesis_blip2_local
bash run_grid_search_tmux.sh
# Presiona 'n' para dejarlo en segundo plano
```

### Terminal 2: Monitorear progreso
```bash
cd /workspace/tesis-blip2-mimic/Tesis_blip2_local
watch -n 10 python monitor_grid_search.py
```

### Terminal 3 (opcional): Ver logs
```bash
tail -f grid_search_results/grid_search.log
```

**Ahora puedes:**
- Cerrar VS Code ✅
- Apagar la pantalla ✅
- Ir a dormir ✅
- Volver en 4 horas y reconectarte ✅

---

## ⏱️ Tiempo Estimado

- **Por época**: ~5-8 minutos
- **Por combinación**: 4 épocas × 6 min = ~24 minutos
- **Total**: 24 combinaciones × 24 min = **~9-10 horas**

---

## 🆘 Troubleshooting

### "tmux: command not found"
```bash
sudo apt-get install tmux
```

### "Session grid_search already exists"
```bash
# Opción 1: Reconectar
tmux attach -t grid_search

# Opción 2: Matar y crear nueva
tmux kill-session -t grid_search
bash run_grid_search_tmux.sh
```

### Ver si el grid search sigue corriendo
```bash
# Ver última actualización
cat grid_search_results/progress.txt

# Ver procesos Python
ps aux | grep grid_search.py
```

### CUDA Out of Memory
El script ya tiene limpieza agresiva de memoria, pero si falla:
- Reduce `batch_size` de 4 a 2 en `grid_search.py` línea 144
- Reduce `eval_samples` de 50 a 30 en línea 146

---

## 📞 Soporte

Si tienes problemas:
1. Revisa `grid_search_results/grid_search.log`
2. Ejecuta `python monitor_grid_search.py` para ver estado
3. Verifica que tmux esté instalado: `tmux -V`

---

**¡Listo para ejecutar!** 🚀
