#!/bin/bash
###############################################################################
# Script para ejecutar Grid Search en TMUX (persistente en segundo plano)
###############################################################################
#
# DESCRIPCIÓN:
#   Este script ejecuta el grid search en una sesión de tmux, permitiendo que
#   el proceso continúe incluso si cierras VS Code o se apaga la pantalla.
#
# USO:
#   bash run_grid_search_tmux.sh
#
# COMANDOS TMUX ÚTILES:
#   - Ver sesiones activas:        tmux ls
#   - Reconectar a sesión:         tmux attach -t grid_search
#   - Desconectar (sin cerrar):    Ctrl+B, luego D
#   - Matar sesión:                tmux kill-session -t grid_search
#
# MONITOREO:
#   # En otra terminal, monitorear progreso cada 10 segundos:
#   watch -n 10 python monitor_grid_search.py
#
#   # O ver logs en tiempo real:
#   tail -f grid_search_results/grid_search.log
#
# Autor: Evelyn Silva Rozas
# Fecha: Abril 2026
###############################################################################

set -e  # Detener si hay error

SESSION_NAME="grid_search"
SCRIPT_PATH="/workspace/tesis-blip2-mimic/Tesis_blip2_local/grid_search.py"
LOG_DIR="/workspace/tesis-blip2-mimic/Tesis_blip2_local/grid_search_results"

echo "================================================================================"
echo "🚀 EJECUTANDO GRID SEARCH EN TMUX"
echo "================================================================================"
echo ""

# Verificar que tmux está instalado
if ! command -v tmux &> /dev/null; then
    echo "❌ ERROR: tmux no está instalado."
    echo ""
    echo "Para instalar en Ubuntu/Debian:"
    echo "   sudo apt-get update && sudo apt-get install -y tmux"
    echo ""
    echo "Para instalar con conda:"
    echo "   conda install -c conda-forge tmux"
    exit 1
fi

# Verificar que el script existe
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ ERROR: No se encontró grid_search.py en:"
    echo "   $SCRIPT_PATH"
    exit 1
fi

# Crear directorio de logs si no existe
mkdir -p "$LOG_DIR"

# Verificar si ya existe una sesión con ese nombre
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Ya existe una sesión de tmux llamada '$SESSION_NAME'"
    echo ""
    read -p "¿Quieres reconectar a ella? (s/N): " choice
    case "$choice" in 
      s|S|si|SI|sí|SÍ ) 
        echo ""
        echo "🔗 Reconectando a sesión existente..."
        echo ""
        tmux attach -t "$SESSION_NAME"
        exit 0
        ;;
      * ) 
        echo ""
        read -p "¿Quieres matar la sesión existente y crear una nueva? (s/N): " kill_choice
        case "$kill_choice" in
          s|S|si|SI|sí|SÍ )
            echo ""
            echo "💀 Matando sesión existente..."
            tmux kill-session -t "$SESSION_NAME"
            ;;
          * )
            echo ""
            echo "❌ Operación cancelada."
            exit 0
            ;;
        esac
        ;;
    esac
fi

# Crear nueva sesión de tmux
echo "✅ Creando sesión de tmux '$SESSION_NAME'..."
echo ""

# Crear sesión en modo detached y ejecutar el script
tmux new-session -d -s "$SESSION_NAME" -n "grid_search"

# Configurar la sesión
tmux send-keys -t "$SESSION_NAME" "cd /workspace/tesis-blip2-mimic/Tesis_blip2_local" C-m
tmux send-keys -t "$SESSION_NAME" "clear" C-m

# Ejecutar el grid search con logging
tmux send-keys -t "$SESSION_NAME" "echo '🚀 Iniciando Grid Search...'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Log: $LOG_DIR/grid_search.log'" C-m
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME" "python grid_search.py 2>&1 | tee $LOG_DIR/grid_search.log" C-m

echo "================================================================================"
echo "✅ SESIÓN DE TMUX CREADA Y EJECUTÁNDOSE"
echo "================================================================================"
echo ""
echo "📋 Información de la sesión:"
echo "   • Nombre: $SESSION_NAME"
echo "   • Script: grid_search.py"
echo "   • Log: $LOG_DIR/grid_search.log"
echo ""
echo "📚 Comandos útiles:"
echo ""
echo "   1️⃣  Ver sesiones activas:"
echo "       tmux ls"
echo ""
echo "   2️⃣  Conectar a la sesión (ver ejecución en vivo):"
echo "       tmux attach -t $SESSION_NAME"
echo ""
echo "   3️⃣  Desconectar sin cerrar (desde dentro de tmux):"
echo "       Presiona: Ctrl+B, luego D"
echo ""
echo "   4️⃣  Monitorear progreso (en otra terminal):"
echo "       watch -n 10 python monitor_grid_search.py"
echo ""
echo "   5️⃣  Ver logs en tiempo real:"
echo "       tail -f $LOG_DIR/grid_search.log"
echo ""
echo "   6️⃣  Matar la sesión (detener grid search):"
echo "       tmux kill-session -t $SESSION_NAME"
echo ""
echo "================================================================================"
echo "💡 TIP: Puedes cerrar esta terminal. El grid search seguirá corriendo."
echo "================================================================================"
echo ""

# Preguntar si quiere conectarse inmediatamente
read -p "¿Quieres conectarte a la sesión ahora? (S/n): " attach_choice
case "$attach_choice" in 
  n|N|no|NO ) 
    echo ""
    echo "👋 Sesión corriendo en segundo plano. Usa 'tmux attach -t $SESSION_NAME' para conectar."
    echo ""
    ;;
  * ) 
    echo ""
    echo "🔗 Conectando a sesión..."
    echo "   (Para desconectar sin cerrar: Ctrl+B, luego D)"
    echo ""
    sleep 2
    tmux attach -t "$SESSION_NAME"
    ;;
esac
