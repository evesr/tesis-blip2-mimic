#!/bin/bash
###############################################################################
# Script para ejecutar Entrenamiento Semifinal en TMUX (persistente)
###############################################################################
#
# DESCRIPCIÓN:
#   Este script ejecuta el entrenamiento semifinal en una sesión de tmux,
#   permitiendo que el proceso continúe incluso si cierras VS Code o
#   se apaga la pantalla.
#
# USO:
#   bash run_semifinal_tmux.sh
#
# COMANDOS TMUX ÚTILES:
#   - Ver sesiones activas:        tmux ls
#   - Reconectar a sesión:         tmux attach -t semifinal
#   - Desconectar (sin cerrar):    Ctrl+B, luego D
#   - Matar sesión:                tmux kill-session -t semifinal
#
# MONITOREO:
#   # En otra terminal, monitorear progreso cada 10 segundos:
#   watch -n 10 python monitor_semifinal.py
#
#   # O ver logs en tiempo real:
#   tail -f semifinal_results/semifinal.log
#
# Autor: Evelyn Silva Rozas
# Fecha: Abril 2026
###############################################################################

set -e  # Detener si hay error

SESSION_NAME="semifinal"
SCRIPT_PATH="/workspace/tesis-blip2-mimic/Tesis_blip2_local/train_semifinal.py"
LOG_DIR="/workspace/tesis-blip2-mimic/Tesis_blip2_local/semifinal_results"

echo "================================================================================"
echo "🚀 EJECUTANDO ENTRENAMIENTO SEMIFINAL EN TMUX"
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
    echo "❌ ERROR: No se encontró train_semifinal.py en:"
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
tmux new-session -d -s "$SESSION_NAME" -n "semifinal"

# Configurar la sesión
tmux send-keys -t "$SESSION_NAME" "cd /workspace/tesis-blip2-mimic/Tesis_blip2_local" C-m
tmux send-keys -t "$SESSION_NAME" "clear" C-m

# Ejecutar el entrenamiento semifinal con logging
tmux send-keys -t "$SESSION_NAME" "echo '🚀 Iniciando Entrenamiento Semifinal...'" C-m
tmux send-keys -t "$SESSION_NAME" "echo 'Log: $LOG_DIR/semifinal.log'" C-m
tmux send-keys -t "$SESSION_NAME" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME" "python train_semifinal.py 2>&1 | tee $LOG_DIR/semifinal.log" C-m

echo "================================================================================"
echo "✅ SESIÓN DE TMUX CREADA Y EJECUTÁNDOSE"
echo "================================================================================"
echo ""
echo "📋 Información de la sesión:"
echo "   • Nombre: $SESSION_NAME"
echo "   • Script: train_semifinal.py"
echo "   • Log: $LOG_DIR/semifinal.log"
echo ""
echo "📊 Configuraciones a entrenar:"
echo "   • Config 4: r=16, alpha=32"
echo "   • Config 5: r=16, alpha=16"
echo "   • Config 6: r=32, alpha=64"
echo ""
echo "⏱️  Tiempo estimado: 6-18 horas (early stopping)"
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
echo "       watch -n 10 python monitor_semifinal.py"
echo ""
echo "   5️⃣  Ver logs en tiempo real:"
echo "       tail -f $LOG_DIR/semifinal.log"
echo ""
echo "   6️⃣  Verificar GPU:"
echo "       watch -n 2 nvidia-smi"
echo ""
echo "   7️⃣  Matar la sesión (detener entrenamiento):"
echo "       tmux kill-session -t $SESSION_NAME"
echo ""
echo "================================================================================"
echo "💡 TIP: Puedes cerrar esta terminal. El entrenamiento seguirá corriendo."
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
