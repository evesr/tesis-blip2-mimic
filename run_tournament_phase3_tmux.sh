#!/bin/bash
#
# run_tournament_phase3_tmux.sh — Ejecutor de Fase 3 con tmux
# ============================================================
# Lanza el mini-torneo de Q-Former en una sesión tmux persistente que
# sobrevive a desconexiones de SSH, cierre de VS Code, suspensión del laptop, etc.
#
# Uso:
#   bash run_tournament_phase3_tmux.sh
#
# Comandos tmux útiles:
#   tmux ls                    # Ver sesiones activas
#   tmux attach -t fase3       # Conectarse a la sesión
#   Ctrl+B, luego D            # Desconectarse SIN cerrar (desde dentro)
#   tmux kill-session -t fase3 # Detener el torneo
#
# Autor: Evelyn Silva Rozas
# Fecha: Mayo 2026
# ============================================================

SESSION_NAME="fase3"

echo ""
echo "================================================================"
echo "  FASE 3 — OPTIMIZACIÓN SEMÁNTICA Q-FORMER"
echo "  Lanzando torneo en sesión tmux: $SESSION_NAME"
echo "================================================================"
echo ""

# Verificar si tmux está instalado
if ! command -v tmux &> /dev/null; then
    echo "❌ ERROR: tmux no está instalado."
    echo ""
    echo "Instálalo con:"
    echo "  Ubuntu/Debian:  sudo apt-get install -y tmux"
    echo "  Conda:          conda install -c conda-forge tmux"
    echo ""
    exit 1
fi

# Verificar si la sesión ya existe
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "⚠️  La sesión tmux '$SESSION_NAME' ya existe."
    echo ""
    echo "¿Qué quieres hacer?"
    echo "  1) Conectarme a la sesión existente (para ver el progreso)"
    echo "  2) Matar la sesión y crear una nueva (ADVERTENCIA: detiene el torneo en curso)"
    echo "  3) Cancelar"
    echo ""
    read -p "Opción [1/2/3]: " choice
    
    case $choice in
        1)
            echo ""
            echo "🔗 Conectándote a la sesión existente..."
            echo "   (Presiona Ctrl+B, luego D para desconectarte sin cerrar)"
            sleep 2
            tmux attach -t $SESSION_NAME
            exit 0
            ;;
        2)
            echo ""
            echo "💀 Matando sesión existente..."
            tmux kill-session -t $SESSION_NAME
            echo "   Sesión eliminada."
            ;;
        3)
            echo ""
            echo "❌ Operación cancelada."
            exit 0
            ;;
        *)
            echo ""
            echo "❌ Opción inválida. Cancelando."
            exit 1
            ;;
    esac
fi

# Crear nueva sesión tmux en segundo plano con el comando de entrenamiento
echo ""
echo "🚀 Creando nueva sesión tmux '$SESSION_NAME'..."
echo ""

tmux new-session -d -s $SESSION_NAME "python run_tournament_phase3.py 2>&1 | tee phase3_results/tournament_phase3.log"

echo "✅ Sesión tmux creada exitosamente."
echo ""
echo "================================================================"
echo "  OPCIONES DE MONITOREO"
echo "================================================================"
echo ""
echo "  1️⃣  Ver en vivo (conectarse a tmux):"
echo "      tmux attach -t $SESSION_NAME"
echo "      (Presiona Ctrl+B, luego D para desconectarte sin cerrar)"
echo ""
echo "  2️⃣  Ver log en tiempo real:"
echo "      tail -f phase3_results/tournament_phase3.log"
echo ""
echo "  3️⃣  Ver progreso de entrenamiento:"
echo "      watch -n 10 'tail -20 phase3_results/history_phase3_*.csv'"
echo ""
echo "  4️⃣  Matar el torneo:"
echo "      tmux kill-session -t $SESSION_NAME"
echo ""
echo "================================================================"
echo ""
echo "¿Quieres conectarte a la sesión ahora? [s/N]"
read -p "> " connect_now

if [[ "$connect_now" =~ ^[sS]$ ]]; then
    echo ""
    echo "🔗 Conectándote a tmux..."
    echo "   (Presiona Ctrl+B, luego D para desconectarte sin cerrar)"
    sleep 2
    tmux attach -t $SESSION_NAME
else
    echo ""
    echo "✅ El torneo está corriendo en segundo plano."
    echo "   Conéctate en cualquier momento con: tmux attach -t $SESSION_NAME"
    echo ""
fi

exit 0
