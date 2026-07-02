#!/bin/bash
################################################################################
# run_phase3_10pct_tmux.sh — Launcher de Tmux para Fase 3.2 (10%)
################################################################################
# Lanza el torneo de la Fase 3.2 en una sesión tmux persistente.
# La sesión se mantiene activa incluso si te desconectas del servidor.
#
# Uso:
#   bash run_phase3_10pct_tmux.sh
#
# Comandos útiles de tmux:
#   tmux ls                      # Listar sesiones activas
#   tmux attach -t fase3-10pct   # Conectarse a la sesión
#   Ctrl+B, luego D              # Desconectarse (sin cerrar)
#   tmux kill-session -t fase3-10pct  # Matar la sesión
#
# Autor: Evelyn Silva Rozas
# Fecha: Mayo 2026
################################################################################

SESSION_NAME="fase3-10pct"
LOG_FILE="phase3_results-10pct/tournament_phase3_10pct.log"

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================================================="
echo "  LAUNCHER TMUX — FASE 3.2 (Escalamiento Final al 10%)"
echo "=========================================================================="
echo ""

# Verificar si tmux está instalado
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}✗ Error: tmux no está instalado${NC}"
    echo "  Instálalo con: sudo apt-get install tmux"
    exit 1
fi

# Verificar si la sesión ya existe
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  La sesión tmux '$SESSION_NAME' ya existe${NC}"
    echo ""
    echo "Opciones:"
    echo "  1) Adjuntarse a la sesión existente (ver progreso)"
    echo "  2) Matar la sesión y crear una nueva"
    echo "  3) Cancelar"
    echo ""
    read -p "Elige una opción [1-3]: " choice
    
    case $choice in
        1)
            echo -e "${GREEN}✓ Adjuntándose a la sesión '$SESSION_NAME'...${NC}"
            tmux attach-session -t "$SESSION_NAME"
            exit 0
            ;;
        2)
            echo -e "${YELLOW}⚠️  Matando sesión '$SESSION_NAME'...${NC}"
            tmux kill-session -t "$SESSION_NAME"
            sleep 1
            ;;
        3)
            echo "Cancelado."
            exit 0
            ;;
        *)
            echo -e "${RED}✗ Opción inválida${NC}"
            exit 1
            ;;
    esac
fi

# Crear directorio de resultados si no existe
mkdir -p "$(dirname "$LOG_FILE")"

echo -e "${GREEN}✓ Creando nueva sesión tmux '$SESSION_NAME'...${NC}"
echo ""
echo "Configuración:"
echo "  • Sesión: $SESSION_NAME"
echo "  • Script: run_phase3_10pct.py"
echo "  • Log: $LOG_FILE"
echo "  • Configs: Config_Final_A, Config_Final_B"
echo ""
echo "Comandos útiles:"
echo "  • Ver progreso:    tmux attach -t $SESSION_NAME"
echo "  • Desconectarse:   Ctrl+B, luego D"
echo "  • Ver log:         tail -f $LOG_FILE"
echo "  • Matar sesión:    tmux kill-session -t $SESSION_NAME"
echo ""

# Crear la sesión tmux y ejecutar el script
tmux new-session -d -s "$SESSION_NAME" \
    "python run_phase3_10pct.py 2>&1 | tee $LOG_FILE; echo ''; echo '========================================'; echo 'Torneo finalizado. Presiona Enter para cerrar la sesión.'; read"

echo -e "${GREEN}✓ Sesión '$SESSION_NAME' creada exitosamente${NC}"
echo ""
echo "El torneo está corriendo en background."
echo ""
echo "Para ver el progreso en tiempo real:"
echo -e "  ${YELLOW}tmux attach -t $SESSION_NAME${NC}"
echo ""
echo "O monitorear el log:"
echo -e "  ${YELLOW}tail -f $LOG_FILE${NC}"
echo ""
echo "=========================================================================="
echo ""

# Preguntar si desea adjuntarse inmediatamente
read -p "¿Adjuntarse a la sesión ahora? [s/N]: " attach_now
case $attach_now in
    [sS]|[sS][iI])
        echo ""
        echo -e "${GREEN}✓ Adjuntándose a la sesión '$SESSION_NAME'...${NC}"
        echo "  (Presiona Ctrl+B, luego D para desconectarte sin cerrar la sesión)"
        sleep 2
        tmux attach-session -t "$SESSION_NAME"
        ;;
    *)
        echo ""
        echo -e "${GREEN}✓ La sesión está corriendo en background${NC}"
        echo "  Usa: tmux attach -t $SESSION_NAME  para conectarte más tarde"
        ;;
esac

exit 0
