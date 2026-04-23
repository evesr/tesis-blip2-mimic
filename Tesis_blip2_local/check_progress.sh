#!/bin/bash
# Monitorear progreso de la evaluación

LOG_FILE="/workspace/tesis-blip2-mimic/Tesis_blip2_local/evaluation_full.log"

echo "📊 Progreso de Evaluación Zero-Shot"
echo "===================================="
echo ""

# Verificar si el proceso está corriendo
if pgrep -f "zeroshot_evaluation.py" > /dev/null; then
    echo "✅ Proceso activo"
else
    echo "⚠️  Proceso no encontrado (puede haber terminado)"
fi

echo ""
echo "📝 Últimas líneas del log:"
echo "---"
tail -15 "$LOG_FILE" | grep -E "Evaluando|checkpoint|✓|completadas|Error|✅|📊" || tail -5 "$LOG_FILE"

echo ""
echo "📈 Checkpoints detectados:"
grep "checkpoint" "$LOG_FILE" | tail -5

echo ""
echo "📁 Archivos generados:"
ls -lh /workspace/tesis-blip2-mimic/Tesis_blip2_local/resultados_zeroshot/*.csv 2>/dev/null | tail -5 || echo "   (ninguno aún)"

echo ""
echo "💡 Para ver logs en tiempo real: tail -f $LOG_FILE"
