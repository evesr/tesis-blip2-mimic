#!/bin/bash
###############################################################################
# Lanzador Experimento 3 (Fase 5) — ViT(Fase 3) + Asclepius-R + Puente
# BLIP-2  ·  sesión tmux con 3 paneles (train / log / GPU)
###############################################################################
#
# Arquitectura entrenada (run_experiment_3.py):
#   vision_model   : EVA-CLIP + LoRA Fase 3   → CONGELADO (100%)
#   language_model : Asclepius-R (LLaMA clínico, notas MIMIC) → CONGELADO, SIN LoRA
#   qformer + language_projection (+ adaptador) + query_tokens → Full Fine-Tuning
#
# USO:
#   bash run_experiment_3_tmux.sh                  # entrena (resume-aware) + evalúa
#   bash run_experiment_3_tmux.sh --fresh          # ignora checkpoints previos
#   bash run_experiment_3_tmux.sh --no-query-tokens # congela query_tokens
#   bash run_experiment_3_tmux.sh --smoke          # 1 paso (valida tubería)
#   bash run_experiment_3_tmux.sh --skip-eval      # entrena sin evaluación final
#   bash run_experiment_3_tmux.sh --reattach       # reconectarse a la sesión
#
# DENTRO DE TMUX:
#   Panel superior      → run_experiment_3.py (stdout en vivo + tee al log)
#   Panel inferior izq. → tail -f del log
#   Panel inferior der. → nvidia-smi (refresco 10s)
#
# NAVEGACIÓN:
#   Ctrl+b + flecha   → cambiar de panel
#   Ctrl+b + d        → desconectarse (el entrenamiento sigue)
#   tmux attach -t exp3_asclepius   → reconectarse
#
# Autor: Evelyn Silva Rozas
# Fecha: Junio 2026
###############################################################################

set -euo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
SCRIPT="$WORKDIR/run_experiment_3.py"
OUTPUT_DIR="$WORKDIR/exp3_asclepius_results-50pct"
LOG_FILE="$OUTPUT_DIR/run_experiment_3.log"
SESSION="exp3_asclepius"

# Intérprete del sistema (SIN entornos virtuales, por requisito del proyecto)
PYBIN="python"

# Pre-requisitos
VIT_LORA_DIR="$WORKDIR/phase2_results-50pct/checkpoints/best_model_Config_B1/vit"
ASCLEPIUS_VARIANT="${ASCLEPIUS_VARIANT:-Asclepius-R-7B}"
ASCLEPIUS_DIR="${ASCLEPIUS_R_DIR:-$WORKDIR/models/$ASCLEPIUS_VARIANT}"

# ── Parsear argumentos ───────────────────────────────────────────────────────
EXTRA_ARGS=""
SMOKE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reattach)
            if tmux has-session -t "$SESSION" 2>/dev/null; then
                echo "Reconectando a sesión '$SESSION'..."
                tmux attach -t "$SESSION"
            else
                echo "ERROR: No existe la sesión '$SESSION'. Lánzala con: bash run_experiment_3_tmux.sh"
            fi
            exit 0
            ;;
        --fresh)           EXTRA_ARGS="$EXTRA_ARGS --fresh"; shift ;;
        --no-query-tokens) EXTRA_ARGS="$EXTRA_ARGS --no-query-tokens"; shift ;;
        --skip-eval)       EXTRA_ARGS="$EXTRA_ARGS --skip-eval"; shift ;;
        --smoke)           EXTRA_ARGS="$EXTRA_ARGS --smoke"; SMOKE=1; shift ;;
        *)
            echo "Argumento desconocido: $1"
            echo "Uso: bash run_experiment_3_tmux.sh [--fresh|--no-query-tokens|--smoke|--skip-eval|--reattach]"
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "  EXPERIMENTO 3 — ViT(Fase 3) + Asclepius-R + Puente (Full Fine-Tuning)"
echo "======================================================================"

# ── Verificación 0: script principal ─────────────────────────────────────────
if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: no se encontró $SCRIPT"
    exit 1
fi

# ── Verificación 1: dependencias (transformers / peft / accelerate) ──────────
# El entorno está "pelado": si faltan, run_experiment_3.py NO puede ni importar,
# así que el gate debe estar AQUÍ (bash) y NO en Python.
if ! $PYBIN -c "import transformers, peft, accelerate" >/dev/null 2>&1; then
    cat <<EOF

##############################################################################
#                          *** ACCION OBLIGATORIA ***                        #
#        (a) Reinstalar las dependencias del ecosistema Hugging Face         #
##############################################################################

El entorno NO tiene instaladas transformers / peft / accelerate (entorno
recién aprovisionado). Sin ellas el Experimento 3 no puede ejecutarse.

Instálalas en el PYTHON DEL SISTEMA (NO crear entornos virtuales, por
requisito del proyecto) con las versiones pineadas y probadas:

    $PYBIN -m pip install --no-input \\
        "transformers==4.46.3" "peft==0.13.2" \\
        "accelerate>=0.20.0" "datasets>=2.12.0" \\
        "bitsandbytes>=0.39.0" "sentencepiece"

Verifica con:
    $PYBIN -c "import transformers, peft, accelerate; \\
        print(transformers.__version__, peft.__version__)"

Cuando estén instaladas, vuelve a ejecutar:
    bash run_experiment_3_tmux.sh
##############################################################################
EOF
    exit 2
fi
echo "  Dependencias  : transformers / peft / accelerate  OK"

# ── Verificación 2: ViT LoRA de la Fase 3 ────────────────────────────────────
if [ ! -f "$VIT_LORA_DIR/adapter_config.json" ]; then
    echo "ERROR: no se encontró el adaptador LoRA del ViT (Fase 3) en:"
    echo "       $VIT_LORA_DIR/adapter_config.json"
    echo "       Completa primero la Fase 2/3 (best_model_Config_B1)."
    exit 1
fi
echo "  ViT LoRA F3   : OK ($VIT_LORA_DIR)"

# ── Verificación 3: splits del 50% ───────────────────────────────────────────
for f in train_split_50pct.csv val_split_50pct.csv test_split_50pct.csv; do
    if [ ! -f "$WORKDIR/$f" ]; then
        echo "ERROR: no se encontró $f en $WORKDIR"
        exit 1
    fi
done
echo "  Splits 50%    : OK (train/val/test)"

# ── Verificación 4: Asclepius-R (acceso credenciado) ─────────────────────────
# Requiere config.json + (pesos *.bin | *.safetensors | índice).
ASCLEPIUS_OK=0
if [ -f "$ASCLEPIUS_DIR/config.json" ]; then
    if ls "$ASCLEPIUS_DIR"/*.safetensors >/dev/null 2>&1 \
       || ls "$ASCLEPIUS_DIR"/*.bin >/dev/null 2>&1 \
       || ls "$ASCLEPIUS_DIR"/*.index.json >/dev/null 2>&1; then
        ASCLEPIUS_OK=1
    fi
fi

if [ "$ASCLEPIUS_OK" -eq 0 ]; then
    cat <<EOF

##############################################################################
#                          *** ACCION OBLIGATORIA ***                        #
#        (b) Descargar Asclepius-R (LLM clínico, acceso credenciado)         #
##############################################################################

Asclepius-R es el LLM clínico entrenado sobre notas REALES de MIMIC-III. Es de
ACCESO CREDENCIADO en PhysioNet: no se puede redistribuir ni incluir en el
repositorio, por lo que debes descargarlo TÚ con tus credenciales. NO se admite
la versión sintética pública (starmpcc/Asclepius-*) ni cuantizaciones: el
enunciado exige el modelo COMPLETO y la variante "-R".

Esperado en (variante por defecto $ASCLEPIUS_VARIANT):
    $ASCLEPIUS_DIR
(Para 13B:  export ASCLEPIUS_VARIANT=Asclepius-R-13B  antes de lanzar.)

PRE-REQUISITOS (una vez, en tu cuenta de PhysioNet):
  1) Credencialización:   https://physionet.org/settings/credentialing/
  2) Training CITI "Data or Specimens Only Research".
  3) Firmar el DUA:       https://physionet.org/sign-dua/asclepius-r/1.1.0/
  Recurso:                https://physionet.org/content/asclepius-r/1.1.0/

DESCARGA (wget recursivo de PhysioNet):
    mkdir -p "$(dirname "$ASCLEPIUS_DIR")"
    cd "$(dirname "$ASCLEPIUS_DIR")"
    wget -r -N -c -np --user TU_USUARIO_PHYSIONET --ask-password \\
      https://physionet.org/files/asclepius-r/1.1.0/$ASCLEPIUS_VARIANT/
    # Deja los archivos (config.json + *.bin/*.safetensors + tokenizer) en:
    #   $ASCLEPIUS_DIR

VERIFICA:
    ls -la "$ASCLEPIUS_DIR"

Cuando el modelo esté presente, vuelve a ejecutar:
    bash run_experiment_3_tmux.sh
##############################################################################
EOF
    exit 3
fi
echo "  Asclepius-R   : OK ($ASCLEPIUS_DIR)"

mkdir -p "$OUTPUT_DIR"
echo ""
echo "  Output dir    : $OUTPUT_DIR"
echo "  Log           : $LOG_FILE"
[ "$SMOKE" -eq 1 ] && echo "  Modo          : --smoke (1 paso, sin evaluación final)"
[ -n "$EXTRA_ARGS" ] && echo "  Args          :$EXTRA_ARGS"
echo ""

# ── Sesión previa ────────────────────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "AVISO: ya existe la sesión tmux '$SESSION'."
    printf "  [K] Matar y relanzar   [R] Reconectarse   [C] Cancelar -> "
    read -r choice
    case "$choice" in
        k|K) tmux kill-session -t "$SESSION"; echo "Sesión anterior eliminada." ;;
        r|R) tmux attach -t "$SESSION"; exit 0 ;;
        *)   echo "Cancelado."; exit 0 ;;
    esac
fi

# ── Crear sesión tmux ────────────────────────────────────────────────────────
echo "Creando sesión tmux '$SESSION'..."
tmux new-session -d -s "$SESSION" -x 220 -y 50

# Variables de entorno para eficiencia de GPU (también las fija el .py)
ENV_PREFIX="PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ASCLEPIUS_VARIANT=$ASCLEPIUS_VARIANT ASCLEPIUS_R_DIR=$ASCLEPIUS_DIR"

# Banner final SIN comillas anidadas ni Unicode (evita romper el quoting de bash)
BANNER="echo; echo ======================================================; echo '  EXPERIMENTO 3 FINALIZADO.'; echo '  Resultados en: $OUTPUT_DIR'; echo ======================================================"

# Panel superior (70%): entrenamiento + tee al log
CMD_TRAIN="cd $WORKDIR && $ENV_PREFIX $PYBIN -u $SCRIPT$EXTRA_ARGS 2>&1 | tee $LOG_FILE; $BANNER"
tmux send-keys -t "$SESSION" "$CMD_TRAIN" Enter

# Panel inferior izquierdo: tail -f del log
tmux split-window -v -p 30 -t "$SESSION"
tmux send-keys -t "$SESSION" "sleep 2; tail -f $LOG_FILE" Enter

# Panel inferior derecho: nvidia-smi
tmux split-window -h -p 50 -t "$SESSION"
tmux send-keys -t "$SESSION" "watch -n 10 -t nvidia-smi" Enter

tmux select-pane -t "$SESSION:0.0"

echo ""
echo "======================================================================"
echo "  SESIÓN TMUX '$SESSION' LISTA"
echo "======================================================================"
echo "  Panel superior      : run_experiment_3.py en vivo"
echo "  Panel inferior izq. : tail -f del log"
echo "  Panel inferior der. : nvidia-smi (refresco 10s)"
echo ""
echo "NAVEGACIÓN:"
echo "  Ctrl+b + flecha   → cambiar de panel"
echo "  Ctrl+b + d        → desconectarse (el entrenamiento sigue)"
echo "  tmux attach -t $SESSION   → reconectarse"
echo ""

# Adjuntar automáticamente si hay TTY
if [ -t 1 ]; then
    tmux attach -t "$SESSION"
fi
