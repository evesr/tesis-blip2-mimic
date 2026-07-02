#!/bin/bash
###############################################################################
# descargar_asclepius.sh — Descarga RÁPIDA de Asclepius-R desde PhysioNet
#                          (aria2c multi-conexión, 10-16× más rápido que wget)
###############################################################################
#
# ¿POR QUÉ ES MÁS RÁPIDO?
#   PhysioNet limita la velocidad POR CONEXIÓN TCP. `wget` abre UNA sola
#   conexión → te capa a unos pocos MB/s (de ahí las ~12 h). aria2c abre hasta
#   16 conexiones por archivo y descarga varios archivos en paralelo, saturando
#   tu ancho de banda real (típicamente 8-16× más rápido). Asclepius-R NO tiene
#   espejo en AWS/GCP (a diferencia de MIMIC-CXR-JPG), así que esta es la vía.
#
# DOS PALANCAS DE VELOCIDAD:
#   1) aria2c -x16  (multi-conexión)            → ~10× sobre wget.
#   2) Descargar SOLO la variante que usas (7B ≈ 13 GB) en vez de TODO el
#      repositorio (7B + 13B + *-Pretrained-Only + mimiciii_discharge.csv ≈
#      80-100 GB). Si tu wget recursivo apuntaba a .../1.1.0/ estabas bajando
#      TODO → otra fuente del retraso.
#
# USO:
#   # Credenciales por variable de entorno (recomendado en tmux):
#   export PHYSIONET_USER=tu_usuario
#   export PHYSIONET_PASS='tu_password'
#   bash descargar_asclepius.sh                 # baja Asclepius-R-7B (default)
#   bash descargar_asclepius.sh 13B             # baja Asclepius-R-13B
#   bash descargar_asclepius.sh 7B --pretrained # incluye *-Pretrained-Only
#
#   # Sin variables: el script pedirá usuario/clave de forma interactiva.
#
# REANUDABLE: aria2c -c reanuda byte a byte; relánzalo y sigue donde quedó.
#
# Autor: Evelyn Silva Rozas · Junio 2026
###############################################################################

set -euo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
PROYECTO="asclepius-r"
VERSION="1.1.0"
HOST="https://physionet.org/files/${PROYECTO}/${VERSION}"

# ── Argumentos ───────────────────────────────────────────────────────────────
VARIANTE="${1:-7B}"          # 7B | 13B
INCLUIR_PRETRAINED=0
for a in "$@"; do
    [ "$a" = "--pretrained" ] && INCLUIR_PRETRAINED=1
done

case "$VARIANTE" in
    7B|7b)   SUBDIR="Asclepius-R-7B";  PRESUBDIR="Asclepius-R-7B-Pretrained-Only" ;;
    13B|13b) SUBDIR="Asclepius-R-13B"; PRESUBDIR="Asclepius-R-13B-Pretrained-Only" ;;
    *) echo "Variante no válida: '$VARIANTE' (usa 7B o 13B)"; exit 1 ;;
esac

DEST_BASE="$WORKDIR/models"
DEST="$DEST_BASE/$SUBDIR"
LIST_FILE="$(mktemp /tmp/asclepius_urls.XXXXXX)"
trap 'rm -f "$LIST_FILE"' EXIT

# ── Verificar aria2c ─────────────────────────────────────────────────────────
if ! command -v aria2c >/dev/null 2>&1; then
    echo "ERROR: aria2c no está instalado."
    echo "  Instálalo con:  apt-get update && apt-get install -y aria2"
    echo "  (o, sin root, usa el método wget-paralelo del final de este archivo)."
    exit 1
fi

# ── Credenciales PhysioNet (env → .netrc → prompt) ───────────────────────────
NETRC="$HOME/.netrc"
USE_NETRC=0
if [ -n "${PHYSIONET_USER:-}" ] && [ -n "${PHYSIONET_PASS:-}" ]; then
    :
elif [ -f "$NETRC" ] && grep -q "physionet.org" "$NETRC"; then
    USE_NETRC=1
    echo "  Usando credenciales de $NETRC"
else
    read -r -p "  Usuario PhysioNet: " PHYSIONET_USER
    read -r -s -p "  Password PhysioNet: " PHYSIONET_PASS; echo
fi

# curl/aria2c auth args
if [ "$USE_NETRC" -eq 1 ]; then
    CURL_AUTH=(--netrc)
    ARIA_AUTH=(--netrc-path="$NETRC")
else
    CURL_AUTH=(--user "${PHYSIONET_USER}:${PHYSIONET_PASS}")
    ARIA_AUTH=(--http-user="${PHYSIONET_USER}" --http-passwd="${PHYSIONET_PASS}")
fi

echo "======================================================================"
echo "  DESCARGA Asclepius-R  ·  variante $SUBDIR"
echo "======================================================================"
echo "  Origen  : $HOST/$SUBDIR/"
echo "  Destino : $DEST"
[ "$INCLUIR_PRETRAINED" -eq 1 ] && echo "  + incluye: $PRESUBDIR"
echo "  Motor   : aria2c (16 conexiones/archivo, 4 archivos en paralelo)"
echo "======================================================================"

# ── Enumerar archivos (recursivo) leyendo el listado HTML de PhysioNet ───────
# PhysioNet sirve un índice HTML por carpeta con <a href="archivo">. Recorremos
# subcarpetas (las variantes son planas en formato HF, pero lo hacemos genérico).
enumerar() {
    local url="$1"
    local html
    if ! html="$(curl -fsSL "${CURL_AUTH[@]}" "$url")"; then
        echo "  ⚠ No se pudo listar: $url" >&2
        return 1
    fi
    # Extraer hrefs (sin query/fragmentos)
    local h
    while IFS= read -r h; do
        case "$h" in
            ""|"../"|"./"|/*|http*|\?*|\#*) continue ;;
        esac
        if [[ "$h" == */ ]]; then
            enumerar "${url}${h}"
        else
            # Línea de URL + ruta de salida relativa (preserva estructura)
            local rel="${url#"$HOST/$SUBDIR/"}"
            echo "${url}${h}"
            echo "  dir=${DEST}/${rel%/}"
            echo "  out=${h}"
        fi
    done < <(printf '%s\n' "$html" | grep -oE 'href="[^"?#]+"' | sed -E 's/href="//; s/"//')
}

echo
echo "  → Enumerando archivos de $SUBDIR ..."
mkdir -p "$DEST"
enumerar "$HOST/$SUBDIR/" > "$LIST_FILE" || true

if [ "$INCLUIR_PRETRAINED" -eq 1 ]; then
    echo "  → Enumerando $PRESUBDIR ..."
    SUBDIR_BAK="$SUBDIR"; DEST_BAK="$DEST"
    SUBDIR="$PRESUBDIR"; DEST="$DEST_BASE/$PRESUBDIR"; mkdir -p "$DEST"
    enumerar "$HOST/$PRESUBDIR/" >> "$LIST_FILE" || true
    SUBDIR="$SUBDIR_BAK"; DEST="$DEST_BAK"
fi

N_FILES=$(grep -cE '^https?://' "$LIST_FILE" || true)
if [ "${N_FILES:-0}" -eq 0 ]; then
    echo
    echo "  ✗ No se enumeró ningún archivo. Causas probables:"
    echo "     · Credenciales incorrectas o DUA de Asclepius-R sin firmar."
    echo "     · Aún no completaste el training CITI requerido."
    echo "    Verifica acceso abriendo (logueado) en el navegador:"
    echo "       $HOST/$SUBDIR/"
    exit 2
fi
echo "  ✓ $N_FILES archivos a descargar."

# ── Descarga paralela con aria2c ─────────────────────────────────────────────
#   -x16  : hasta 16 conexiones por servidor (por archivo)  ← clave anti-throttle
#   -s16  : divide cada archivo en 16 segmentos simultáneos
#   -j4   : 4 archivos a la vez
#   -k1M  : tamaño mínimo de segmento 1 MB
#   -c    : reanudar (continue) → relanzable sin recomenzar
#   --retry/--max-tries=0 : reintentos infinitos ante cortes transitorios
echo
echo "  → Lanzando aria2c (esto satura tu ancho de banda; mucho más rápido que wget)..."
aria2c \
    "${ARIA_AUTH[@]}" \
    --input-file="$LIST_FILE" \
    --max-connection-per-server=16 \
    --split=16 \
    --max-concurrent-downloads=4 \
    --min-split-size=1M \
    --continue=true \
    --max-tries=0 \
    --retry-wait=5 \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --conditional-get=true \
    --summary-interval=10 \
    --console-log-level=warn \
    --show-console-readout=true \
    --file-allocation=falloc

# ── Verificación final ───────────────────────────────────────────────────────
echo
echo "======================================================================"
echo "  VERIFICACIÓN"
echo "======================================================================"
OK=1
if [ -f "$DEST/config.json" ]; then
    echo "  ✓ config.json"
else
    echo "  ✗ FALTA config.json en $DEST"; OK=0
fi
if ls "$DEST"/*.bin >/dev/null 2>&1 || ls "$DEST"/*.safetensors >/dev/null 2>&1; then
    echo "  ✓ pesos del modelo ($(ls "$DEST"/*.bin "$DEST"/*.safetensors 2>/dev/null | wc -l) archivo(s))"
else
    echo "  ✗ FALTAN pesos (*.bin / *.safetensors) en $DEST"; OK=0
fi
echo
echo "  Contenido de $DEST:"
ls -lh "$DEST" 2>/dev/null | tail -n +2 | awk '{printf "     %-10s %s\n", $5, $9}'
TOTAL=$(du -sh "$DEST" 2>/dev/null | cut -f1)
echo "  Tamaño total: ${TOTAL:-?}"
echo "======================================================================"

if [ "$OK" -eq 1 ]; then
    echo "  ✓ Asclepius-R ($SUBDIR) descargado y listo en:"
    echo "      $DEST"
    echo
    echo "  Siguiente paso (Experimento 3):"
    if [ "$VARIANTE" = "13B" ] || [ "$VARIANTE" = "13b" ]; then
        echo "      export ASCLEPIUS_VARIANT=Asclepius-R-13B"
    fi
    echo "      bash run_experiment_3_tmux.sh --smoke"
else
    echo "  ⚠ Descarga incompleta. Relanza este mismo script (aria2c reanuda con -c)."
    exit 3
fi

###############################################################################
# ALTERNATIVA SIN aria2c (si no tienes root para instalarlo): wget en paralelo
# -----------------------------------------------------------------------------
# Lanza varios wget a la vez (un proceso por archivo). Menos óptimo que aria2c
# (1 conexión por archivo) pero N archivos en paralelo ya supera a un wget -r:
#
#   cd /workspace/tesis-blip2-mimic/models
#   wget -r -N -c -np -nH --cut-dirs=3 \
#        --user "$PHYSIONET_USER" --password "$PHYSIONET_PASS" \
#        "https://physionet.org/files/asclepius-r/1.1.0/Asclepius-R-7B/" &
#   # …pero para paralelizar de verdad, usa la lista de URLs + xargs -P:
#   #   curl -s --user U:P .../Asclepius-R-7B/ | grep -oE 'href="[^"]+"' | ...
#   #   ... | xargs -P 8 -I{} wget -c --user U:P .../Asclepius-R-7B/{}
###############################################################################
