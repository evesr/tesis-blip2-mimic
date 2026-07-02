#!/bin/bash
###############################################################################
# descargar_asclepius_aws.sh — Descarga Asclepius-R vía AWS S3 (PhysioNet)
#                              La vía MÁS RÁPIDA: S3 no limita por conexión.
###############################################################################
#
# ¿POR QUÉ S3 ES LO MÁS RÁPIDO?
#   El servidor web de PhysioNet limita la velocidad POR CONEXIÓN TCP (de ahí
#   las ~12 h con wget). Su ESPEJO en AWS S3 (mismo Access Point de la cuenta
#   724665945834 que ya usaste para MIMIC-CXR) NO tiene ese límite y `aws s3
#   sync` paraleliza descargas y multipart automáticamente. Además es 100 %
#   reanudable (sync solo baja lo que falta).
#
# ACCESS POINT (dado por PhysioNet para Asclepius-R):
#   arn:aws:s3:us-east-1:724665945834:accesspoint/asclepius-r-v1-1-0-01
#   prefijo de clave: asclepius-r/1.1.0/
#
# REQUISITOS (igual que tu descarga MIMIC-CXR):
#   • Acceso credenciado + DUA de Asclepius-R firmado en PhysioNet.
#   • Credenciales AWS exportadas (las que te dio PhysioNet):
#       export AWS_ACCESS_KEY_ID=...
#       export AWS_SECRET_ACCESS_KEY=...
#     (o ~/.aws/credentials con un perfil; usa --profile NOMBRE)
#
# USO:
#   export AWS_ACCESS_KEY_ID=...   AWS_SECRET_ACCESS_KEY=...
#   bash descargar_asclepius_aws.sh                 # Asclepius-R-7B (lo que usa Exp3)
#   bash descargar_asclepius_aws.sh 13B             # Asclepius-R-13B
#   bash descargar_asclepius_aws.sh 7B --pretrained # + Asclepius-R-7B-Pretrained-Only
#   bash descargar_asclepius_aws.sh --full          # TODO el repo (7B+13B+pretrained+csv)
#   bash descargar_asclepius_aws.sh 7B --dry-run    # solo listar/medir, sin bajar
#   bash descargar_asclepius_aws.sh 7B --profile physionet
#
# Autor: Evelyn Silva Rozas · Junio 2026
###############################################################################

set -euo pipefail

WORKDIR="/workspace/tesis-blip2-mimic"
REGION="us-east-1"
ACCESS_POINT="arn:aws:s3:us-east-1:724665945834:accesspoint/asclepius-r-v1-1-0-01"
KEY_PREFIX="asclepius-r/1.1.0"
S3_ROOT="s3://${ACCESS_POINT}/${KEY_PREFIX}"
DEST_BASE="$WORKDIR/models"

# ── Parsear argumentos ───────────────────────────────────────────────────────
VARIANTE="7B"
FULL=0
INCLUIR_PRETRAINED=0
DRY_RUN=0
PROFILE_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        7B|7b)   VARIANTE="7B";  shift ;;
        13B|13b) VARIANTE="13B"; shift ;;
        --full)        FULL=1; shift ;;
        --pretrained)  INCLUIR_PRETRAINED=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --profile)     PROFILE_ARGS=(--profile "$2"); shift 2 ;;
        *) echo "Argumento desconocido: $1"; echo "Uso: bash descargar_asclepius_aws.sh [7B|13B] [--full|--pretrained|--dry-run|--profile NOMBRE]"; exit 1 ;;
    esac
done

case "$VARIANTE" in
    7B)  SUBDIR="Asclepius-R-7B";  PRESUBDIR="Asclepius-R-7B-Pretrained-Only" ;;
    13B) SUBDIR="Asclepius-R-13B"; PRESUBDIR="Asclepius-R-13B-Pretrained-Only" ;;
esac

# ── Verificar aws CLI ────────────────────────────────────────────────────────
if ! command -v aws >/dev/null 2>&1; then
    echo "ERROR: aws CLI no está instalado."
    echo "  Instálalo con:  python -m pip install awscli"
    exit 1
fi

AWS_COMMON=(--region "$REGION")
[ ${#PROFILE_ARGS[@]} -gt 0 ] && AWS_COMMON+=("${PROFILE_ARGS[@]}")

echo "======================================================================"
echo "  DESCARGA Asclepius-R vía AWS S3 (espejo PhysioNet)"
echo "======================================================================"
echo "  Access Point : $ACCESS_POINT"
echo "  Región       : $REGION"
if [ "$FULL" -eq 1 ]; then
    echo "  Alcance      : TODO el repositorio ($KEY_PREFIX/)"
else
    echo "  Variante     : $SUBDIR"
    [ "$INCLUIR_PRETRAINED" -eq 1 ] && echo "  + incluye    : $PRESUBDIR"
fi
echo "  Destino      : $DEST_BASE"
echo "======================================================================"

# ── Validar credenciales AWS (STS) ───────────────────────────────────────────
echo
echo "  → Verificando credenciales AWS..."
if ! IDENT="$(aws "${AWS_COMMON[@]}" sts get-caller-identity --output text 2>&1)"; then
    echo
    echo "  ✗ Credenciales AWS no válidas o ausentes."
    echo "    $IDENT"
    echo
    echo "  Exporta las credenciales que PhysioNet te asignó para el Access Point:"
    echo "      export AWS_ACCESS_KEY_ID=tu_access_key"
    echo "      export AWS_SECRET_ACCESS_KEY=tu_secret_key"
    echo "    (o usa --profile NOMBRE con ~/.aws/credentials)"
    echo
    echo "  Verifica también que firmaste el DUA de Asclepius-R en PhysioNet."
    exit 2
fi
echo "  ✓ Identidad AWS: $IDENT"

# ── Helper de sync ───────────────────────────────────────────────────────────
sync_uno() {
    local sub="$1"
    local src="${S3_ROOT}/${sub}/"
    local dst="${DEST_BASE}/${sub}/"
    echo
    echo "  ── Sincronizando: $sub ───────────────────────────────────────────"
    echo "     src: $src"
    echo "     dst: $dst"
    mkdir -p "$dst"
    local extra=()
    [ "$DRY_RUN" -eq 1 ] && extra+=(--dryrun)
    # --exact-timestamps: comparación robusta al reanudar.
    # --only-show-errors: salida limpia en tmux (quita el spinner por archivo)
    #   PERO mostramos un resumen al final con du. Mantén progreso por defecto.
    aws "${AWS_COMMON[@]}" s3 sync "$src" "$dst" \
        --exact-timestamps \
        "${extra[@]}"
}

# ── Ejecutar ─────────────────────────────────────────────────────────────────
if [ "$FULL" -eq 1 ]; then
    echo
    echo "  ── Sincronizando TODO el repositorio ─────────────────────────────"
    echo "     src: ${S3_ROOT}/"
    echo "     dst: ${DEST_BASE}/"
    mkdir -p "$DEST_BASE"
    EXTRA=()
    [ "$DRY_RUN" -eq 1 ] && EXTRA+=(--dryrun)
    aws "${AWS_COMMON[@]}" s3 sync "${S3_ROOT}/" "${DEST_BASE}/" \
        --exact-timestamps "${EXTRA[@]}"
    PRIMARY_DEST="${DEST_BASE}/${SUBDIR}"
else
    sync_uno "$SUBDIR"
    [ "$INCLUIR_PRETRAINED" -eq 1 ] && sync_uno "$PRESUBDIR"
    PRIMARY_DEST="${DEST_BASE}/${SUBDIR}"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "  [--dry-run] Listado mostrado arriba; no se descargó nada."
    exit 0
fi

# ── Verificación final ───────────────────────────────────────────────────────
echo
echo "======================================================================"
echo "  VERIFICACIÓN  ·  $PRIMARY_DEST"
echo "======================================================================"
OK=1
if [ -f "$PRIMARY_DEST/config.json" ]; then
    echo "  ✓ config.json"
else
    echo "  ✗ FALTA config.json"; OK=0
fi
if ls "$PRIMARY_DEST"/*.bin >/dev/null 2>&1 || ls "$PRIMARY_DEST"/*.safetensors >/dev/null 2>&1; then
    n_w=$(ls "$PRIMARY_DEST"/*.bin "$PRIMARY_DEST"/*.safetensors 2>/dev/null | wc -l)
    echo "  ✓ pesos del modelo ($n_w archivo(s) .bin/.safetensors)"
else
    echo "  ✗ FALTAN pesos (*.bin / *.safetensors)"; OK=0
fi
# Tokenizer LLaMA (use_fast=False) — útil que esté presente
ls "$PRIMARY_DEST"/tokenizer*.model >/dev/null 2>&1 && echo "  ✓ tokenizer.model (LLaMA)"
echo
echo "  Contenido:"
ls -lh "$PRIMARY_DEST" 2>/dev/null | tail -n +2 | awk '{printf "     %-10s %s\n", $5, $NF}'
echo "  Tamaño total: $(du -sh "$PRIMARY_DEST" 2>/dev/null | cut -f1)"
echo "======================================================================"

if [ "$OK" -eq 1 ]; then
    echo "  ✓ Asclepius-R ($SUBDIR) listo en: $PRIMARY_DEST"
    echo
    echo "  Siguiente paso (Experimento 3):"
    [ "$VARIANTE" = "13B" ] && echo "      export ASCLEPIUS_VARIANT=Asclepius-R-13B"
    echo "      bash run_experiment_3_tmux.sh --smoke"
else
    echo "  ⚠ Descarga incompleta. Relanza este mismo script (s3 sync reanuda)."
    exit 3
fi
