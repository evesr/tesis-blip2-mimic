"""
Script de Limpieza de Reportes MIMIC-CXR
========================================
Elimina marcadores de anonimización y estructura de plantilla.
Crea backup antes de procesar.
"""

import re
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
INPUT_FILE = "mimic_database_unificada.csv"
BACKUP_FILE = f"mimic_database_unificada_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
OUTPUT_FILE = "mimic_database_unificada_limpia.csv"

# Patrones a eliminar (más agresivos)
PATRONES_ANONIMIZACION = [
    r'\b\w+\s+_{2,}\b',           # Palabras seguidas de 2+ guiones bajos (ej: "comparison ___")
    r'\b_{2,}\s+\w+\b',           # 2+ guiones bajos seguidos de palabras (ej: "___ findings")
    r'\b_{2,}\b',                 # Cualquier secuencia de 2+ guiones bajos aislados
    r'___ year old',              # Específico
    r'___ year',
    r'year old',
]

PATRONES_PLANTILLA = [
    r'final report examination',  # Específico primero (más largo)
    r'final report',
    r'report examination',
    r'examination chest',         # Ruido de encabezado "EXAMINATION: CHEST"
    r'examination\s+\w+',         # "examination" seguido de cualquier palabra
    r'indication\b',              # Palabra "indication" completa
    r'comparison\b',              # Palabra "comparison" completa
]

# ============================================================================
# FUNCIONES
# ============================================================================

def limpiar_texto(text):
    """
    Aplica todas las reglas de limpieza al texto del reporte.
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    texto_limpio = text
    
    # Eliminar marcadores de anonimización (case insensitive)
    for patron in PATRONES_ANONIMIZACION:
        texto_limpio = re.sub(patron, '', texto_limpio, flags=re.IGNORECASE)
    
    # Eliminar estructura de plantilla
    for patron in PATRONES_PLANTILLA:
        texto_limpio = re.sub(patron, '', texto_limpio, flags=re.IGNORECASE)
    
    # Limpiar espacios múltiples
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio)
    
    # Limpiar líneas vacías múltiples
    texto_limpio = re.sub(r'\n\s*\n+', '\n\n', texto_limpio)
    
    # Trim
    texto_limpio = texto_limpio.strip()
    
    return texto_limpio


def main():
    print("=" * 70)
    print("LIMPIEZA DE REPORTES MIMIC-CXR")
    print("=" * 70)
    
    # 1. Crear backup
    print(f"\n[1/4] Creando backup...")
    shutil.copy2(INPUT_FILE, BACKUP_FILE)
    print(f"  ✓ Backup guardado: {BACKUP_FILE}")
    
    # 2. Cargar datos
    print(f"\n[2/4] Cargando datos...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  ✓ Cargados {len(df):,} registros")
    
    # Estadísticas ANTES de limpieza
    print(f"\n[3/4] Estadísticas ANTES de limpieza:")
    tokens_antes = df['report_text'].apply(lambda x: len(str(x).split())).sum()
    print(f"  - Total de tokens: {tokens_antes:,}")
    print(f"  - Longitud promedio: {df['report_text'].str.len().mean():.1f} caracteres")
    
    # Muestra antes
    print(f"\n  Ejemplo de texto ANTES:")
    print(f"  {'-'*66}")
    sample_text = df['report_text'].iloc[100][:200]
    print(f"  {sample_text}...")
    print(f"  {'-'*66}")
    
    # 3. Aplicar limpieza
    print(f"\n[4/4] Aplicando limpieza...")
    df['report_text_limpio'] = df['report_text'].apply(limpiar_texto)
    
    # Estadísticas DESPUÉS
    tokens_despues = df['report_text_limpio'].apply(lambda x: len(str(x).split())).sum()
    reduccion_tokens = tokens_antes - tokens_despues
    porcentaje_reduccion = (reduccion_tokens / tokens_antes) * 100
    
    print(f"\n  ✓ Limpieza completada")
    print(f"\n  Estadísticas DESPUÉS de limpieza:")
    print(f"  - Total de tokens: {tokens_despues:,}")
    print(f"  - Tokens eliminados: {reduccion_tokens:,} ({porcentaje_reduccion:.1f}%)")
    print(f"  - Longitud promedio: {df['report_text_limpio'].str.len().mean():.1f} caracteres")
    
    # Muestra después
    print(f"\n  Ejemplo de texto DESPUÉS:")
    print(f"  {'-'*66}")
    sample_text_clean = df['report_text_limpio'].iloc[100][:200]
    print(f"  {sample_text_clean}...")
    print(f"  {'-'*66}")
    
    # 4. Guardar resultado
    # Reemplazar la columna original con la limpia
    df['report_text'] = df['report_text_limpio']
    df = df.drop(columns=['report_text_limpio'])
    
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ PROCESO COMPLETADO")
    print(f"{'='*70}")
    print(f"  Backup original:  {BACKUP_FILE}")
    print(f"  Dataset limpio:   {OUTPUT_FILE}")
    print(f"  Registros:        {len(df):,}")
    print(f"  Reducción:        {reduccion_tokens:,} tokens ({porcentaje_reduccion:.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
