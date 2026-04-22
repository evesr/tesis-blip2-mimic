"""
Script de Limpieza de Reportes en Splits
=========================================
Aplica limpieza de texto a los archivos train/val/test split.
"""

import re
import pandas as pd
from datetime import datetime

# ============================================================================
# PATRONES DE LIMPIEZA (mismos que limpiar_reportes.py)
# ============================================================================

PATRONES_ANONIMIZACION = [
    r'\b\w+\s+_{2,}\b',           # Palabras seguidas de 2+ guiones bajos
    r'\b_{2,}\s+\w+\b',           # 2+ guiones bajos seguidos de palabras
    r'\b_{2,}\b',                 # Cualquier secuencia de 2+ guiones bajos
    r'___ year old',              # Específico
    r'___ year',
    r'year old',
]

PATRONES_PLANTILLA = [
    r'final report examination',  # Específico primero
    r'final report',
    r'report examination',
    r'examination chest',
    r'examination\s+\w+',         # "examination" seguido de cualquier palabra
    r'indication\b',
    r'comparison\b',
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
    
    # Eliminar marcadores de anonimización
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


def limpiar_split(input_file, output_file):
    """
    Limpia un archivo de split y guarda la versión limpia.
    """
    print(f"\n{'='*70}")
    print(f"Procesando: {input_file}")
    print(f"{'='*70}")
    
    # Cargar
    print(f"[1/3] Cargando archivo...")
    df = pd.read_csv(input_file)
    print(f"  ✓ Cargados {len(df):,} registros")
    
    # Estadísticas ANTES
    tokens_antes = df['report_text'].apply(lambda x: len(str(x).split())).sum()
    print(f"\n[2/3] Estadísticas ANTES de limpieza:")
    print(f"  - Total de tokens: {tokens_antes:,}")
    
    # Aplicar limpieza
    print(f"\n[3/3] Aplicando limpieza...")
    df['report_text'] = df['report_text'].apply(limpiar_texto)
    
    # Estadísticas DESPUÉS
    tokens_despues = df['report_text'].apply(lambda x: len(str(x).split())).sum()
    reduccion = tokens_antes - tokens_despues
    porcentaje = (reduccion / tokens_antes) * 100
    
    print(f"  ✓ Limpieza completada")
    print(f"\n  Estadísticas DESPUÉS:")
    print(f"  - Total de tokens: {tokens_despues:,}")
    print(f"  - Tokens eliminados: {reduccion:,} ({porcentaje:.1f}%)")
    
    # Guardar
    df.to_csv(output_file, index=False)
    print(f"\n  ✓ Archivo guardado: {output_file}")
    
    return tokens_antes, tokens_despues, reduccion


def main():
    print("="*70)
    print("LIMPIEZA DE REPORTES EN SPLITS")
    print("="*70)
    
    archivos = [
        ('train_split.csv', 'train_split_limpio.csv'),
        ('val_split.csv', 'val_split_limpio.csv'),
        ('test_split.csv', 'test_split_limpio.csv')
    ]
    
    resultados = []
    
    for input_file, output_file in archivos:
        tokens_antes, tokens_despues, reduccion = limpiar_split(input_file, output_file)
        resultados.append({
            'archivo': input_file,
            'tokens_antes': tokens_antes,
            'tokens_despues': tokens_despues,
            'reduccion': reduccion
        })
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE LIMPIEZA")
    print("="*70)
    
    total_antes = sum(r['tokens_antes'] for r in resultados)
    total_despues = sum(r['tokens_despues'] for r in resultados)
    total_reduccion = total_antes - total_despues
    
    for r in resultados:
        pct = (r['reduccion'] / r['tokens_antes']) * 100
        print(f"\n{r['archivo']}:")
        print(f"  Antes:     {r['tokens_antes']:,} tokens")
        print(f"  Después:   {r['tokens_despues']:,} tokens")
        print(f"  Reducción: {r['reduccion']:,} tokens ({pct:.1f}%)")
    
    print("\n" + "-"*70)
    pct_total = (total_reduccion / total_antes) * 100
    print(f"TOTAL:")
    print(f"  Antes:     {total_antes:,} tokens")
    print(f"  Después:   {total_despues:,} tokens")
    print(f"  Reducción: {total_reduccion:,} tokens ({pct_total:.1f}%)")
    
    print("\n" + "="*70)
    print("✓ LIMPIEZA COMPLETADA")
    print("="*70)
    print("\nArchivos generados:")
    print("  - train_split_limpio.csv")
    print("  - val_split_limpio.csv")
    print("  - test_split_limpio.csv")
    print("="*70)


if __name__ == "__main__":
    main()
