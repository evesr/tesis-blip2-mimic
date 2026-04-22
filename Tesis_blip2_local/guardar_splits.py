"""
Script para guardar los splits Train/Validation/Test
======================================================
Genera archivos CSV separados para garantizar reproducibilidad.
IMPORTANTE: Divide por ESTUDIO (study_id) para evitar data leakage.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

print("="*70)
print("GUARDANDO SPLITS TRAIN/VALIDATION/TEST (POR ESTUDIO)")
print("="*70)

# Cargar dataset del 1%
print("\n[1/4] Cargando dataset del 1%...")
df_1pct = pd.read_csv("dataset_1_porciento_final.csv")
print(f"  ✓ Cargados {len(df_1pct):,} registros")
print(f"  ✓ Estudios únicos: {df_1pct['study_id'].nunique():,}")

# Obtener lista única de estudios
print("\n[2/4] Obteniendo lista de estudios únicos...")
studies = df_1pct[['study_id', 'subject_id']].drop_duplicates('study_id')
print(f"  ✓ Total de estudios: {len(studies):,}")

# Crear splits con semilla fija (DIVIDIENDO POR ESTUDIOS)
print("\n[3/4] Creando splits por estudio (random_state=42)...")

# Primero separar 10% de estudios para test
studies_temp, studies_test = train_test_split(
    studies, 
    test_size=0.10, 
    random_state=42,
    stratify=studies['subject_id'].apply(lambda x: x % 10)
)

# Luego separar el 90% restante en 70% train y 20% validation
studies_train, studies_val = train_test_split(
    studies_temp, 
    test_size=0.2222,  # 20% del dataset total
    random_state=42,
    stratify=studies_temp['subject_id'].apply(lambda x: x % 10)
)

# Asignar todas las imágenes de cada estudio al split correspondiente
df_train = df_1pct[df_1pct['study_id'].isin(studies_train['study_id'])]
df_val = df_1pct[df_1pct['study_id'].isin(studies_val['study_id'])]
df_test = df_1pct[df_1pct['study_id'].isin(studies_test['study_id'])]

print(f"  ✓ Train:      {len(studies_train):,} estudios → {len(df_train):,} imágenes ({len(df_train)/len(df_1pct)*100:.1f}%)")
print(f"  ✓ Validation: {len(studies_val):,} estudios → {len(df_val):,} imágenes ({len(df_val)/len(df_1pct)*100:.1f}%)")
print(f"  ✓ Test:       {len(studies_test):,} estudios → {len(df_test):,} imágenes ({len(df_test)/len(df_1pct)*100:.1f}%)")

# Guardar splits
print("\n[4/4] Guardando archivos CSV...")
df_train.to_csv("train_split.csv", index=False)
print(f"  ✓ train_split.csv guardado ({len(df_train):,} registros)")

df_val.to_csv("val_split.csv", index=False)
print(f"  ✓ val_split.csv guardado ({len(df_val):,} registros)")

df_test.to_csv("test_split.csv", index=False)
print(f"  ✓ test_split.csv guardado ({len(df_test):,} registros)")

# Verificación
print("\n" + "="*70)
print("VERIFICACIÓN DE SPLITS")
print("="*70)

# Verificar que no hay overlap de ESTUDIOS
train_studies = set(df_train['study_id'].unique())
val_studies = set(df_val['study_id'].unique())
test_studies = set(df_test['study_id'].unique())

overlap_train_val = train_studies & val_studies
overlap_train_test = train_studies & test_studies
overlap_val_test = val_studies & test_studies

print(f"\nEstudios únicos por split:")
print(f"  Train:      {len(train_studies):,}")
print(f"  Validation: {len(val_studies):,}")
print(f"  Test:       {len(test_studies):,}")

print(f"\nVerificación de data leakage (por estudio):")
print(f"  Train ∩ Validation: {len(overlap_train_val)} estudios (debe ser 0)")
print(f"  Train ∩ Test:       {len(overlap_train_test)} estudios (debe ser 0)")
print(f"  Validation ∩ Test:  {len(overlap_val_test)} estudios (debe ser 0)")

if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
    print("\n✅ ¡Splits correctos! No hay data leakage entre conjuntos.")
else:
    print("\n⚠️  ADVERTENCIA: Hay estudios compartidos entre splits.")

print("\n" + "="*70)
print("✓ ARCHIVOS GENERADOS EXITOSAMENTE")
print("="*70)
print("\nArchivos creados:")
print("  - train_split.csv")
print("  - val_split.csv")
print("  - test_split.csv")
print("\nEstos archivos garantizan reproducibilidad en cualquier máquina.")
print("="*70)
