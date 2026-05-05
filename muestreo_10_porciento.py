"""
Script de Muestreo Estratificado 10% - MIMIC-CXR
=================================================
Extrae el 10% de pacientes (subject_id) manteniendo la proporción
Normal/Anormal según la columna 'No Finding' de CheXpert.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================================
# RUTAS
# ============================================================================
BASE_DIR = "/Users/evelynsilvarozas/Tesis_blip2_local"
UNIFIED_PATH = f"{BASE_DIR}/mimic_database_unificada.csv"
CHEXPERT_PATH = f"{BASE_DIR}/mimic-cxr-2.0.0-chexpert.csv.gz"
OUTPUT_PATH = f"{BASE_DIR}/dataset_10_porciento_final.csv"

# ============================================================================
# 1. Cargar datos
# ============================================================================
print("="*80)
print("MUESTREO ESTRATIFICADO 10% - MIMIC-CXR")
print("="*80)
print("\n[1/7] Cargando base unificada ...")
df_main = pd.read_csv(UNIFIED_PATH)
print(f"      → {len(df_main):,} filas, {df_main['subject_id'].nunique():,} pacientes")

print("\n[2/7] Cargando etiquetas CheXpert ...")
df_chexpert = pd.read_csv(CHEXPERT_PATH)
print(f"      → {len(df_chexpert):,} filas")

# ============================================================================
# 2. INNER JOIN por study_id
# ============================================================================
print("\n[3/7] INNER JOIN por study_id ...")
df_joined = df_main.merge(df_chexpert, on=["subject_id", "study_id"], how="inner")
print(f"      → {len(df_joined):,} filas tras JOIN")

# ============================================================================
# 3. Crear categoría temporal para estratificación
# ============================================================================
print("\n[4/7] Creando categorías Normal/Anormal ...")
# 'No Finding' == 1 → Normal, cualquier otro valor → Anormal
df_joined["_categoria"] = df_joined["No Finding"].apply(
    lambda x: "Normal" if x == 1 else "Anormal"
)

print("      Distribución de clases:")
dist = df_joined["_categoria"].value_counts(normalize=True)
for cat, pct in dist.items():
    print(f"        {cat:10s}: {pct*100:.2f}%")

# ============================================================================
# 4. Muestreo estratificado del 10% a nivel de paciente
# ============================================================================
print("\n[5/7] Muestreo estratificado del 10% a nivel de paciente ...")
# Tabla de pacientes únicos con su categoría mayoritaria
df_patients = (
    df_joined.groupby("subject_id")["_categoria"]
    .agg(lambda s: s.mode().iloc[0])  # categoría más frecuente del paciente
    .reset_index()
    .rename(columns={"_categoria": "_cat_paciente"})
)

print(f"      Pacientes únicos: {len(df_patients):,}")

# Separar 10% estratificado
_, sample_patients = train_test_split(
    df_patients,
    test_size=0.10,  # 10%
    stratify=df_patients["_cat_paciente"],
    random_state=42,
)

print(f"      Pacientes en muestra 10%: {len(sample_patients):,}")
print("      Distribución muestra:")
dist_sample = sample_patients["_cat_paciente"].value_counts(normalize=True)
for cat, pct in dist_sample.items():
    print(f"        {cat:10s}: {pct*100:.2f}%")

# ============================================================================
# 5. Filtrar filas del 10% de pacientes
# ============================================================================
print("\n[6/7] Filtrando filas del 10% de pacientes ...")
df_sample = df_joined[df_joined["subject_id"].isin(sample_patients["subject_id"])]
print(f"      → {len(df_sample):,} filas en muestra final")
print(f"      → {df_sample['study_id'].nunique():,} estudios únicos")

# ============================================================================
# 6. Eliminar columnas CheXpert y la columna temporal
# ============================================================================
print("\n[7/7] Limpiando columnas temporales ...")
chexpert_cols = [c for c in df_chexpert.columns if c not in ("subject_id", "study_id")]
cols_to_drop = list(chexpert_cols) + ["_categoria"]
df_sample = df_sample.drop(columns=cols_to_drop)

# ============================================================================
# 7. Guardar resultado
# ============================================================================
df_sample.to_csv(OUTPUT_PATH, index=False)

print("\n" + "="*80)
print("RESULTADO FINAL")
print("="*80)
print(f"Archivo guardado:   {OUTPUT_PATH}")
print(f"Total de filas:     {len(df_sample):,}")
print(f"Pacientes únicos:   {df_sample['subject_id'].nunique():,}")
print(f"Estudios únicos:    {df_sample['study_id'].nunique():,}")
print(f"Imágenes por estudio (promedio): {len(df_sample) / df_sample['study_id'].nunique():.2f}")
print("="*80)
print("\n✅ Muestreo del 10% completado exitosamente!")
