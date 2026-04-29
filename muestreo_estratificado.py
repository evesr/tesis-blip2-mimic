"""
Script de Muestreo Estratificado 1% - MIMIC-CXR
=================================================
Extrae el 1% de pacientes (subject_id) manteniendo la proporción
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
OUTPUT_PATH = f"{BASE_DIR}/dataset_1_porciento_final.csv"

# ============================================================================
# 1. Cargar datos
# ============================================================================
print("[INFO] Cargando base unificada ...")
df_main = pd.read_csv(UNIFIED_PATH)
print(f"  -> {len(df_main):,} filas, {df_main['subject_id'].nunique():,} pacientes")

print("[INFO] Cargando etiquetas CheXpert ...")
df_chexpert = pd.read_csv(CHEXPERT_PATH)
print(f"  -> {len(df_chexpert):,} filas")

# ============================================================================
# 2. INNER JOIN por study_id
# ============================================================================
print("[INFO] INNER JOIN por study_id ...")
df_joined = df_main.merge(df_chexpert, on=["subject_id", "study_id"], how="inner")
print(f"  -> {len(df_joined):,} filas tras JOIN")

# ============================================================================
# 3. Crear categoría temporal para estratificación
# ============================================================================
# 'No Finding' == 1 → Normal, cualquier otro valor → Anormal
df_joined["_categoria"] = df_joined["No Finding"].apply(
    lambda x: "Normal" if x == 1 else "Anormal"
)

print(f"[INFO] Distribución de clases:")
print(df_joined["_categoria"].value_counts(normalize=True).to_string())

# ============================================================================
# 4. Muestreo estratificado del 1% a nivel de paciente
# ============================================================================
# Tabla de pacientes únicos con su categoría mayoritaria
df_patients = (
    df_joined.groupby("subject_id")["_categoria"]
    .agg(lambda s: s.mode().iloc[0])  # categoría más frecuente del paciente
    .reset_index()
    .rename(columns={"_categoria": "_cat_paciente"})
)

print(f"[INFO] Pacientes únicos: {len(df_patients):,}")

# Separar 1% estratificado
_, sample_patients = train_test_split(
    df_patients,
    test_size=0.01,
    stratify=df_patients["_cat_paciente"],
    random_state=42,
)

print(f"[INFO] Pacientes en muestra 1%: {len(sample_patients):,}")
print(f"  Distribución muestra:")
print(sample_patients["_cat_paciente"].value_counts(normalize=True).to_string())

# ============================================================================
# 5. Filtrar filas del 1% de pacientes
# ============================================================================
df_sample = df_joined[df_joined["subject_id"].isin(sample_patients["subject_id"])]
print(f"[INFO] Filas en muestra final: {len(df_sample):,}")

# ============================================================================
# 6. Eliminar columnas CheXpert y la columna temporal
# ============================================================================
chexpert_cols = [c for c in df_chexpert.columns if c not in ("subject_id", "study_id")]
cols_to_drop = list(chexpert_cols) + ["_categoria"]
df_sample = df_sample.drop(columns=cols_to_drop)

# ============================================================================
# 7. Guardar resultado
# ============================================================================
df_sample.to_csv(OUTPUT_PATH, index=False)
print(f"[OK] Guardado en: {OUTPUT_PATH}  ({len(df_sample):,} filas, {df_sample['subject_id'].nunique():,} pacientes)")
