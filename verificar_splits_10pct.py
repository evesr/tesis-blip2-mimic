#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificación de Integridad - Splits 10%
========================================
Verifica que los splits están correctos y completos.
"""

import pandas as pd

print("="*80)
print("VERIFICACIÓN DE INTEGRIDAD - SPLITS 10%")
print("="*80)

# Cargar dataset original
print("\n[1/5] Cargando dataset original del 10%...")
df_original = pd.read_csv("dataset_10_porciento_final.csv")
print(f"      → {len(df_original):,} registros")

# Cargar splits
print("\n[2/5] Cargando splits...")
df_train = pd.read_csv("train_split_10pct.csv")
df_val = pd.read_csv("val_split_10pct.csv")
df_test = pd.read_csv("test_split_10pct.csv")

print(f"      → Train:      {len(df_train):,} registros")
print(f"      → Validation: {len(df_val):,} registros")
print(f"      → Test:       {len(df_test):,} registros")
print(f"      → TOTAL:      {len(df_train) + len(df_val) + len(df_test):,} registros")

# Verificar totales
print("\n[3/5] Verificando totales...")
total_splits = len(df_train) + len(df_val) + len(df_test)
if total_splits == len(df_original):
    print(f"      ✅ Total correcto: {total_splits:,} = {len(df_original):,}")
else:
    print(f"      ❌ ERROR: {total_splits:,} ≠ {len(df_original):,}")

# Verificar que todos los dicom_ids existen
print("\n[4/5] Verificando dicom_ids...")
original_ids = set(df_original['dicom_id'])
train_ids = set(df_train['dicom_id'])
val_ids = set(df_val['dicom_id'])
test_ids = set(df_test['dicom_id'])

all_split_ids = train_ids | val_ids | test_ids

missing = original_ids - all_split_ids
extra = all_split_ids - original_ids

print(f"      → IDs en original:  {len(original_ids):,}")
print(f"      → IDs en splits:    {len(all_split_ids):,}")
print(f"      → IDs faltantes:    {len(missing):,}")
print(f"      → IDs extra:        {len(extra):,}")

if len(missing) == 0 and len(extra) == 0:
    print(f"      ✅ Todos los IDs coinciden")
else:
    print(f"      ❌ Hay IDs faltantes o extra")

# Verificar overlap entre splits
print("\n[5/5] Verificando data leakage...")
overlap_train_val = train_ids & val_ids
overlap_train_test = train_ids & test_ids
overlap_val_test = val_ids & test_ids

print(f"      → Train ∩ Validation: {len(overlap_train_val)} imágenes")
print(f"      → Train ∩ Test:       {len(overlap_train_test)} imágenes")
print(f"      → Validation ∩ Test:  {len(overlap_val_test)} imágenes")

if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
    print(f"      ✅ NO HAY DATA LEAKAGE")
else:
    print(f"      ❌ HAY DATA LEAKAGE")

# Verificar splits por estudio
print("\n" + "="*80)
print("VERIFICACIÓN POR ESTUDIO")
print("="*80)

train_studies = set(df_train['study_id'])
val_studies = set(df_val['study_id'])
test_studies = set(df_test['study_id'])

overlap_studies_tv = train_studies & val_studies
overlap_studies_tt = train_studies & test_studies
overlap_studies_vt = val_studies & test_studies

print(f"Estudios en Train:      {len(train_studies):,}")
print(f"Estudios en Validation: {len(val_studies):,}")
print(f"Estudios en Test:       {len(test_studies):,}")
print(f"\nOverlap de estudios:")
print(f"  Train ∩ Validation: {len(overlap_studies_tv)} estudios")
print(f"  Train ∩ Test:       {len(overlap_studies_tt)} estudios")
print(f"  Validation ∩ Test:  {len(overlap_studies_vt)} estudios")

if len(overlap_studies_tv) == 0 and len(overlap_studies_tt) == 0 and len(overlap_studies_vt) == 0:
    print(f"\n✅ Splits por estudio correctos - NO HAY DATA LEAKAGE")
else:
    print(f"\n❌ HAY DATA LEAKAGE a nivel de estudio")

print("\n" + "="*80)
if total_splits == len(df_original) and len(missing) == 0 and len(extra) == 0 and \
   len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
    print("✅ VERIFICACIÓN EXITOSA: Splits están correctos e íntegros")
else:
    print("❌ VERIFICACIÓN FALLIDA: Hay problemas en los splits")
print("="*80)
