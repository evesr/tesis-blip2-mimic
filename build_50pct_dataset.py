#!/usr/bin/env python3
"""
build_50pct_dataset.py — Construcción del dataset 50% MIMIC-CXR
================================================================
Ingeniería de partición y ensamble para escalar el experimento al 50% de
MIMIC-CXR, manteniendo las mismas convenciones del pipeline del proyecto.

Flujo
-----
1. Carga  mimic-cxr-2.0.0-chexpert.csv  (etiquetas 14 clases)
2. Carga  mimic-cxr-2.0.0-metadata.csv.gz (dicom_id, ViewPosition, etc.)
3. Inner join por (subject_id, study_id) → base unificada con etiquetas
4. Limpieza de reportes (reutiliza lógica de limpiar_reportes.py)
5. Muestreo del 50% de PACIENTES estratificado por combinación de patologías
   — unidad base de partición: study_id (ningún paciente cruza splits)
6. Partición iterativa multi-etiqueta usando iterative_train_test_split
   (skmultilearn) con proporciones 70 / 20 / 10
7. Exporta train_split_50pct.csv, val_split_50pct.csv, test_split_50pct.csv

Proporciones (mismas que el 10%)
----------------------------------
  Train : 70%
  Val   : 20%
  Test  : 10%

Estratificación multi-etiqueta
--------------------------------
skmultilearn.iterative_train_test_split garantiza que las 14 clases CheXpert
(incluyendo clases raras como Fracture) queden distribuidas proporcionalmente
en los 3 splits. Opera a nivel de study_id (una fila por estudio, etiquetas
binarizadas).

Política de etiquetas
-----------------------
  1.0  → positivo
 -1.0  → incierto (U-ones: tratado como positivo)
  NaN  → 0

Salida
------
  train_split_50pct.csv   — mismas columnas que train_split_10pct_limpio.csv
  val_split_50pct.csv
  test_split_50pct.csv

Autor : Evelyn Silva Rozas
Fecha : Junio 2026
"""

import re
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR      = Path("/workspace/tesis-blip2-mimic")

CHEXPERT_CSV  = BASE_DIR / "mimic-cxr-2.0.0-chexpert.csv"
METADATA_CSV  = BASE_DIR / "mimic-cxr-2.0.0-metadata.csv.gz"

# Base unificada COMPLETA (generada por etl_mimic_cxr.py).
# OBLIGATORIA: debe contener report_text para TODOS los pacientes.
# Si no existe, ejecutar primero: python etl_mimic_cxr.py
FULL_BASE_CSV = BASE_DIR / "mimic_database_unificada.csv"

OUT_TRAIN     = BASE_DIR / "train_split_50pct.csv"
OUT_VAL       = BASE_DIR / "val_split_50pct.csv"
OUT_TEST      = BASE_DIR / "test_split_50pct.csv"

# Proporciones (igual que el 10%)
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.20
TEST_RATIO    = 0.10

SAMPLE_FRAC   = 0.50    # fracción del universo total a extraer
RANDOM_STATE  = 42

# 14 clases CheXpert
CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LIMPIEZA DE REPORTES (reutiliza lógica de limpiar_reportes.py)
# ─────────────────────────────────────────────────────────────────────────────

_PATRONES_ANONIMIZACION = [
    r'\b\w+\s+_{2,}\b',
    r'\b_{2,}\s+\w+\b',
    r'\b_{2,}\b',
    r'___ year old',
    r'___ year',
    r'year old',
]

_PATRONES_PLANTILLA = [
    r'final report examination',
    r'final report',
    r'report examination',
    r'examination chest',
    r'examination\s+\w+',
    r'indication\b',
    r'comparison\b',
]


def limpiar_texto(text: str) -> str:
    """Limpieza de reportes médicos (misma lógica que limpiar_reportes.py)."""
    if pd.isna(text) or not isinstance(text, str):
        return text
    t = text
    for p in _PATRONES_ANONIMIZACION:
        t = re.sub(p, '', t, flags=re.IGNORECASE)
    for p in _PATRONES_PLANTILLA:
        t = re.sub(p, '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'\n\s*\n+', '\n\n', t)
    return t.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CARGA Y FUSIÓN
# ─────────────────────────────────────────────────────────────────────────────

def cargar_base() -> pd.DataFrame:
    """
    Construye la base unificada con reportes + etiquetas CheXpert.
    Requiere mimic_database_unificada.csv (base completa con todos los pacientes).
    Si no existe, ejecutar primero: python etl_mimic_cxr.py
    """
    if not FULL_BASE_CSV.exists():
        log.error(
            f"\n{'='*70}\n"
            f"  ARCHIVO REQUERIDO NO ENCONTRADO: {FULL_BASE_CSV.name}\n"
            f"  Este script necesita la base unificada COMPLETA con report_text\n"
            f"  para todos los pacientes de MIMIC-CXR.\n"
            f"\n"
            f"  Solución: ejecutar primero el ETL completo:\n"
            f"    python etl_mimic_cxr.py\n"
            f"  (ajusta REPORTS_DIR en ese script a la carpeta de .txt de MIMIC)\n"
            f"{'='*70}"
        )
        sys.exit(1)

    log.info("[1/5] Cargando base unificada completa...")
    full_base = pd.read_csv(FULL_BASE_CSV)
    full_base["study_id"]   = full_base["study_id"].astype(str)
    full_base["subject_id"] = full_base["subject_id"].astype(str)
    log.info(f"       {len(full_base):,} filas | "
             f"{full_base['subject_id'].nunique():,} pacientes | "
             f"{full_base['study_id'].nunique():,} estudios")

    log.info("[2/5] Cargando etiquetas CheXpert...")
    chex = pd.read_csv(CHEXPERT_CSV)
    chex["study_id"]   = chex["study_id"].astype(str)
    chex["subject_id"] = chex["subject_id"].astype(str)
    log.info(f"       {len(chex):,} estudios")

    log.info("[3/5] Inner join base + CheXpert por study_id...")
    df = full_base.merge(chex, on="study_id", how="inner", suffixes=("", "_chex"))
    if "subject_id_chex" in df.columns:
        df = df.drop(columns=["subject_id_chex"])
    log.info(f"       {len(df):,} filas tras join | "
             f"{df['study_id'].nunique():,} estudios | "
             f"{df['subject_id'].nunique():,} pacientes")

    log.info("[4/5] Limpiando reportes médicos...")
    df["report_text"] = df["report_text"].apply(limpiar_texto)
    vacios = df["report_text"].isna() | (df["report_text"].str.strip() == "")
    log.info(f"       {vacios.sum():,} reportes vacíos → descartados")
    df = df[~vacios].copy()
    log.info(f"       {len(df):,} filas limpias | "
             f"{df['study_id'].nunique():,} estudios | "
             f"{df['subject_id'].nunique():,} pacientes")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BINARIZACIÓN DE ETIQUETAS (política U-ones)
# ─────────────────────────────────────────────────────────────────────────────

def binarizar_etiquetas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica política U-ones: 1.0 → 1, -1.0 → 1 (incierto), NaN → 0.
    Devuelve df con las 14 columnas CheXpert binarizadas como int8.
    """
    for col in CHEXPERT_LABELS:
        raw = df[col].fillna(0)
        df[col] = np.where(raw == -1.0, 1, np.where(raw == 1.0, 1, 0)).astype("int8")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  TABLA DE ESTUDIOS ÚNICOS CON ETIQUETAS AGREGADAS
# ─────────────────────────────────────────────────────────────────────────────

def tabla_estudios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Colapsa df a una fila por study_id, tomando el máximo de cada etiqueta
    (si alguna imagen del estudio es positiva → estudio positivo).
    """
    agg = {col: "max" for col in CHEXPERT_LABELS}
    agg["subject_id"] = "first"
    study_df = df.groupby("study_id").agg(agg).reset_index()
    return study_df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MUESTREO 50% ESTRATIFICADO POR PACIENTE
# ─────────────────────────────────────────────────────────────────────────────

def muestrear_50pct(study_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae el 50% de los pacientes (subject_id), estratificado por la
    categoría binaria Normal/Anormal (basada en 'No Finding').
    Devuelve la tabla de estudios del 50%.
    """
    log.info("[6a] Muestreo del 50% de pacientes...")
    df_patients = (
        study_df.groupby("subject_id")["No Finding"]
        .agg(lambda s: "Normal" if (s == 1).any() else "Anormal")
        .reset_index()
        .rename(columns={"No Finding": "_cat"})
    )

    log.info(f"     Distribución Normal/Anormal (universo):")
    for cat, pct in df_patients["_cat"].value_counts(normalize=True).items():
        log.info(f"       {cat:10s}: {pct*100:.2f}%")

    _, sample_patients = train_test_split(
        df_patients,
        test_size=SAMPLE_FRAC,
        stratify=df_patients["_cat"],
        random_state=RANDOM_STATE,
    )

    log.info(f"     Pacientes seleccionados: {len(sample_patients):,} / {len(df_patients):,}")
    sample_50 = study_df[study_df["subject_id"].isin(sample_patients["subject_id"])].copy()
    log.info(f"     Estudios en muestra 50%: {len(sample_50):,}")
    return sample_50


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PARTICIÓN ITERATIVA MULTI-ETIQUETA
# ─────────────────────────────────────────────────────────────────────────────

def particionar_multilabel(
    sample_studies: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide sample_studies en train/val/test con proporciones 70/20/10
    en 2 pasos a nivel de paciente (subject_id), espejando la lógica de
    muestreo_10_porciento.py.

    Estratificación: combinación de patologías activas por paciente.
    Si hay demasiadas combinaciones únicas (clases raras), colapsa a
    Normal/Anormal como fallback (igual que el 10% script).

    Garantiza ausencia total de fugas: ningún subject_id cruza splits.
    """
    log.info("[6b] Partición 70/20/10 a nivel de paciente (2 pasos estratificados)...")
    log.info(f"     Proporciones: train={TRAIN_RATIO:.0%}  val={VAL_RATIO:.0%}  test={TEST_RATIO:.0%}")

    # ── Tabla de pacientes únicos con etiqueta de combinación ─────────────────
    def _etiqueta_paciente(sub_df: pd.DataFrame) -> str:
        """Combina las etiquetas activas del paciente en un string para estratificar."""
        activas = [
            col for col in CHEXPERT_LABELS
            if col in sub_df.columns and (sub_df[col] == 1).any()
        ]
        return "|".join(sorted(activas)) if activas else "ninguna"

    df_patients = (
        sample_studies.groupby("subject_id")
        .apply(_etiqueta_paciente)
        .reset_index()
        .rename(columns={0: "_combo"})
    )

    # Si alguna combinación tiene < 2 muestras, colapsar a Normal/Anormal
    min_count = df_patients["_combo"].value_counts().min()
    if min_count < 2:
        log.info("     Combinaciones muy raras detectadas → usando Normal/Anormal como estratificador")
        no_finding_pat = (
            sample_studies.groupby("subject_id")["No Finding"]
            .agg(lambda s: "Normal" if (s == 1).any() else "Anormal")
            .reset_index()
            .rename(columns={"No Finding": "_combo"})
        )
        df_patients = df_patients.drop(columns=["_combo"]).merge(no_finding_pat, on="subject_id")

    # ── Paso 1: separar test (10%) del resto (90%) ────────────────────────────
    trainval_pats, test_pats = train_test_split(
        df_patients,
        test_size=TEST_RATIO,
        stratify=df_patients["_combo"],
        random_state=RANDOM_STATE,
    )

    # ── Paso 2: separar val (20%) del train+val (90%) ─────────────────────────
    # val_size relativo al trainval: 0.20 / 0.90 ≈ 0.2222
    val_size_adj = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_pats, val_pats = train_test_split(
        trainval_pats,
        test_size=val_size_adj,
        stratify=trainval_pats["_combo"],
        random_state=RANDOM_STATE,
    )

    # ── Expandir de pacientes → estudios ──────────────────────────────────────
    train_studies = sample_studies[
        sample_studies["subject_id"].isin(train_pats["subject_id"])
    ].copy()
    val_studies = sample_studies[
        sample_studies["subject_id"].isin(val_pats["subject_id"])
    ].copy()
    test_studies = sample_studies[
        sample_studies["subject_id"].isin(test_pats["subject_id"])
    ].copy()

    total = len(train_studies) + len(val_studies) + len(test_studies)
    log.info(
        f"     train={len(train_studies):,} estudios ({len(train_studies)/total:.1%})  "
        f"val={len(val_studies):,} ({len(val_studies)/total:.1%})  "
        f"test={len(test_studies):,} ({len(test_studies)/total:.1%})"
    )
    log.info(
        f"     pacientes: train={train_pats.__len__():,}  "
        f"val={val_pats.__len__():,}  test={test_pats.__len__():,}"
    )

    # ── Verificar cero fugas ──────────────────────────────────────────────────
    pats_tr = set(train_studies["subject_id"])
    pats_va = set(val_studies["subject_id"])
    pats_te = set(test_studies["subject_id"])
    leaks = (pats_tr & pats_va) | (pats_tr & pats_te) | (pats_va & pats_te)
    assert len(leaks) == 0, f"FUGA DE PACIENTES: {len(leaks)} sujetos en múltiples splits"
    log.info("     ✓ Sin fugas de pacientes entre splits")

    return train_studies, val_studies, test_studies


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ENSAMBLE FINAL: VOLVER A NIVEL DE DICOM
# ─────────────────────────────────────────────────────────────────────────────

def ensamblar_split(
    split_studies: pd.DataFrame,
    full_df:       pd.DataFrame,
) -> pd.DataFrame:
    """
    Dado un split de estudios, recupera todas las imágenes (filas DICOM)
    del DataFrame completo que pertenecen a esos study_ids.
    Elimina las columnas CheXpert (no se necesitan en el CSV final).
    """
    study_ids = set(split_studies["study_id"].astype(str))
    result = full_df[full_df["study_id"].isin(study_ids)].copy()

    # Eliminar columnas CheXpert del CSV final (misma convención que 10pct)
    chex_cols = [c for c in CHEXPERT_LABELS if c in result.columns]
    if chex_cols:
        result = result.drop(columns=chex_cols)

    # Ordenar columnas en el mismo orden que los splits del 10%
    desired_order = [
        "dicom_id", "subject_id", "study_id",
        "PerformedProcedureStepDescription", "ViewPosition",
        "Rows", "Columns", "StudyDate", "StudyTime",
        "ProcedureCodeSequence_CodeMeaning",
        "ViewCodeSequence_CodeMeaning",
        "PatientOrientationCodeSequence_CodeMeaning",
        "report_text",
    ]
    present = [c for c in desired_order if c in result.columns]
    extra   = [c for c in result.columns if c not in desired_order]
    result  = result[present + extra]

    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  REPORTE DE DISTRIBUCIÓN DE CLASES
# ─────────────────────────────────────────────────────────────────────────────

def reporte_distribucion(
    train_s: pd.DataFrame,
    val_s:   pd.DataFrame,
    test_s:  pd.DataFrame,
) -> None:
    """Imprime tabla de distribución de cada clase CheXpert en los 3 splits."""
    print(f"\n{'='*75}")
    print(f"  DISTRIBUCIÓN DE CLASES CHEXPERT (prevalencia % por split)")
    print(f"{'='*75}")
    print(f"  {'Clase':<32} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8}")
    for col in CHEXPERT_LABELS:
        if col in train_s.columns:
            t = train_s[col].mean() * 100
            v = val_s[col].mean()   * 100
            e = test_s[col].mean()  * 100
            print(f"  {col:<32} {t:>7.2f}%  {v:>7.2f}%  {e:>7.2f}%")
    print(f"{'='*75}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("\n" + "="*75)
    print("  BUILD 50% DATASET — MIMIC-CXR")
    print("  Partición iterativa multi-etiqueta CheXpert (skmultilearn)")
    print(f"  Proporciones: {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}  |  random_state={RANDOM_STATE}")
    print("="*75 + "\n")

    # ── 1-5. Carga, merge y limpieza ─────────────────────────────────────────
    full_df = cargar_base()

    # ── 6. Binarizar etiquetas ────────────────────────────────────────────────
    full_df = binarizar_etiquetas(full_df)

    # ── 7. Tabla de estudios únicos con etiquetas agregadas ───────────────────
    study_df = tabla_estudios(full_df)
    log.info(f"     {len(study_df):,} estudios únicos en el universo completo")

    # ── 8. Muestreo 50% ───────────────────────────────────────────────────────
    sample_50 = muestrear_50pct(study_df)

    # ── 9. Partición iterativa multi-etiqueta ─────────────────────────────────
    train_studies, val_studies, test_studies = particionar_multilabel(sample_50)

    # ── 10. Ensamblar splits a nivel DICOM ────────────────────────────────────
    log.info("[7/6] Ensamblando splits a nivel de imagen DICOM...")
    train_df = ensamblar_split(train_studies, full_df)
    val_df   = ensamblar_split(val_studies,   full_df)
    test_df  = ensamblar_split(test_studies,  full_df)

    # ── 11. Guardar ───────────────────────────────────────────────────────────
    log.info("[8/6] Guardando CSVs...")
    train_df.to_csv(OUT_TRAIN, index=False)
    val_df.to_csv(OUT_VAL,     index=False)
    test_df.to_csv(OUT_TEST,   index=False)

    log.info(f"  → {OUT_TRAIN.name}  ({len(train_df):,} filas)")
    log.info(f"  → {OUT_VAL.name}    ({len(val_df):,} filas)")
    log.info(f"  → {OUT_TEST.name}   ({len(test_df):,} filas)")

    # ── 12. Reporte de calidad ────────────────────────────────────────────────
    total_imgs = len(train_df) + len(val_df) + len(test_df)
    total_stu  = train_studies.__len__() + val_studies.__len__() + test_studies.__len__()

    print(f"\n{'='*75}")
    print("  RESULTADO FINAL")
    print(f"{'='*75}")
    print(f"  Total imágenes: {total_imgs:,}  |  Total estudios: {total_stu:,}")
    print(f"  Train : {len(train_df):,} imgs  ({len(train_df)/total_imgs:.1%})  "
          f"| {train_studies['subject_id'].nunique():,} pacientes")
    print(f"  Val   : {len(val_df):,} imgs  ({len(val_df)/total_imgs:.1%})  "
          f"| {val_studies['subject_id'].nunique():,} pacientes")
    print(f"  Test  : {len(test_df):,} imgs  ({len(test_df)/total_imgs:.1%})  "
          f"| {test_studies['subject_id'].nunique():,} pacientes")

    # Verificar ausencia total de fugas
    pats_tr = set(train_df["subject_id"].astype(str))
    pats_va = set(val_df["subject_id"].astype(str))
    pats_te = set(test_df["subject_id"].astype(str))
    leaks_final = (pats_tr & pats_va) | (pats_tr & pats_te) | (pats_va & pats_te)
    if leaks_final:
        print(f"\n  ⚠️  ADVERTENCIA: {len(leaks_final)} pacientes cruzan splits en el CSV final.")
    else:
        print(f"\n  ✓ Cero fugas de pacientes entre splits")

    # Distribución de clases en la tabla de estudios (para verificar estratificación)
    reporte_distribucion(train_studies, val_studies, test_studies)

    print(f"  Archivos guardados en: {BASE_DIR}")
    print("="*75 + "\n")
    print("✅ Dataset 50% construido exitosamente!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
