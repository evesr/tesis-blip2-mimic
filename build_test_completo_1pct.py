#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_test_completo_1pct.py
===========================
Construye un Test Set 1% COMPLETO de 397 muestras con imagen garantizada, para
evaluar los modelos SEMIFINAL (phase1_semifinal_results: Config_4/5/6) sobre el
banco de imagenes disponible.

Contexto
--------
El split original `test_split_limpio.csv` tiene 397 dicoms, pero solo 181 de
esas imagenes existen fisicamente en el banco `imagenes_50_porciento`
(el directorio `imagenes_1_porciento` no esta disponible). Los 216 dicoms
restantes NO tienen imagen.

Solucion (misma metodologia que build_test_completo_10pct.py)
-------------------------------------------------------------
  • Conservar las 181 filas cuyas imagenes SI estan presentes.
  • Rellenar las 216 faltantes con imagenes ALEATORIAS del banco disponible,
    RESPETANDO LA ESTRATIFICACION DE PATOLOGIAS (perfil CheXpert de 14 clases).
    Cada sustituto trae SU PROPIA imagen, reporte y study_id -> tripleta
    (imagen, reporte, etiquetas) internamente consistente.

Anti-leakage
------------
Los sustitutos se toman del banco EXCLUYENDO dicoms/subjects de
train_split_limpio / val_split_limpio / test_split_limpio (los pacientes del
experimento 1% con que se entrenaron/validaron los modelos SEMIFINAL).

Salida
------
  • test_split_completo_1pct.csv        (397 filas, MISMO esquema de 13 cols)
  • test_split_completo_1pct_META.csv   (auditoria: origen real/sustituto)

Autor : Evelyn Silva Rozas
Fecha : Julio 2026
"""

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Reutilizar helpers YA VALIDADOS del builder 10% (indexado de imagenes,
# perfiles CheXpert, categoria Normal/Anormal, etiquetas).
from build_test_completo_10pct import (
    CHEX_LABELS,
    NOFINDING_IDX,
    indexar_imagenes,
    cargar_perfiles_chexpert,
    perfil_de,
    categoria_de,
)

# ─────────────────────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(os.path.dirname(os.path.abspath(__file__)))

TEST_1PCT    = BASE / "test_split_limpio.csv"
TRAIN_1PCT   = BASE / "train_split_limpio.csv"
VAL_1PCT     = BASE / "val_split_limpio.csv"
BANK_META    = BASE / "dataset_50pct_para_descarga.csv"     # metadata del banco
BANK_IMAGES  = BASE / "imagenes_50_porciento"               # banco de imagenes
CHEXPERT_CSV = BASE / "mimic-cxr-2.0.0-chexpert.csv"        # etiquetas GT

OUT_CSV  = BASE / "test_split_completo_1pct.csv"
OUT_META = BASE / "test_split_completo_1pct_META.csv"

SEED = 42
SEP = "=" * 78


def main() -> int:
    print(SEP)
    print(" CONSTRUCCION TEST SET 1% COMPLETO (397) — imagen garantizada")
    print(SEP)

    # 1) Indexar banco de imagenes ------------------------------------------------
    print(f"\n[1/6] Indexando banco de imagenes: {BANK_IMAGES.name} ...")
    if not BANK_IMAGES.exists():
        print(f"  ERROR: no existe {BANK_IMAGES}")
        return 1
    stems = indexar_imagenes(BANK_IMAGES)
    print(f"      -> {len(stems):,} imagenes presentes en el banco")

    # 2) Cargar test 1% + perfiles CheXpert --------------------------------------
    print("\n[2/6] Cargando test 1% y perfiles CheXpert ...")
    test = pd.read_csv(TEST_1PCT)
    orig_cols = list(test.columns)
    test["dicom_id"] = test["dicom_id"].astype(str)
    perfiles = cargar_perfiles_chexpert(CHEXPERT_CSV)
    print(f"      -> test 1%: {len(test):,} filas | {len(perfiles):,} study_id con etiquetas")

    presentes_mask = test["dicom_id"].isin(stems)
    df_presentes = test[presentes_mask].copy()
    df_faltantes = test[~presentes_mask].copy()
    n_pres, n_falt = len(df_presentes), len(df_faltantes)
    print(f"      -> presentes: {n_pres:,} | faltantes: {n_falt:,} (a sustituir)")

    df_faltantes["_perfil"] = df_faltantes["study_id"].map(lambda s: perfil_de(s, perfiles))
    df_faltantes["_cat"]    = df_faltantes["_perfil"].map(categoria_de)

    print("      Distribucion Normal/Anormal de las faltantes:")
    for cat, n in df_faltantes["_cat"].value_counts().items():
        print(f"        {cat:8s}: {n:5d} ({n/max(n_falt,1)*100:5.1f}%)")

    # 3) Construir pool de candidatos del banco (anti-leakage) -------------------
    print("\n[3/6] Construyendo pool de candidatos (anti-leakage 1%) ...")
    train = pd.read_csv(TRAIN_1PCT, usecols=["dicom_id", "subject_id"])
    val   = pd.read_csv(VAL_1PCT,   usecols=["dicom_id", "subject_id"])

    excl_dicoms   = set(test["dicom_id"]) \
                  | set(train["dicom_id"].astype(str)) \
                  | set(val["dicom_id"].astype(str))
    excl_subjects = set(train["subject_id"].astype(str)) \
                  | set(val["subject_id"].astype(str))
    print(f"      -> excluir {len(excl_dicoms):,} dicoms y {len(excl_subjects):,} subjects (train/val/test 1%)")

    bank = pd.read_csv(BANK_META)
    bank["dicom_id"]   = bank["dicom_id"].astype(str)
    bank["subject_id"] = bank["subject_id"].astype(str)

    cand = bank[
        bank["dicom_id"].isin(stems)                 # imagen presente
        & ~bank["dicom_id"].isin(excl_dicoms)        # no duplicar / no train-val-test
        & ~bank["subject_id"].isin(excl_subjects)    # no pacientes de train/val 1%
        & bank["report_text"].notna()                # reporte valido
    ].copy()
    cand = cand[cand["report_text"].astype(str).str.strip() != ""]
    cand = cand.drop_duplicates(subset=["dicom_id"]).reset_index(drop=True)
    print(f"      -> candidatos disponibles: {len(cand):,}")

    if len(cand) < n_falt:
        print(f"  ERROR: candidatos ({len(cand)}) < faltantes ({n_falt}).")
        return 1

    cand["_perfil"] = cand["study_id"].map(lambda s: perfil_de(s, perfiles))
    cand["_cat"]    = cand["_perfil"].map(categoria_de)

    # 4) Muestreo estratificado por perfil de patologias -------------------------
    print("\n[4/6] Muestreo estratificado por perfil CheXpert ...")
    cand = cand.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    cand_by_key = defaultdict(list)
    cand_by_cat = defaultdict(list)
    for i in range(len(cand)):
        cand_by_key[cand.at[i, "_perfil"]].append(i)
        cand_by_cat[cand.at[i, "_cat"]].append(i)

    used = set()
    chosen = []

    # (a) match exacto de perfil
    need_by_key = Counter(df_faltantes["_perfil"])
    deficit_by_cat = Counter()
    for key, need in need_by_key.items():
        pool = [i for i in cand_by_key.get(key, []) if i not in used]
        take = pool[:need]
        for i in take:
            used.add(i)
        chosen.extend(take)
        short = need - len(take)
        if short > 0:
            cat = "Normal" if key[NOFINDING_IDX] == 1 else "Anormal"
            deficit_by_cat[cat] += short

    n_exact = len(chosen)

    # (b) fallback por categoria Normal/Anormal
    for cat, need in list(deficit_by_cat.items()):
        if need <= 0:
            continue
        pool = [i for i in cand_by_cat.get(cat, []) if i not in used]
        take = pool[:need]
        for i in take:
            used.add(i)
        chosen.extend(take)

    n_cat = len(chosen) - n_exact

    # (c) fallback aleatorio puro
    faltan_aun = n_falt - len(chosen)
    if faltan_aun > 0:
        pool = [i for i in range(len(cand)) if i not in used]
        take = pool[:faltan_aun]
        for i in take:
            used.add(i)
        chosen.extend(take)

    n_rand = len(chosen) - n_exact - n_cat
    assert len(chosen) == n_falt, f"Sustitutos={len(chosen)} != faltantes={n_falt}"
    print(f"      -> match exacto perfil : {n_exact:5d}")
    print(f"      -> fallback categoria  : {n_cat:5d}")
    print(f"      -> fallback aleatorio  : {n_rand:5d}")
    print(f"      -> TOTAL sustitutos    : {len(chosen):5d}")

    df_sustitutos = cand.iloc[chosen].copy().reset_index(drop=True)

    # 5) Ensamblar test completo (397) -------------------------------------------
    print("\n[5/6] Ensamblando test completo (397) ...")
    pres_out = df_presentes[orig_cols].copy()
    sust_out = df_sustitutos[orig_cols].copy()

    completo = pd.concat([pres_out, sust_out], ignore_index=True)
    completo["dicom_id"] = completo["dicom_id"].astype(str)
    completo = completo.drop_duplicates(subset=["dicom_id"]).reset_index(drop=True)

    assert len(completo) == len(test), f"Completo={len(completo)} != esperado={len(test)}"
    faltan_img = [d for d in completo["dicom_id"] if d not in stems]
    assert not faltan_img, f"{len(faltan_img)} dicoms SIN imagen en el resultado"

    completo.to_csv(OUT_CSV, index=False)
    print(f"      -> guardado: {OUT_CSV.name}  ({len(completo):,} filas, {len(completo.columns)} cols)")
    print(f"      -> TODAS las {len(completo):,} filas tienen imagen presente ✔")

    # 6) Auditoria + verificacion de estratificacion -----------------------------
    print("\n[6/6] Auditoria de estratificacion ...")
    meta = completo[["dicom_id", "study_id"]].copy()
    meta["origen"] = np.where(meta["dicom_id"].isin(set(df_presentes["dicom_id"])),
                              "real", "sustituto")
    meta["_perfil"] = meta["study_id"].map(lambda s: perfil_de(s, perfiles))
    meta["categoria"] = meta["_perfil"].map(categoria_de)
    meta["n_patologias"] = meta["_perfil"].map(lambda t: int(sum(t)) - (1 if t[NOFINDING_IDX] == 1 else 0))
    meta.drop(columns=["_perfil"]).to_csv(OUT_META, index=False)

    def _prev(df_sids):
        m = np.zeros(14, dtype=float)
        for s in df_sids:
            p = perfil_de(s, perfiles)
            m += np.array(p, dtype=float)
        return m / max(len(df_sids), 1)

    prev_orig = _prev(test["study_id"])
    prev_comp = _prev(completo["study_id"])
    print(f"      {'Patologia':30s} {'orig%':>7s} {'compl%':>7s} {'dif':>7s}")
    for j, lbl in enumerate(CHEX_LABELS):
        print(f"      {lbl:30s} {prev_orig[j]*100:6.1f}% {prev_comp[j]*100:6.1f}% "
              f"{(prev_comp[j]-prev_orig[j])*100:+6.1f}")

    print(f"\n      Normal/Anormal (completo):")
    for cat, n in meta["categoria"].value_counts().items():
        print(f"        {cat:8s}: {n:5d} ({n/len(meta)*100:5.1f}%)")

    print("\n" + SEP)
    print(" RESULTADO")
    print(SEP)
    print(f"  • {OUT_CSV}")
    print(f"      {len(completo):,} filas | {n_pres:,} reales + {len(chosen):,} sustitutos")
    print(f"  • {OUT_META}")
    print(SEP + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
