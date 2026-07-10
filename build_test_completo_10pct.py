#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_test_completo_10pct.py
============================
Construye un Test Set 10% COMPLETO de 3796 muestras con imagen garantizada,
para poder ejecutar inferencia con los dos mejores modelos de FASE 1
(phase1_final_results) sobre el banco de imagenes disponible.

Contexto
--------
El split original `test_split_10pct_limpio.csv` tiene 3796 dicoms, pero solo
1930 de esas imagenes existen fisicamente en el banco `imagenes_50_porciento`
(el directorio `imagenes_10_porciento` no esta disponible). Los 1866 dicoms
restantes NO tienen imagen.

Solucion (segun instruccion del usuario)
----------------------------------------
  • Conservar las 1930 filas cuyas imagenes SI estan presentes (imagen +
    reporte + etiquetas 100% consistentes).
  • Rellenar las 1866 faltantes con imagenes ALEATORIAS del banco disponible,
    RESPETANDO LA ESTRATIFICACION DE PATOLOGIAS: cada sustituto se elige de modo
    que su perfil CheXpert (vector de 14 patologias) reproduzca la distribucion
    de perfiles de las filas faltantes. Cada sustituto trae SU PROPIA imagen,
    reporte y study_id -> la tripleta (imagen, reporte, etiquetas) queda
    internamente consistente y las metricas son validas fila por fila.

Anti-leakage
------------
Los sustitutos se toman del banco EXCLUYENDO:
  • dicoms ya presentes en el test 10% (evita duplicados),
  • dicoms y subjects de train_split_10pct / val_split_10pct
    (los pacientes con que se entrenaron/validaron los modelos de FASE 1).

Salida
------
  • test_split_10pct_completo.csv        (3796 filas, MISMO esquema de 13 cols)
  • test_split_10pct_completo_META.csv   (auditoria: origen real/sustituto,
                                          categoria Normal/Anormal, perfil)

Autor : Evelyn Silva Rozas
Fecha : Julio 2026
"""

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(os.path.dirname(os.path.abspath(__file__)))

TEST_10PCT   = BASE / "test_split_10pct_limpio.csv"
TRAIN_10PCT  = BASE / "train_split_10pct_limpio.csv"
VAL_10PCT    = BASE / "val_split_10pct_limpio.csv"
BANK_META    = BASE / "dataset_50pct_para_descarga.csv"     # metadata del banco
BANK_IMAGES  = BASE / "imagenes_50_porciento"               # banco de imagenes
CHEXPERT_CSV = BASE / "mimic-cxr-2.0.0-chexpert.csv"        # etiquetas GT

OUT_CSV  = BASE / "test_split_10pct_completo.csv"
OUT_META = BASE / "test_split_10pct_completo_META.csv"

SEED = 42

# 14 patologias oficiales CheXpert (orden canonico).
CHEX_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
NOFINDING_IDX = CHEX_LABELS.index("No Finding")

SEP = "=" * 78


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def _norm_sid(s) -> str:
    """Normaliza study_id: '50414267' o 's50414267' -> '50414267'."""
    s = str(s).strip()
    return s.lstrip("s") if s.startswith("s") else s


def indexar_imagenes(root: Path) -> set:
    """
    Devuelve el conjunto de 'stems' (dicom_id) presentes en el banco de imagenes.
    Usa os.scandir recursivo (rapido) y soporta .jpg/.jpeg/.png.
    """
    exts = {".jpg", ".jpeg", ".png"}
    stems = set()
    pendientes = [str(root)]
    while pendientes:
        d = pendientes.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        pendientes.append(entry.path)
                    else:
                        name = entry.name
                        dot = name.rfind(".")
                        if dot != -1 and name[dot:].lower() in exts:
                            stems.add(name[:dot])
        except (PermissionError, FileNotFoundError):
            continue
    return stems


def cargar_perfiles_chexpert(path: Path) -> dict:
    """
    Devuelve dict: sid_normalizado -> tupla de 14 ints (1 si patologia == 1.0).
    """
    # auto-fallback .gz
    if not path.exists():
        gz = Path(str(path) + ".gz")
        if gz.exists():
            path = gz
    df = pd.read_csv(path, compression="infer")
    cols_presentes = [c for c in CHEX_LABELS if c in df.columns]
    if len(cols_presentes) != 14:
        print(f"[WARN] CheXpert CSV: solo {len(cols_presentes)}/14 columnas encontradas.")
    sids = df["study_id"].astype(str).map(_norm_sid).values
    # matriz 1.0 -> 1 else 0
    mat = np.zeros((len(df), 14), dtype=np.int8)
    for j, lbl in enumerate(CHEX_LABELS):
        if lbl in df.columns:
            mat[:, j] = (pd.to_numeric(df[lbl], errors="coerce").fillna(0.0).values == 1.0).astype(np.int8)
    perfiles = {}
    for i, sid in enumerate(sids):
        perfiles[sid] = tuple(int(x) for x in mat[i])
    return perfiles


def perfil_de(sid, perfiles: dict) -> tuple:
    """Perfil de 14 dims para un study_id (todo-cero si no esta en CheXpert)."""
    return perfiles.get(_norm_sid(sid), tuple([0] * 14))


def categoria_de(perfil: tuple) -> str:
    """Normal si 'No Finding' == 1, si no Anormal."""
    return "Normal" if perfil[NOFINDING_IDX] == 1 else "Anormal"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print(SEP)
    print(" CONSTRUCCION TEST SET 10% COMPLETO (3796) — imagen garantizada")
    print(SEP)

    # 1) Indexar banco de imagenes ------------------------------------------------
    print(f"\n[1/6] Indexando banco de imagenes: {BANK_IMAGES.name} ...")
    if not BANK_IMAGES.exists():
        print(f"  ERROR: no existe {BANK_IMAGES}")
        return 1
    stems = indexar_imagenes(BANK_IMAGES)
    print(f"      -> {len(stems):,} imagenes presentes en el banco")

    # 2) Cargar test 10% + perfiles CheXpert -------------------------------------
    print("\n[2/6] Cargando test 10% y perfiles CheXpert ...")
    test = pd.read_csv(TEST_10PCT)
    orig_cols = list(test.columns)
    test["dicom_id"] = test["dicom_id"].astype(str)
    perfiles = cargar_perfiles_chexpert(CHEXPERT_CSV)
    print(f"      -> test 10%: {len(test):,} filas | {len(perfiles):,} study_id con etiquetas")

    presentes_mask = test["dicom_id"].isin(stems)
    df_presentes = test[presentes_mask].copy()
    df_faltantes = test[~presentes_mask].copy()
    n_pres, n_falt = len(df_presentes), len(df_faltantes)
    print(f"      -> presentes: {n_pres:,} | faltantes: {n_falt:,} (a sustituir)")

    # Perfil + categoria de las filas faltantes (objetivo de estratificacion)
    df_faltantes["_perfil"] = df_faltantes["study_id"].map(lambda s: perfil_de(s, perfiles))
    df_faltantes["_cat"]    = df_faltantes["_perfil"].map(categoria_de)

    print("      Distribucion Normal/Anormal de las faltantes:")
    for cat, n in df_faltantes["_cat"].value_counts().items():
        print(f"        {cat:8s}: {n:5d} ({n/n_falt*100:5.1f}%)")

    # 3) Construir pool de candidatos del banco ----------------------------------
    print("\n[3/6] Construyendo pool de candidatos (anti-leakage) ...")
    train = pd.read_csv(TRAIN_10PCT, usecols=["dicom_id", "subject_id"])
    val   = pd.read_csv(VAL_10PCT,   usecols=["dicom_id", "subject_id"])

    excl_dicoms   = set(test["dicom_id"]) \
                  | set(train["dicom_id"].astype(str)) \
                  | set(val["dicom_id"].astype(str))
    excl_subjects = set(train["subject_id"].astype(str)) \
                  | set(val["subject_id"].astype(str))
    print(f"      -> excluir {len(excl_dicoms):,} dicoms y {len(excl_subjects):,} subjects (train/val/test 10%)")

    bank = pd.read_csv(BANK_META)
    bank["dicom_id"]   = bank["dicom_id"].astype(str)
    bank["subject_id"] = bank["subject_id"].astype(str)

    cand = bank[
        bank["dicom_id"].isin(stems)                 # imagen presente
        & ~bank["dicom_id"].isin(excl_dicoms)        # no duplicar / no train-val-test
        & ~bank["subject_id"].isin(excl_subjects)    # no pacientes de train/val
        & bank["report_text"].notna()                # reporte valido
    ].copy()
    cand = cand[cand["report_text"].astype(str).str.strip() != ""]
    cand = cand.drop_duplicates(subset=["dicom_id"]).reset_index(drop=True)
    print(f"      -> candidatos disponibles: {len(cand):,}")

    if len(cand) < n_falt:
        print(f"  ERROR: candidatos ({len(cand)}) < faltantes ({n_falt}).")
        return 1

    # Perfil + categoria de cada candidato
    cand["_perfil"] = cand["study_id"].map(lambda s: perfil_de(s, perfiles))
    cand["_cat"]    = cand["_perfil"].map(categoria_de)

    # 4) Muestreo estratificado por perfil de patologias -------------------------
    print("\n[4/6] Muestreo estratificado por perfil CheXpert ...")
    # Barajar candidatos de forma reproducible
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

    # (c) fallback aleatorio puro (por si algo quedo corto)
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

    # 5) Ensamblar test completo (3796) ------------------------------------------
    print("\n[5/6] Ensamblando test completo (3796) ...")
    pres_out = df_presentes[orig_cols].copy()
    sust_out = df_sustitutos[orig_cols].copy()

    completo = pd.concat([pres_out, sust_out], ignore_index=True)
    completo["dicom_id"] = completo["dicom_id"].astype(str)
    completo = completo.drop_duplicates(subset=["dicom_id"]).reset_index(drop=True)

    # Verificaciones duras
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

    # Comparar prevalencia por patologia: test ORIGINAL vs test COMPLETO
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
    cats = meta["categoria"].value_counts()
    for cat, n in cats.items():
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
