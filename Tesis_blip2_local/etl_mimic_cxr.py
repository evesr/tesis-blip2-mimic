"""
ETL Script - MIMIC-CXR: Unificación de reportes de texto con metadatos.
========================================================================
Autor: Evelyn Silva Rozas
Fecha: Abril 2026

Este script:
1. Recorre recursivamente la carpeta de reportes (.txt) extrayendo study_id y texto.
2. Carga los metadatos desde mimic-cxr-2.0.0-metadata.csv.gz.
3. Realiza un INNER JOIN por study_id.
4. Elimina filas con report_text nulo.
5. Guarda el resultado en mimic_database_unificada.csv.
"""

import re
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ============================================================================
# CONFIGURACIÓN DE RUTAS (modificar según tu estructura local)
# ============================================================================

# Carpeta raíz que contiene las subcarpetas con los archivos s[study_id].txt
REPORTS_DIR = Path("/Users/evelynsilvarozas/Tesis_blip2_local/files")

# Archivo comprimido de metadatos
METADATA_PATH = Path("/Users/evelynsilvarozas/Tesis_blip2_local/mimic-cxr-2.0.0-metadata.csv.gz")

# Ruta de salida
OUTPUT_PATH = Path("/Users/evelynsilvarozas/Tesis_blip2_local/mimic_database_unificada.csv")

# Patrón regex para extraer study_id del nombre de archivo (ej: s50414267.txt -> 50414267)
STUDY_ID_PATTERN = re.compile(r"^s(\d+)\.txt$")


# ============================================================================
# FUNCIONES
# ============================================================================

def extract_reports(reports_dir: Path) -> pd.DataFrame:
    """
    Recorre recursivamente `reports_dir` buscando archivos .txt,
    extrae el study_id del nombre y lee el contenido del reporte.

    Returns
    -------
    pd.DataFrame con columnas ['study_id', 'report_text']
    """
    # Recolectar todos los .txt primero para poder mostrar barra de progreso
    txt_files = list(reports_dir.rglob("*.txt"))
    print(f"[INFO] Se encontraron {len(txt_files):,} archivos .txt en {reports_dir}")

    records: list[dict] = []
    skipped = 0

    for filepath in tqdm(txt_files, desc="Leyendo reportes", unit="archivo"):
        # Extraer study_id del nombre del archivo
        match = STUDY_ID_PATTERN.match(filepath.name)
        if not match:
            continue  # no cumple el patrón esperado, saltar

        study_id = int(match.group(1))

        # Leer contenido con manejo de errores de codificación
        try:
            text = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = filepath.read_text(encoding="latin-1")
                warnings.warn(f"Archivo leído con latin-1 (fallback): {filepath}")
            except Exception as e:
                warnings.warn(f"No se pudo leer {filepath}: {e}")
                skipped += 1
                continue
        except Exception as e:
            warnings.warn(f"Error inesperado leyendo {filepath}: {e}")
            skipped += 1
            continue

        records.append({"study_id": study_id, "report_text": text.strip()})

    if skipped:
        print(f"[WARN] Se omitieron {skipped} archivos por errores de lectura.")

    df = pd.DataFrame(records)
    print(f"[INFO] Reportes extraídos: {len(df):,}")
    return df


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Carga el CSV comprimido de metadatos."""
    print(f"[INFO] Cargando metadatos desde {metadata_path} ...")
    df = pd.read_csv(metadata_path)
    print(f"[INFO] Metadatos cargados: {len(df):,} filas, columnas: {list(df.columns)}")
    return df


def run_etl() -> None:
    """Pipeline ETL completo."""
    # --- Validaciones iniciales ---
    if not REPORTS_DIR.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de reportes: {REPORTS_DIR}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de metadatos: {METADATA_PATH}")

    # 1. Extraer reportes
    df_reports = extract_reports(REPORTS_DIR)

    # 2. Cargar metadatos
    df_metadata = load_metadata(METADATA_PATH)

    # Asegurar tipo consistente para el join
    df_reports["study_id"] = df_reports["study_id"].astype(int)
    df_metadata["study_id"] = df_metadata["study_id"].astype(int)

    # 3. INNER JOIN
    print("[INFO] Realizando INNER JOIN por study_id ...")
    df_merged = df_metadata.merge(df_reports, on="study_id", how="inner")
    print(f"[INFO] Filas después del JOIN: {len(df_merged):,}")

    # 4. Eliminar filas con report_text nulo o vacío
    before = len(df_merged)
    df_merged = df_merged.dropna(subset=["report_text"])
    df_merged = df_merged[df_merged["report_text"].str.strip() != ""]
    after = len(df_merged)
    if before != after:
        print(f"[INFO] Se eliminaron {before - after} filas con report_text vacío/nulo.")

    # 5. Guardar resultado
    df_merged.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Archivo guardado en: {OUTPUT_PATH}  ({len(df_merged):,} filas)")


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    run_etl()
