import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
# ============================================================================
# CONFIGURACIÓN
# ============================================================================
MI_COOKIE = "_gid=GA1.2.1613089432.1776267253; csrftoken=eFTX9dYuzX9EJLq4p2utAQsdAsBYmVx0; sessionid=ybj1zzssmkzpxa2f3cmp1swbcoozow6v; _ga_YKC8ZQQ4FF=GS2.1.s1776481188$o9$g1$t1776481189$j59$l0$h0; _ga=GA1.1.1237063727.1775746742"

URL_BASE = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files"
BASE_DIR = Path("/Users/evelynsilvarozas/Tesis_blip2_local")
CSV_PATH = BASE_DIR / "dataset_1_porciento_final.csv"
OUTPUT_DIR = BASE_DIR / "imagenes_1_porciento"
# ============================================================================

def descargar_imagen(args):
    url, dest_path, session = args
    if dest_path.exists():
        return "OK_EXISTE"
    
    try:
        # Timeout ajustado a 15 para evitar cortes abruptos
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                f.write(r.content)
            return "OK"
        elif r.status_code == 403:
            return "ERROR_403"
        else:
            return f"ERROR_{r.status_code}"
    except Exception:
        return "ERROR_RED"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    
    # Ajustamos el pool de conexiones a 10
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=3)
    session.mount('https://', adapter)
    session.headers.update({
        "Cookie": MI_COOKIE,
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    })

    print(f"[INFO] Iniciando descarga EQUILIBRADA (10 workers) de {len(df)} imágenes...")

    tareas = []
    for _, row in df.iterrows():
        subject_id = str(int(row["subject_id"]))
        study_id = str(int(row["study_id"]))
        dicom_id = row["dicom_id"]
        
        dest_path = OUTPUT_DIR / f"{dicom_id}.jpg"
        prefix = f"p{subject_id[:2]}"
        url = f"{URL_BASE}/{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
        tareas.append((url, dest_path, session))

    # EXACTAMENTE 10 TRABAJADORES
    MAX_WORKERS = 20
    errores_403 = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futuros = {executor.submit(descargar_imagen, t): t for t in tareas}
        
        for futuro in tqdm(as_completed(futuros), total=len(tareas), desc="Descargando"):
            resultado = futuro.result()
            if resultado == "ERROR_403":
                errores_403 += 1
                if errores_403 > 5:
                    print("\n[ALERTA] Demasiados 403. Cookie expirada o IP bloqueada temporalmente.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    return

if __name__ == "__main__":
    main()