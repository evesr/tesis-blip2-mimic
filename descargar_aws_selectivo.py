#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga SELECTIVA desde AWS S3 - Solo las 37,608 imágenes del 10%
====================================================================
"""

import pandas as pd
import boto3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Configuración
BASE_DIR = Path("/workspace/tesis-blip2-mimic/Tesis_blip2_local")
CSV_PATH = BASE_DIR / "dataset_10_porciento_final.csv"
OUTPUT_DIR = BASE_DIR / "imagenes_10_porciento"

# AWS S3 con Access Point de PhysioNet
S3_BASE = "arn:aws:s3:us-east-1:724665945834:accesspoint/mimic-cxr-jpg-v2-1-0-01"
BUCKET_PATH = "mimic-cxr-jpg/2.1.0/files"

MAX_WORKERS = 100  # Alto paralelismo

# Cliente S3
s3_client = boto3.client('s3', region_name='us-east-1')

def descargar_imagen(args):
    """Descarga UNA imagen específica desde S3."""
    subject_id, study_id, dicom_id, dest_path = args
    
    if dest_path.exists():
        return "OK_EXISTE"
    
    try:
        prefix = f"p{subject_id[:2]}"
        s3_key = f"{BUCKET_PATH}/{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
        
        # Usar el Access Point ARN
        s3_url = f"{S3_BASE}/{s3_key}"
        
        # Descargar con download_file requiere bucket name, no ARN
        # Usamos get_object en su lugar
        response = s3_client.get_object(Bucket=S3_BASE, Key=s3_key)
        dest_path.write_bytes(response['Body'].read())
        
        return "OK"
        
    except s3_client.exceptions.NoSuchKey:
        return "ERROR_404"
    except Exception as e:
        if "403" in str(e) or "Forbidden" in str(e):
            return "ERROR_403"
        return f"ERROR_{type(e).__name__}"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    
    print("="*80)
    print("DESCARGA SELECTIVA DESDE AWS S3 ACCESS POINT")
    print("="*80)
    print(f"\nTotal de imágenes: {len(df):,}")
    print(f"Pacientes únicos: {df['subject_id'].nunique():,}")
    print(f"Estudios únicos: {df['study_id'].nunique():,}")
    
    # Verificar distribución
    df['prefix'] = df['subject_id'].astype(str).str[:2]
    print(f"\nPrefijos presentes: {sorted(df['prefix'].unique())}")
    print(f"Workers: {MAX_WORKERS}")
    print("="*80)
    
    # Preparar tareas
    tareas = []
    for _, row in df.iterrows():
        subject_id = str(int(row["subject_id"]))
        study_id = str(int(row["study_id"]))
        dicom_id = row["dicom_id"]
        dest_path = OUTPUT_DIR / f"{dicom_id}.jpg"
        tareas.append((subject_id, study_id, dicom_id, dest_path))
    
    ok_nuevas = 0
    ok_existentes = 0
    errores = {}
    
    print(f"\nIniciando descarga...\n")
    tiempo_inicio = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futuros = {executor.submit(descargar_imagen, t): t for t in tareas}
        
        for futuro in tqdm(as_completed(futuros), total=len(tareas), desc="Descargando"):
            resultado = futuro.result()
            
            if resultado == "OK":
                ok_nuevas += 1
            elif resultado == "OK_EXISTE":
                ok_existentes += 1
            else:
                errores[resultado] = errores.get(resultado, 0) + 1
                if "ERROR_403" in resultado and errores.get("ERROR_403", 0) > 10:
                    print("\n❌ >10 errores 403. Verifica credenciales AWS.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
    
    tiempo_total = time.time() - tiempo_inicio
    
    # Resumen
    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)
    print(f"Nuevas descargadas:    {ok_nuevas:,}")
    print(f"Ya existentes:         {ok_existentes:,}")
    for error, count in sorted(errores.items()):
        print(f"{error:20s}  {count:,}")
    print(f"\nTOTAL EXITOSAS:        {ok_nuevas + ok_existentes:,} / {len(df):,}")
    print(f"Tiempo total:          {tiempo_total/60:.1f} minutos")
    
    if ok_nuevas > 0:
        velocidad = ok_nuevas / tiempo_total
        print(f"Velocidad:             {velocidad:.1f} imágenes/segundo")
    
    print("="*80)
    
    if ok_nuevas + ok_existentes == len(df):
        print("\n✅ ¡TODAS LAS 37,608 IMÁGENES DESCARGADAS!")
    else:
        print(f"\n⚠️  Faltan {len(df) - ok_nuevas - ok_existentes:,} imágenes")

if __name__ == "__main__":
    main()
