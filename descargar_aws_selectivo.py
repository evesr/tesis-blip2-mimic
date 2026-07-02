#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga SELECTIVA desde AWS S3 - Solo las imagenes del 10%
============================================================
Requisitos previos:
  - Credenciales AWS configuradas (una de las siguientes opciones):
      export AWS_ACCESS_KEY_ID=...
      export AWS_SECRET_ACCESS_KEY=...
    o archivo ~/.aws/credentials con perfil physionet
  - Acceso aprobado a MIMIC-CXR-JPG en PhysioNet

Uso:
    python descargar_aws_selectivo.py
    python descargar_aws_selectivo.py --workers 50
    python descargar_aws_selectivo.py --dry-run   # solo cuenta sin descargar
"""

import sys
import argparse
import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# ============================================================================
# CONFIGURACION
# ============================================================================

BASE_DIR   = Path("/workspace/tesis-blip2-mimic")
CSV_PATH   = BASE_DIR / "dataset_50pct_para_descarga.csv"
OUTPUT_DIR = BASE_DIR / "imagenes_50_porciento"

# ARN del Access Point otorgado por PhysioNet para MIMIC-CXR-JPG v2.1.0
S3_ACCESS_POINT = "arn:aws:s3:us-east-1:724665945834:accesspoint/mimic-cxr-jpg-v2-1-0-01"
BUCKET_PATH     = "mimic-cxr-jpg/2.1.0/files"

MAX_WORKERS_DEFAULT = 100  # Reducido para evitar throttling en el Access Point

# ============================================================================
# VALIDACION DE CREDENCIALES
# ============================================================================

def validar_credenciales() -> bool:
    """
    Verifica que las credenciales AWS esten configuradas usando STS.
    Detecta NoCredentials y AuthFailure antes de iniciar la descarga masiva.
    """
    print("Verificando credenciales AWS...")
    try:
        sts = boto3.client('sts', region_name='us-east-1')
        identity = sts.get_caller_identity()
        print(f"  Cuenta AWS : {identity['Account']}")
        print(f"  Identidad  : {identity['Arn'].split('/')[-1]}")
        return True
    except NoCredentialsError:
        print("\nERROR: No se encontraron credenciales AWS.")
        print("Configura las variables de entorno antes de ejecutar:")
        print("  export AWS_ACCESS_KEY_ID=tu_access_key")
        print("  export AWS_SECRET_ACCESS_KEY=tu_secret_key")
        print("O crea ~/.aws/credentials con tus credenciales de PhysioNet.")
        return False
    except ClientError as e:
        code = e.response['Error']['Code']
        if code in ('InvalidClientTokenId', 'AuthFailure', 'AccessDenied'):
            print(f"\nERROR de autenticacion AWS ({code}): {e}")
            return False
        # Otros errores de red: no son fatales en la validacion
        print(f"  Advertencia al verificar credenciales: {e} (continuando...)")
        return True
    except Exception as e:
        print(f"  Advertencia al verificar credenciales: {e} (continuando...)")
        return True


# ============================================================================
# DESCARGA DE UNA IMAGEN
# ============================================================================

def descargar_imagen(args):
    """Descarga UNA imagen especifica desde el S3 Access Point de PhysioNet."""
    subject_id, study_id, dicom_id, dest_path, s3_client = args

    # Saltar si ya existe y no esta vacia
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return "OK_EXISTE"

    try:
        prefix  = f"p{subject_id[:2]}"
        s3_key  = f"{BUCKET_PATH}/{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg"

        # Para S3 Access Points: Bucket = ARN, Key = ruta relativa
        response = s3_client.get_object(Bucket=S3_ACCESS_POINT, Key=s3_key)
        dest_path.write_bytes(response['Body'].read())
        return "OK"

    except ClientError as e:
        code = e.response['Error']['Code']
        if code == 'NoSuchKey':
            return "ERROR_404"
        if code in ('AccessDenied', 'Forbidden'):
            return "ERROR_403"
        return f"ERROR_{code}"
    except Exception as e:
        return f"ERROR_{type(e).__name__}"


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Descarga selectiva de imagenes MIMIC-CXR desde AWS S3"
    )
    parser.add_argument(
        '--workers', type=int, default=MAX_WORKERS_DEFAULT,
        help=f'Numero de hilos paralelos (default: {MAX_WORKERS_DEFAULT})'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Solo muestra cuantas imagenes se descargarian sin descargar nada'
    )
    parser.add_argument(
        '--csv', type=str, default=str(CSV_PATH),
        help=f'Ruta al CSV de imagenes (default: {CSV_PATH})'
    )
    parser.add_argument(
        '--output', type=str, default=str(OUTPUT_DIR),
        help=f'Directorio de salida (default: {OUTPUT_DIR})'
    )
    args = parser.parse_args()

    csv_path   = Path(args.csv)
    output_dir = Path(args.output)

    # Validar CSV
    if not csv_path.exists():
        print(f"ERROR: No se encontro el CSV: {csv_path}")
        sys.exit(1)

    # Validar credenciales antes de empezar
    if not validar_credenciales():
        sys.exit(1)

    # Crear cliente S3 (se crea una vez y se pasa a cada worker)
    s3_client = boto3.client('s3', region_name='us-east-1')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar CSV
    df = pd.read_csv(csv_path)

    # Contar imagenes ya existentes
    ya_existentes = sum(
        1 for dicom_id in df['dicom_id']
        if (output_dir / f"{dicom_id}.jpg").exists()
        and (output_dir / f"{dicom_id}.jpg").stat().st_size > 0
    )
    pendientes = len(df) - ya_existentes

    print("\n" + "="*70)
    print("DESCARGA SELECTIVA DESDE AWS S3 ACCESS POINT (PhysioNet)")
    print("="*70)
    print(f"CSV              : {csv_path}")
    print(f"Directorio salida: {output_dir}")
    print(f"Total imagenes   : {len(df):,}")
    print(f"Pacientes unicos : {df['subject_id'].nunique():,}")
    print(f"Estudios unicos  : {df['study_id'].nunique():,}")
    print(f"Workers          : {args.workers}")
    print(f"Ya descargadas   : {ya_existentes:,}")
    print(f"Pendientes       : {pendientes:,}")
    print("="*70)

    if args.dry_run:
        print("\nModo --dry-run: no se descarga nada.")
        sys.exit(0)

    if pendientes == 0:
        print("\nTodas las imagenes ya estan descargadas.")
        sys.exit(0)

    # Preparar tareas (incluye s3_client para que cada thread lo use)
    tareas = []
    for _, row in df.iterrows():
        subject_id = str(int(row["subject_id"]))
        study_id   = str(int(row["study_id"]))
        dicom_id   = row["dicom_id"]
        dest_path  = output_dir / f"{dicom_id}.jpg"
        tareas.append((subject_id, study_id, dicom_id, dest_path, s3_client))

    ok_nuevas    = 0
    ok_existentes = 0
    errores      = {}

    print(f"\nIniciando descarga de {pendientes:,} imagenes...\n")
    tiempo_inicio = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futuros = {executor.submit(descargar_imagen, t): t for t in tareas}

        for futuro in tqdm(as_completed(futuros), total=len(tareas), desc="Descargando"):
            resultado = futuro.result()

            if resultado == "OK":
                ok_nuevas += 1
            elif resultado == "OK_EXISTE":
                ok_existentes += 1
            else:
                errores[resultado] = errores.get(resultado, 0) + 1
                # Abortar si hay muchos 403 (credenciales invalidas o sin acceso)
                if errores.get("ERROR_403", 0) > 10:
                    print("\nERROR: Mas de 10 errores 403.")
                    print("Verifica tus credenciales AWS y que tengas acceso aprobado en PhysioNet.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    tiempo_total = time.time() - tiempo_inicio
    total_ok = ok_nuevas + ok_existentes

    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    print(f"Nuevas descargadas   : {ok_nuevas:,}")
    print(f"Ya existentes        : {ok_existentes:,}")
    if errores:
        print("\nErrores encontrados:")
        for error, count in sorted(errores.items()):
            print(f"  {error:35s} {count:,}")
    print(f"\nTOTAL EXITOSAS       : {total_ok:,} / {len(df):,}")
    print(f"Tiempo total         : {tiempo_total / 60:.1f} minutos")
    if ok_nuevas > 0:
        print(f"Velocidad            : {ok_nuevas / tiempo_total:.1f} imagenes/segundo")
    print("="*70)

    if total_ok == len(df):
        print(f"\nTODAS LAS {len(df):,} IMAGENES DESCARGADAS CORRECTAMENTE")
    else:
        faltantes = len(df) - total_ok
        print(f"\nFaltan {faltantes:,} imagenes. Re-ejecuta el script para reintentar.")


if __name__ == "__main__":
    main()
