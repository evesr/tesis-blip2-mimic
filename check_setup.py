#!/usr/bin/env python3
"""
Script para validar la configuración del entorno y datos.

Verifica que todo esté listo antes de entrenar:
- Dependencias instaladas
- Archivos CSV presentes
- Imágenes accesibles
- GPU disponible (si aplica)

Usage:
    python check_setup.py

Author: [Tu nombre]
Date: 2026-04-22
"""

import sys
from pathlib import Path
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Verifica que las dependencias estén instaladas."""
    print("\n🔍 Verificando dependencias...")
    
    required = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('peft', 'PEFT'),
        ('accelerate', 'Accelerate'),
        ('bitsandbytes', 'BitsAndBytes'),
        ('cv2', 'OpenCV'),
        ('pandas', 'Pandas'),
        ('PIL', 'Pillow')
    ]
    
    missing = []
    
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NO INSTALADO")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Faltan dependencias: {', '.join(missing)}")
        print("   Ejecuta: pip install -r requirements.txt")
        return False
    
    return True


def check_cuda() -> None:
    """Verifica disponibilidad de CUDA."""
    print("\n🖥️  Verificando GPU...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✅ CUDA disponible")
            print(f"     - Versión CUDA: {torch.version.cuda}")
            print(f"     - Dispositivos: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / 1024**3
                print(f"     - GPU {i}: {props.name} ({vram_gb:.1f} GB)")
        else:
            print("  ⚠️  CUDA no disponible - se usará CPU")
            print("     (El entrenamiento será muy lento)")
    
    except Exception as e:
        print(f"  ❌ Error verificando CUDA: {e}")


def check_files() -> bool:
    """Verifica que los archivos de datos existan."""
    print("\n📁 Verificando archivos de datos...")
    
    from config import config
    
    files_to_check = [
        (config.paths.train_csv, "Train CSV"),
        (config.paths.val_csv, "Val CSV"),
        (config.paths.test_csv, "Test CSV"),
        (config.paths.images_dir, "Directorio de imágenes")
    ]
    
    all_ok = True
    
    for file_path, name in files_to_check:
        if file_path.exists():
            if file_path.is_file():
                size_mb = file_path.stat().st_size / 1024**2
                print(f"  ✅ {name}: {file_path.name} ({size_mb:.2f} MB)")
            else:
                # Es un directorio
                print(f"  ✅ {name}: {file_path.name}")
        else:
            print(f"  ❌ {name}: NO ENCONTRADO")
            print(f"     Esperado en: {file_path}")
            all_ok = False
    
    return all_ok


def check_images() -> Tuple[int, bool]:
    """Cuenta imágenes disponibles."""
    print("\n🖼️  Verificando imágenes...")
    
    from config import config
    
    if not config.paths.images_dir.exists():
        print(f"  ❌ Directorio de imágenes no existe: {config.paths.images_dir}")
        return 0, False
    
    # Contar imágenes
    imagenes = list(config.paths.images_dir.rglob("*.jpg"))
    imagenes += list(config.paths.images_dir.rglob("*.png"))
    
    num_imagenes = len(imagenes)
    
    if num_imagenes == 0:
        print(f"  ❌ No se encontraron imágenes en {config.paths.images_dir}")
        return 0, False
    
    print(f"  ✅ Encontradas {num_imagenes} imágenes")
    
    # Verificar una imagen de muestra
    if num_imagenes > 0:
        muestra = imagenes[0]
        print(f"  📸 Muestra: {muestra.relative_to(config.paths.images_dir)}")
    
    return num_imagenes, True


def check_csv_data() -> bool:
    """Verifica el contenido de los CSV."""
    print("\n📊 Verificando contenido de CSV...")
    
    try:
        import pandas as pd
        from config import config
        
        # Verificar train CSV
        df_train = pd.read_csv(config.paths.train_csv)
        print(f"  ✅ Train: {len(df_train)} muestras")
        
        # Verificar columnas requeridas
        required_cols = ['dicom_id', 'report_text']
        missing_cols = [col for col in required_cols if col not in df_train.columns]
        
        if missing_cols:
            print(f"  ❌ Faltan columnas en CSV: {missing_cols}")
            return False
        
        print(f"     - Columnas: {', '.join(df_train.columns[:5])}...")
        
        # Verificar datos no vacíos
        empty_reports = df_train['report_text'].isna().sum()
        if empty_reports > 0:
            print(f"  ⚠️  {empty_reports} reportes vacíos en train")
        
        # Verificar val CSV
        if config.paths.val_csv.exists():
            df_val = pd.read_csv(config.paths.val_csv)
            print(f"  ✅ Val: {len(df_val)} muestras")
        
        # Verificar test CSV
        if config.paths.test_csv.exists():
            df_test = pd.read_csv(config.paths.test_csv)
            print(f"  ✅ Test: {len(df_test)} muestras")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error leyendo CSV: {e}")
        return False


def check_disk_space() -> None:
    """Verifica espacio en disco."""
    print("\n💾 Verificando espacio en disco...")
    
    try:
        import shutil
        from config import config
        
        total, used, free = shutil.disk_usage(config.paths.base_dir)
        
        free_gb = free / 1024**3
        total_gb = total / 1024**3
        
        print(f"  📊 Espacio libre: {free_gb:.1f} GB / {total_gb:.1f} GB")
        
        if free_gb < 10:
            print(f"  ⚠️  Poco espacio libre (se recomienda >10 GB)")
        else:
            print(f"  ✅ Espacio suficiente")
            
    except Exception as e:
        print(f"  ⚠️  No se pudo verificar espacio: {e}")


def main():
    print("\n" + "="*60)
    print("🔧 VERIFICACIÓN DE CONFIGURACIÓN")
    print("="*60)
    
    checks = {
        'dependencies': check_dependencies(),
        'files': check_files(),
        'csv': check_csv_data(),
    }
    
    # Verificaciones informativas (no bloquean)
    check_cuda()
    num_images, images_ok = check_images()
    checks['images'] = images_ok
    check_disk_space()
    
    # Resumen final
    print("\n" + "="*60)
    print("📋 RESUMEN")
    print("="*60)
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name.capitalize()}")
    
    if all_passed:
        print("\n🎉 ¡Todo listo para entrenar!")
        print("\n💡 Comandos útiles:")
        print("   - Entrenar: python train.py")
        print("   - Evaluar: python evaluate.py --model-dir modelo_entrenado")
        print("   - Prueba rápida: python quick_test.py")
        return 0
    else:
        print("\n⚠️  Hay problemas que resolver antes de entrenar")
        print("   Revisa los errores anteriores")
        return 1


if __name__ == "__main__":
    sys.exit(main())
