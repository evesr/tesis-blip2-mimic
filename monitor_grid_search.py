#!/usr/bin/env python
"""
Monitor de Grid Search en Tiempo Real
======================================
Script para monitorear el progreso del grid search mientras se ejecuta.

Uso:
    python monitor_grid_search.py

    # Actualización continua cada 10 segundos
    watch -n 10 python monitor_grid_search.py

Autor: Evelyn Silva Rozas
Fecha: Abril 2026
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

def monitor_grid_search():
    """Muestra el estado actual del grid search."""
    
    results_dir = Path("grid_search_results")
    
    if not results_dir.exists():
        print("❌ No se encontró la carpeta grid_search_results/")
        print("   El grid search aún no ha iniciado.")
        return
    
    print("\n" + "="*80)
    print("📊 MONITOR DE GRID SEARCH - BLIP2 + LoRA")
    print("="*80 + "\n")
    
    # ========================================================================
    # 1. Leer archivo de progreso
    # ========================================================================
    progress_file = results_dir / "progress.txt"
    if progress_file.exists():
        print("📈 PROGRESO GENERAL:")
        print("-" * 40)
        with open(progress_file, 'r') as f:
            print(f.read())
    else:
        print("⏳ Esperando inicio del grid search...\n")
        return
    
    # ========================================================================
    # 2. Leer resultados intermedios
    # ========================================================================
    intermediate_file = results_dir / "intermediate_results.csv"
    if not intermediate_file.exists():
        print("📝 Aún no hay resultados disponibles.\n")
        return
    
    df = pd.read_csv(intermediate_file)
    
    if len(df) == 0:
        print("📝 Archivo de resultados vacío.\n")
        return
    
    # ========================================================================
    # 3. Análisis de resultados actuales
    # ========================================================================
    print("\n" + "="*80)
    print("📊 RESULTADOS HASTA AHORA")
    print("="*80 + "\n")
    
    # Combinaciones completadas
    combos_completed = df.groupby(['r', 'alpha', 'lr']).size().reset_index(name='epochs')
    n_combos = len(combos_completed)
    print(f"✅ Combinaciones completadas: {n_combos}")
    print(f"📝 Total de épocas registradas: {len(df)}\n")
    
    # Última época de cada combinación
    df_latest = df.loc[df.groupby(['r', 'alpha', 'lr'])['epoch'].idxmax()]
    
    # Mejores combinaciones hasta ahora (por val_loss)
    df_best = df_latest.sort_values('val_loss').head(5)
    
    print("🏆 TOP 5 MEJORES COMBINACIONES (hasta ahora):")
    print("-" * 80)
    print(df_best[['r', 'alpha', 'lr', 'epoch', 'val_loss', 'bleu4', 'rougeL']].to_string(index=False))
    
    # ========================================================================
    # 4. Última combinación en progreso
    # ========================================================================
    if 'timestamp' in df.columns:
        df_sorted = df.sort_values('timestamp', ascending=False)
        last_row = df_sorted.iloc[0]
        
        print("\n" + "="*80)
        print("⏱️  ÚLTIMA ÉPOCA COMPLETADA")
        print("="*80)
        print(f"\n🔧 Combinación: r={last_row['r']}, alpha={last_row['alpha']}, lr={last_row['lr']:.2e}")
        if 'combo_num' in df.columns:
            print(f"📍 Número: {last_row['combo_num']}")
        print(f"📊 Época: {last_row['epoch']}")
        print(f"🕐 Timestamp: {last_row['timestamp']}")
        print(f"\n📉 Métricas:")
        print(f"   • Train Loss: {last_row['train_loss']:.4f}")
        print(f"   • Val Loss: {last_row['val_loss']:.4f}")
        print(f"   • BLEU-4: {last_row['bleu4']:.4f}")
        print(f"   • ROUGE-L: {last_row['rougeL']:.4f}")
    
    # ========================================================================
    # 5. Estadísticas generales
    # ========================================================================
    print("\n" + "="*80)
    print("📈 ESTADÍSTICAS GENERALES")
    print("="*80 + "\n")
    
    print(f"Val Loss  - Min: {df['val_loss'].min():.4f} | Max: {df['val_loss'].max():.4f} | Media: {df['val_loss'].mean():.4f}")
    print(f"BLEU-4    - Min: {df['bleu4'].min():.4f} | Max: {df['bleu4'].max():.4f} | Media: {df['bleu4'].mean():.4f}")
    print(f"ROUGE-L   - Min: {df['rougeL'].min():.4f} | Max: {df['rougeL'].max():.4f} | Media: {df['rougeL'].mean():.4f}")
    
    print("\n" + "="*80)
    print("💡 TIP: Usa 'watch -n 10 python monitor_grid_search.py' para actualización automática")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        monitor_grid_search()
    except KeyboardInterrupt:
        print("\n\n👋 Monitor detenido por usuario.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)
