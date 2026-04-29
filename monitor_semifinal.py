#!/usr/bin/env python
"""
Monitor de Entrenamiento Semifinal en Tiempo Real
==================================================
Script para monitorear el progreso del entrenamiento semifinal mientras se ejecuta.

Uso:
    python monitor_semifinal.py

    # Actualización continua cada 10 segundos
    watch -n 10 python monitor_semifinal.py

Autor: Evelyn Silva Rozas
Fecha: Abril 2026
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import json

def monitor_semifinal():
    """Muestra el estado actual del entrenamiento semifinal."""
    
    results_dir = Path("semifinal_results")
    
    if not results_dir.exists():
        print("❌ No se encontró la carpeta semifinal_results/")
        print("   El entrenamiento semifinal aún no ha iniciado.")
        return
    
    print("\n" + "="*80)
    print("📊 MONITOR DE ENTRENAMIENTO SEMIFINAL - BLIP2 + LoRA")
    print("="*80 + "\n")
    
    # ========================================================================
    # 1. Leer archivo de progreso
    # ========================================================================
    progress_file = results_dir / "progress.txt"
    if progress_file.exists():
        print("📈 PROGRESO GENERAL:")
        print("-" * 40)
        with open(progress_file, 'r') as f:
            content = f.read()
            print(content if content.strip() else "Iniciando...")
    else:
        print("⏳ Esperando inicio del entrenamiento semifinal...\n")
        return
    
    # ========================================================================
    # 2. Leer resultados intermedios
    # ========================================================================
    intermediate_file = results_dir / "intermediate_results.csv"
    if not intermediate_file.exists():
        print("\n📝 Aún no hay resultados disponibles.\n")
        return
    
    df = pd.read_csv(intermediate_file)
    
    if len(df) == 0:
        print("\n📝 Archivo de resultados vacío.\n")
        return
    
    # ========================================================================
    # 3. Análisis de resultados actuales
    # ========================================================================
    print("\n" + "="*80)
    print("📊 RESULTADOS HASTA AHORA")
    print("="*80 + "\n")
    
    # Configuraciones completadas
    configs_completed = df['config_name'].unique()
    n_configs = len(configs_completed)
    print(f"✅ Configuraciones completadas: {n_configs}/3")
    for config in configs_completed:
        config_df = df[df['config_name'] == config]
        epochs = config_df['epoch'].max()
        print(f"   • {config}: {epochs} épocas")
    
    print(f"\n📝 Total de épocas registradas: {len(df)}\n")
    
    # Última época de cada configuración
    df_latest = df.loc[df.groupby('config_name')['epoch'].idxmax()]
    
    # Mejores configuraciones hasta ahora (por val_loss)
    df_best = df_latest.sort_values('val_loss')
    
    print("🏆 RANKING DE CONFIGURACIONES (hasta ahora):")
    print("-" * 80)
    display_cols = ['config_name', 'r', 'alpha', 'epoch', 'val_loss', 'bleu4', 'rougeL']
    available_cols = [col for col in display_cols if col in df_best.columns]
    print(df_best[available_cols].to_string(index=False))
    
    # ========================================================================
    # 4. Última época completada
    # ========================================================================
    if 'timestamp' in df.columns:
        df_sorted = df.sort_values('timestamp', ascending=False)
        last_row = df_sorted.iloc[0]
        
        print("\n" + "="*80)
        print("⏱️  ÚLTIMA ÉPOCA COMPLETADA")
        print("="*80)
        print(f"\n🔧 Configuración: {last_row['config_name']}")
        print(f"   • r={last_row['r']}, alpha={last_row['alpha']}, lr={last_row['lr']:.2e}")
        print(f"📊 Época: {last_row['epoch']}")
        print(f"🕐 Timestamp: {last_row['timestamp']}")
        print(f"\n📉 Métricas:")
        print(f"   • Train Loss: {last_row['train_loss']:.4f}")
        print(f"   • Val Loss: {last_row['val_loss']:.4f}")
        print(f"   • BLEU-4: {last_row['bleu4']:.4f}")
        print(f"   • ROUGE-L: {last_row['rougeL']:.4f}")
    
    # ========================================================================
    # 5. Early Stopping Info
    # ========================================================================
    print("\n" + "="*80)
    print("🛑 EARLY STOPPING STATUS")
    print("="*80 + "\n")
    
    for config_name in df['config_name'].unique():
        config_df = df[df['config_name'] == config_name].sort_values('epoch')
        
        if len(config_df) > 0:
            # Buscar el mejor val_loss
            best_epoch = config_df.loc[config_df['val_loss'].idxmin(), 'epoch']
            current_epoch = config_df['epoch'].max()
            best_val_loss = config_df['val_loss'].min()
            
            print(f"📌 {config_name}:")
            print(f"   • Mejor época: {best_epoch} (val_loss={best_val_loss:.4f})")
            print(f"   • Época actual: {current_epoch}")
            
            # Verificar si está cerca del early stopping
            epochs_since_improvement = current_epoch - best_epoch
            if epochs_since_improvement >= 3:
                print(f"   ⚠️  {epochs_since_improvement} épocas sin mejora (patience=3) - Próximo a detenerse")
            elif epochs_since_improvement > 0:
                print(f"   ⏳ {epochs_since_improvement}/3 épocas sin mejora")
            else:
                print(f"   ✅ Mejorando activamente")
            print()
    
    # ========================================================================
    # 6. Estadísticas generales
    # ========================================================================
    print("="*80)
    print("📈 ESTADÍSTICAS GENERALES")
    print("="*80 + "\n")
    
    print(f"Val Loss  - Min: {df['val_loss'].min():.4f} | Max: {df['val_loss'].max():.4f} | Media: {df['val_loss'].mean():.4f}")
    print(f"BLEU-4    - Min: {df['bleu4'].min():.4f} | Max: {df['bleu4'].max():.4f} | Media: {df['bleu4'].mean():.4f}")
    print(f"ROUGE-L   - Min: {df['rougeL'].min():.4f} | Max: {df['rougeL'].max():.4f} | Media: {df['rougeL'].mean():.4f}")
    
    # ========================================================================
    # 7. Verificar si hay gráficos generados
    # ========================================================================
    plots_dir = results_dir / "plots"
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        if plot_files:
            print("\n" + "="*80)
            print("📊 GRÁFICOS DISPONIBLES")
            print("="*80 + "\n")
            for plot in plot_files:
                print(f"   📈 {plot.name}")
    
    print("\n" + "="*80)
    print("💡 TIP: Usa 'watch -n 10 python monitor_semifinal.py' para actualización automática")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        monitor_semifinal()
    except KeyboardInterrupt:
        print("\n\n👋 Monitor detenido por usuario.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
