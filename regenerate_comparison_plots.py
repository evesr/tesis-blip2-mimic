#!/usr/bin/env python3
"""
Script para regenerar gráficos de comparación con Config_4 incluida
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    base_dir = Path('/workspace/tesis-blip2-mimic/Tesis_blip2_local/semifinal_results')
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Leer resultados
    df_results = pd.read_csv(base_dir / "final_results_ranked.csv")
    
    print("="*80)
    print("📊 REGENERANDO GRÁFICOS DE COMPARACIÓN CON CONFIG_4")
    print("="*80)
    print("\n📋 Datos cargados:")
    print(df_results[['config_name', 'r', 'alpha', 'best_val_loss', 'bleu4', 'rougeL']].to_string(index=False))
    print()
    
    # Gráfico 1: Comparación de Val Loss
    plt.figure(figsize=(10, 6))
    plt.bar(df_results['config_name'], df_results['best_val_loss'], color='steelblue')
    plt.xlabel('Configuración')
    plt.ylabel('Best Val Loss')
    plt.title('Comparación de Mejor Val Loss - Semifinal')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_val_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ comparison_val_loss.png")
    
    # Gráfico 2: Comparación de BLEU-4
    plt.figure(figsize=(10, 6))
    plt.bar(df_results['config_name'], df_results['bleu4'], color='coral')
    plt.xlabel('Configuración')
    plt.ylabel('BLEU-4')
    plt.title('Comparación de BLEU-4 - Semifinal')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_bleu4.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ comparison_bleu4.png")
    
    # Gráfico 3: Comparación de ROUGE-L
    plt.figure(figsize=(10, 6))
    plt.bar(df_results['config_name'], df_results['rougeL'], color='mediumseagreen')
    plt.xlabel('Configuración')
    plt.ylabel('ROUGE-L')
    plt.title('Comparación de ROUGE-L - Semifinal')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_rougeL.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ comparison_rougeL.png")
    
    # Gráfico 4: Épocas entrenadas
    plt.figure(figsize=(10, 6))
    plt.bar(df_results['config_name'], df_results['total_epochs_trained'], color='plum')
    plt.axhline(y=30, color='red', linestyle='--', label='Máximo (30 épocas)')
    plt.xlabel('Configuración')
    plt.ylabel('Épocas Entrenadas')
    plt.title('Épocas Entrenadas (Early Stopping)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "comparison_epochs.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ comparison_epochs.png")
    
    # Gráfico 5: Dashboard completo (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Val Loss
    axes[0, 0].bar(df_results['config_name'], df_results['best_val_loss'], color='steelblue')
    axes[0, 0].set_title('Best Val Loss')
    axes[0, 0].set_ylabel('Val Loss')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # BLEU-4
    axes[0, 1].bar(df_results['config_name'], df_results['bleu4'], color='coral')
    axes[0, 1].set_title('BLEU-4')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # ROUGE-L
    axes[1, 0].bar(df_results['config_name'], df_results['rougeL'], color='mediumseagreen')
    axes[1, 0].set_title('ROUGE-L')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Épocas
    axes[1, 1].bar(df_results['config_name'], df_results['total_epochs_trained'], color='plum')
    axes[1, 1].axhline(y=30, color='red', linestyle='--', label='Máximo')
    axes[1, 1].set_title('Épocas Entrenadas')
    axes[1, 1].set_ylabel('Épocas')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].legend()
    
    plt.suptitle('Dashboard Semifinal - Comparación de Configuraciones', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(plots_dir / "dashboard_complete.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ dashboard_complete.png")
    
    print("\n" + "="*80)
    print("✅ TODOS LOS GRÁFICOS REGENERADOS CON CONFIG_4")
    print(f"📁 Ubicación: {plots_dir}")
    print("="*80)
    
    print("\n🏆 RANKING FINAL:")
    print(df_results[['config_name', 'r', 'alpha', 'best_val_loss', 'bleu4', 'rougeL']].to_string(index=False))

if __name__ == '__main__':
    main()
