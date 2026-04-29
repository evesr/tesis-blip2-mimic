#!/usr/bin/env python3
"""
Script para generar curvas de entrenamiento completas desde trainer_state.json
Extrae el historial de loss, learning rate y genera gráficos detallados
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_training_history(config_dir):
    """Extrae el historial de entrenamiento desde los checkpoints"""
    
    # Buscar el trainer_state.json más reciente (del último checkpoint)
    checkpoints = []
    for item in os.listdir(config_dir):
        if item.startswith('checkpoint-'):
            checkpoint_path = os.path.join(config_dir, item, 'trainer_state.json')
            if os.path.exists(checkpoint_path):
                checkpoints.append(checkpoint_path)
    
    if not checkpoints:
        print(f"⚠️ No se encontró trainer_state.json en {config_dir}")
        return None
    
    # Usar el último checkpoint (debería tener el historial completo)
    latest_checkpoint = sorted(checkpoints)[-1]
    
    with open(latest_checkpoint, 'r') as f:
        trainer_state = json.load(f)
    
    # Extraer log_history
    log_history = trainer_state.get('log_history', [])
    
    if not log_history:
        print(f"⚠️ log_history vacío en {latest_checkpoint}")
        return None
    
    # Organizar datos
    epochs = []
    train_loss = []
    eval_loss = []
    learning_rate = []
    
    for entry in log_history:
        epoch = entry.get('epoch')
        
        if 'loss' in entry:  # Log de entrenamiento
            epochs.append(epoch)
            train_loss.append(entry['loss'])
            learning_rate.append(entry.get('learning_rate', np.nan))
        
        if 'eval_loss' in entry:  # Log de evaluación
            if epoch not in [e for e in epochs if e == epoch]:
                epochs.append(epoch)
            eval_loss.append((epoch, entry['eval_loss']))
    
    return {
        'log_history': log_history,
        'epochs': epochs,
        'train_loss': train_loss,
        'eval_loss': eval_loss,
        'learning_rate': learning_rate,
        'best_metric': trainer_state.get('best_metric'),
        'best_model_checkpoint': trainer_state.get('best_model_checkpoint')
    }


def plot_individual_config(config_name, history, output_dir):
    """Genera gráficos detallados para una configuración individual"""
    
    if history is None:
        print(f"⚠️ No hay datos para {config_name}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Entrenamiento Semifinal - {config_name}', fontsize=16, fontweight='bold')
    
    # 1. Training Loss
    ax1 = axes[0, 0]
    if history['train_loss']:
        ax1.plot(history['epochs'][:len(history['train_loss'])], history['train_loss'], 
                 'b-', linewidth=2, label='Train Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # 2. Validation Loss
    ax2 = axes[0, 1]
    if history['eval_loss']:
        eval_epochs, eval_losses = zip(*history['eval_loss'])
        ax2.plot(eval_epochs, eval_losses, 'r-o', linewidth=2, 
                 markersize=6, label='Val Loss', alpha=0.7)
        
        # Marcar el mejor punto
        if history['best_metric'] is not None:
            best_loss = history['best_metric']
            # Encontrar la época del mejor loss
            best_idx = np.argmin(eval_losses)
            best_epoch = eval_epochs[best_idx]
            ax2.plot(best_epoch, best_loss, 'g*', markersize=15, 
                     label=f'Best (epoch {best_epoch:.0f}): {best_loss:.4f}')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss (evaluado cada época)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # 3. Training + Validation Loss juntos
    ax3 = axes[1, 0]
    if history['train_loss']:
        ax3.plot(history['epochs'][:len(history['train_loss'])], history['train_loss'], 
                 'b-', linewidth=2, label='Train Loss', alpha=0.6)
    if history['eval_loss']:
        eval_epochs, eval_losses = zip(*history['eval_loss'])
        ax3.plot(eval_epochs, eval_losses, 'r-o', linewidth=2, 
                 markersize=6, label='Val Loss', alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Train vs Validation Loss')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Learning Rate
    ax4 = axes[1, 1]
    if history['learning_rate']:
        valid_lr = [(e, lr) for e, lr in zip(history['epochs'][:len(history['learning_rate'])], 
                                               history['learning_rate']) if not np.isnan(lr)]
        if valid_lr:
            lr_epochs, lr_values = zip(*valid_lr)
            ax4.plot(lr_epochs, lr_values, 'g-', linewidth=2, alpha=0.7)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{config_name}_training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {output_path}")
    plt.close()


def plot_all_configs_comparison(all_histories, output_dir):
    """Genera un gráfico comparativo de todas las configuraciones"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Comparación de Configuraciones Semifinales', fontsize=16, fontweight='bold')
    
    colors = {'Config_4': 'blue', 'Config_5': 'red', 'Config_6': 'green'}
    markers = {'Config_4': 'o', 'Config_5': 's', 'Config_6': '^'}
    
    # 1. Validation Loss Comparison
    ax1 = axes[0]
    for config_name, history in all_histories.items():
        if history and history['eval_loss']:
            eval_epochs, eval_losses = zip(*history['eval_loss'])
            color = colors.get(config_name, 'black')
            marker = markers.get(config_name, 'o')
            ax1.plot(eval_epochs, eval_losses, color=color, marker=marker, 
                     linewidth=2, markersize=5, label=config_name, alpha=0.7, linestyle='-')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss - Todas las Configuraciones')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Training Loss Comparison
    ax2 = axes[1]
    for config_name, history in all_histories.items():
        if history and history['train_loss']:
            color = colors.get(config_name, 'black')
            epochs = history['epochs'][:len(history['train_loss'])]
            ax2.plot(epochs, history['train_loss'], color=color, 
                     linewidth=2, label=config_name, alpha=0.6, linestyle='-')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Training Loss - Todas las Configuraciones')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'all_configs_loss_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Gráfico comparativo guardado: {output_path}")
    plt.close()


def main():
    base_dir = '/workspace/tesis-blip2-mimic/Tesis_blip2_local/semifinal_results'
    output_dir = os.path.join(base_dir, 'plots')
    
    configs = ['Config_4', 'Config_5', 'Config_6']
    all_histories = {}
    
    print("="*60)
    print("📊 GENERANDO CURVAS DE ENTRENAMIENTO COMPLETAS")
    print("="*60)
    
    for config_name in configs:
        config_dir = os.path.join(base_dir, config_name)
        
        if not os.path.exists(config_dir):
            print(f"\n⚠️ {config_name}: Directorio no encontrado")
            continue
        
        print(f"\n🔍 Procesando {config_name}...")
        history = extract_training_history(config_dir)
        
        if history:
            all_histories[config_name] = history
            
            # Información resumida
            print(f"  - Épocas totales: {len(history['eval_loss'])}")
            if history['best_metric']:
                print(f"  - Mejor val_loss: {history['best_metric']:.4f}")
            
            # Generar gráfico individual
            plot_individual_config(config_name, history, output_dir)
    
    # Generar gráfico comparativo
    if all_histories:
        print(f"\n📈 Generando gráfico comparativo...")
        plot_all_configs_comparison(all_histories, output_dir)
    
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print(f"📁 Gráficos guardados en: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
