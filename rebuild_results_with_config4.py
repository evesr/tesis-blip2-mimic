#!/usr/bin/env python3
"""
Script para reconstruir los resultados finales incluyendo Config_4
Extrae métricas de trainer_state.json de todas las configuraciones
"""

import json
import os
import pandas as pd
from datetime import datetime

def extract_metrics_from_trainer_state(config_dir, config_name, r, alpha):
    """Extrae métricas finales desde el último trainer_state.json"""
    
    # Buscar checkpoints
    checkpoints = []
    for item in os.listdir(config_dir):
        if item.startswith('checkpoint-'):
            checkpoint_path = os.path.join(config_dir, item, 'trainer_state.json')
            if os.path.exists(checkpoint_path):
                # Extraer número de checkpoint
                checkpoint_num = int(item.split('-')[1])
                checkpoints.append((checkpoint_num, checkpoint_path))
    
    if not checkpoints:
        print(f"⚠️ No se encontró trainer_state.json en {config_dir}")
        return None
    
    # Usar el checkpoint más reciente
    checkpoints.sort(reverse=True)
    latest_checkpoint_path = checkpoints[0][1]
    
    print(f"📖 Leyendo: {latest_checkpoint_path}")
    
    with open(latest_checkpoint_path, 'r') as f:
        trainer_state = json.load(f)
    
    # Extraer métricas
    log_history = trainer_state.get('log_history', [])
    
    # Encontrar mejor val_loss y su época
    eval_logs = [entry for entry in log_history if 'eval_loss' in entry]
    
    if not eval_logs:
        print(f"⚠️ No hay eval_loss en {config_name}")
        return None
    
    # Encontrar el mejor
    best_eval = min(eval_logs, key=lambda x: x['eval_loss'])
    best_epoch = int(best_eval['epoch'])
    best_val_loss = best_eval['eval_loss']
    
    # Métricas del mejor punto
    bleu4 = best_eval.get('eval_bleu-4', 0.0)
    rougeL = best_eval.get('eval_rouge-L', 0.0)
    
    # Total de épocas
    total_epochs = int(max([e['epoch'] for e in eval_logs]))
    
    # Final train loss (último log de training)
    train_logs = [entry for entry in log_history if 'loss' in entry]
    final_train_loss = train_logs[-1]['loss'] if train_logs else 0.0
    
    # Timestamp (usar el del archivo o generar uno)
    timestamp = datetime.fromtimestamp(os.path.getmtime(latest_checkpoint_path))
    
    return {
        'config_name': config_name,
        'r': r,
        'alpha': alpha,
        'lr': 0.0001,  # Todos usan el mismo LR
        'total_epochs_trained': float(total_epochs),
        'best_epoch': float(best_epoch),
        'best_val_loss': best_val_loss,
        'final_train_loss': final_train_loss,
        'bleu4': bleu4,
        'rougeL': rougeL,
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
    }


def main():
    base_dir = '/workspace/tesis-blip2-mimic/Tesis_blip2_local/semifinal_results'
    
    # Configuraciones y sus parámetros
    configs = [
        ('Config_4', 16, 32),
        ('Config_5', 32, 32),
        ('Config_6', 32, 64),
    ]
    
    results = []
    
    print("="*60)
    print("🔍 EXTRAYENDO MÉTRICAS DE TODAS LAS CONFIGURACIONES")
    print("="*60)
    
    for config_name, r, alpha in configs:
        config_dir = os.path.join(base_dir, config_name)
        
        if not os.path.exists(config_dir):
            print(f"\n⚠️ {config_name}: Directorio no encontrado")
            continue
        
        print(f"\n📊 Procesando {config_name} (r={r}, α={alpha})...")
        
        metrics = extract_metrics_from_trainer_state(config_dir, config_name, r, alpha)
        
        if metrics:
            results.append(metrics)
            print(f"  ✅ Épocas: {metrics['total_epochs_trained']:.0f}, "
                  f"Mejor época: {metrics['best_epoch']:.0f}")
            print(f"  ✅ Best val_loss: {metrics['best_val_loss']:.4f}")
            print(f"  ✅ BLEU-4: {metrics['bleu4']:.4f}, ROUGE-L: {metrics['rougeL']:.4f}")
    
    if not results:
        print("\n❌ No se encontraron resultados")
        return
    
    # Crear DataFrame
    df = pd.DataFrame(results)
    
    # Ordenar por best_val_loss (mejor primero)
    df = df.sort_values('best_val_loss')
    
    # Guardar archivos
    output_files = [
        'intermediate_results.csv',
        'final_results_ranked.csv'
    ]
    
    print("\n" + "="*60)
    print("💾 GUARDANDO RESULTADOS")
    print("="*60)
    
    for filename in output_files:
        output_path = os.path.join(base_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"✅ Guardado: {output_path}")
    
    print("\n📊 RANKING FINAL:")
    print(df[['config_name', 'r', 'alpha', 'best_val_loss', 'bleu4', 'rougeL']].to_string(index=False))
    
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)


if __name__ == '__main__':
    main()
