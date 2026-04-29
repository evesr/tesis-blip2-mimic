#!/usr/bin/env python3
"""
Script para evaluar el mejor modelo de Config_4 y obtener métricas BLEU/ROUGE
"""

import torch
import os
import sys
from pathlib import Path
from tqdm import tqdm
from evaluate import load
import json

# Importar funciones del proyecto
sys.path.append('/workspace/tesis-blip2-mimic/Tesis_blip2_local')
from data_loader import load_datasets
from model_utils import load_blip2_model, generate_report
from config import BASE_MODEL

def evaluate_model_on_val(model, val_dataset, processor, device, num_samples=50):
    """
    Evalúa modelo en subset del val set calculando BLEU-4 y ROUGE-L.
    """
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
    
    # Cargar métricas
    bleu_metric = load('bleu')
    rouge_metric = load('rouge')
    
    predictions = []
    references = []
    
    model.eval()
    
    print(f"\n   🔍 Generando predicciones en {num_samples} muestras...")
    
    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Evaluando"):
            sample = val_dataset[idx]
            
            # Preparar imagen
            pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
            
            # Generar reporte
            generated_text = generate_report(
                model=model,
                processor=processor,
                pixel_values=pixel_values,
                device=device,
                max_new_tokens=150,
                num_beams=4,
                do_sample=False
            )
            
            predictions.append(generated_text)
            references.append(sample['text'])
    
    # Calcular BLEU-4
    bleu_result = bleu_metric.compute(
        predictions=predictions,
        references=[[ref] for ref in references],
        max_order=4
    )
    
    # Calcular ROUGE-L
    rouge_result = rouge_metric.compute(
        predictions=predictions,
        references=references,
        use_aggregator=True
    )
    
    return {
        'bleu4': bleu_result['bleu'],
        'rougeL': rouge_result['rougeL'],
        'predictions': predictions,
        'references': references
    }


def find_best_checkpoint(config_dir):
    """Encuentra el checkpoint del mejor modelo"""
    # Buscar el último trainer_state.json
    checkpoints = []
    for item in os.listdir(config_dir):
        if item.startswith('checkpoint-'):
            checkpoint_path = os.path.join(config_dir, item, 'trainer_state.json')
            if os.path.exists(checkpoint_path):
                checkpoint_num = int(item.split('-')[1])
                checkpoints.append((checkpoint_num, checkpoint_path, item))
    
    if not checkpoints:
        return None
    
    # Ordenar y tomar el más reciente
    checkpoints.sort(reverse=True)
    _, trainer_state_path, checkpoint_dir_name = checkpoints[0]
    
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    
    # Encontrar mejor checkpoint
    best_checkpoint = trainer_state.get('best_model_checkpoint')
    
    if best_checkpoint:
        # Extraer número del path
        best_step = Path(best_checkpoint).name.split('-')[-1]
        best_checkpoint_dir = os.path.join(config_dir, f'checkpoint-{best_step}')
        if os.path.exists(best_checkpoint_dir):
            return best_checkpoint_dir
    
    # Si no hay best_model_checkpoint, buscar el de menor eval_loss
    log_history = trainer_state.get('log_history', [])
    eval_logs = [e for e in log_history if 'eval_loss' in e]
    
    if eval_logs:
        best_eval = min(eval_logs, key=lambda x: x['eval_loss'])
        best_epoch = int(best_eval['epoch'])
        
        # Buscar checkpoint más cercano a esa época
        # (asumiendo que cada época son ~167 steps con tu dataset)
        approx_step = best_epoch * 167
        
        # Buscar el checkpoint más cercano
        available_checkpoints = [int(item.split('-')[1]) for item in os.listdir(config_dir) if item.startswith('checkpoint-')]
        closest_checkpoint = min(available_checkpoints, key=lambda x: abs(x - approx_step))
        
        return os.path.join(config_dir, f'checkpoint-{closest_checkpoint}')
    
    return None


def main():
    config_name = 'Config_4'
    base_dir = Path('/workspace/tesis-blip2-mimic/Tesis_blip2_local/semifinal_results')
    config_dir = base_dir / config_name
    
    print("="*80)
    print(f"🔍 EVALUANDO {config_name} PARA OBTENER MÉTRICAS BLEU/ROUGE")
    print("="*80)
    
    # Encontrar mejor checkpoint
    print(f"\n📂 Buscando mejor checkpoint...")
    best_checkpoint_path = find_best_checkpoint(str(config_dir))
    
    if not best_checkpoint_path:
        print(f"❌ No se encontró checkpoint en {config_dir}")
        return
    
    print(f"✅ Mejor checkpoint: {best_checkpoint_path}")
    
    # Cargar datos
    print(f"\n📥 Cargando datasets...")
    _, val_dataset, _, processor = load_datasets()
    
    # Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Cargar modelo base
    print(f"\n🤖 Cargando modelo base {BASE_MODEL}...")
    model, _ = load_blip2_model(
        model_name=BASE_MODEL,
        device=device,
        load_in_8bit=False,
        use_soft_metadata=True  # Config_4 usó soft metadata
    )
    
    # Cargar adaptadores LoRA
    print(f"\n🔧 Cargando adaptadores LoRA desde checkpoint...")
    from peft import PeftModel
    
    model = PeftModel.from_pretrained(model, best_checkpoint_path)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modelo cargado exitosamente")
    
    # Evaluar
    print(f"\n📊 Evaluando modelo en validation set...")
    metrics = evaluate_model_on_val(
        model=model,
        val_dataset=val_dataset,
        processor=processor,
        device=device,
        num_samples=50
    )
    
    print(f"\n" + "="*80)
    print(f"✅ RESULTADOS DE {config_name}")
    print("="*80)
    print(f"   BLEU-4:  {metrics['bleu4']:.4f}")
    print(f"   ROUGE-L: {metrics['rougeL']:.4f}")
    print("="*80)
    
    # Guardar resultados
    output_file = config_dir / 'metrics_bleu_rouge.json'
    with open(output_file, 'w') as f:
        json.dump({
            'bleu4': metrics['bleu4'],
            'rougeL': metrics['rougeL'],
            'checkpoint': best_checkpoint_path
        }, f, indent=2)
    
    print(f"\n💾 Métricas guardadas en: {output_file}")


if __name__ == '__main__':
    main()
