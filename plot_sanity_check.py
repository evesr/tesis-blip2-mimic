"""
Visualización de resultados del Sanity Check
=============================================
Lee los logs de entrenamiento y genera gráficos del loss.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

# Buscar logs
output_dir = Path("/workspace/tesis-blip2-mimic/Tesis_blip2_local/sanity_check_output")
logs_dir = output_dir / "logs"

print("📂 Buscando logs en:", logs_dir)

# Buscar archivo de estado del trainer
trainer_state_file = None
for f in output_dir.rglob("trainer_state.json"):
    trainer_state_file = f
    break

if trainer_state_file is None:
    print("❌ No se encontró trainer_state.json")
    print("   Buscando en subdirectorios...")
    for checkpoint_dir in output_dir.glob("checkpoint-*"):
        state_file = checkpoint_dir / "trainer_state.json"
        if state_file.exists():
            trainer_state_file = state_file
            break

if trainer_state_file is None:
    print("❌ No se encontraron logs de entrenamiento")
    exit(1)

print(f"✅ Encontrado: {trainer_state_file}")

# Leer el archivo de estado
with open(trainer_state_file, 'r') as f:
    trainer_state = json.load(f)

# Extraer log history
log_history = trainer_state.get('log_history', [])

if not log_history:
    print("❌ No hay datos en log_history")
    exit(1)

print(f"📊 {len(log_history)} entradas de log encontradas")

# Convertir a DataFrame
df = pd.DataFrame(log_history)

# Filtrar solo entradas con loss
df_train = df[df['loss'].notna()].copy() if 'loss' in df.columns else pd.DataFrame()

print(f"\n📈 Entradas de entrenamiento: {len(df_train)}")

if len(df_train) == 0:
    print("❌ No hay datos de loss de entrenamiento")
    exit(1)

# Crear figura
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Loss vs Steps
ax1 = axes[0]
ax1.plot(df_train['step'], df_train['loss'], 'b-', linewidth=2, marker='o', markersize=4)
ax1.set_xlabel('Steps', fontweight='bold', fontsize=12)
ax1.set_ylabel('Training Loss', fontweight='bold', fontsize=12)
ax1.set_title('Training Loss vs Steps (Sanity Check)', fontweight='bold', fontsize=14)
ax1.grid(True, alpha=0.3)

# Anotar loss inicial y final
loss_inicial = df_train['loss'].iloc[0]
loss_final = df_train['loss'].iloc[-1]
ax1.axhline(y=loss_inicial, color='r', linestyle='--', alpha=0.5, label=f'Inicial: {loss_inicial:.4f}')
ax1.axhline(y=loss_final, color='g', linestyle='--', alpha=0.5, label=f'Final: {loss_final:.4f}')
ax1.legend(loc='best')

# Gráfico 2: Loss vs Epoch
ax2 = axes[1]
ax2.plot(df_train['epoch'], df_train['loss'], 'r-', linewidth=2, marker='s', markersize=4)
ax2.set_xlabel('Epoch', fontweight='bold', fontsize=12)
ax2.set_ylabel('Training Loss', fontweight='bold', fontsize=12)
ax2.set_title('Training Loss vs Epoch', fontweight='bold', fontsize=14)
ax2.grid(True, alpha=0.3)

# Estadísticas
reduction = ((loss_inicial - loss_final) / loss_inicial) * 100
ax2.text(0.05, 0.95, f'Reducción: {reduction:.1f}%\nLoss final: {loss_final:.4f}',
         transform=ax2.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Guardar
output_plot = output_dir / "training_loss_plot.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\n💾 Gráfico guardado: {output_plot}")

# Mostrar estadísticas
print("\n" + "="*60)
print("📊 ESTADÍSTICAS DE ENTRENAMIENTO")
print("="*60)
print(f"\n📉 Loss Inicial: {loss_inicial:.4f}")
print(f"📉 Loss Final:   {loss_final:.4f}")
print(f"📉 Reducción:    {reduction:.1f}%")
print(f"\n📊 Steps totales: {df_train['step'].iloc[-1]}")
print(f"📊 Épocas completadas: {df_train['epoch'].iloc[-1]:.1f}")

# Evaluación
print("\n" + "="*60)
print("🎯 EVALUACIÓN DEL SANITY CHECK")
print("="*60)

if loss_final < 0.1:
    print("\n✅ EXCELENTE: Loss < 0.1")
    print("   ✓ El modelo aprende perfectamente")
    print("   ✓ Código validado para entrenamiento completo")
    status = "EXCELENTE"
elif loss_final < 0.5:
    print("\n✅ BUENO: Loss < 0.5")
    print("   ✓ El modelo aprende correctamente")
    print("   ✓ Código validado para entrenamiento completo")
    status = "BUENO"
elif loss_final < 1.0:
    print("\n⚠️  ACEPTABLE: Loss < 1.0")
    print("   ⚠ El modelo aprende pero podría mejorar")
    print("   💡 Considera aumentar épocas o ajustar LR")
    status = "ACEPTABLE"
else:
    print("\n❌ PROBLEMA: Loss > 1.0")
    print("   ✗ El modelo NO está aprendiendo bien")
    print("   🔧 Revisa: Learning rate, datos, arquitectura")
    status = "PROBLEMA"

print("\n" + "="*60)

# Guardar resumen
summary = {
    'loss_inicial': float(loss_inicial),
    'loss_final': float(loss_final),
    'reduccion_pct': float(reduction),
    'steps_totales': int(df_train['step'].iloc[-1]),
    'epochs_completadas': float(df_train['epoch'].iloc[-1]),
    'status': status
}

summary_file = output_dir / "sanity_check_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"💾 Resumen guardado: {summary_file}")

plt.show()
