"""Re-export mrt2_base with 4-bit quantization + num_cfgs=1 for faster inference."""
import sys
sys.path.insert(0, 'venv/lib/python3.12/site-packages')

from magenta_rt.mlx import export

print("🚀 Re-exporting mrt2_base with 4-bit quant + 1 CFG...")
print("   This should give ~2-3x speedup over the original export.")
print()

export.main(
    model_name='mrt2_base',
    bits=4,              # 4-bit quantization (~half memory bandwidth)
    num_cfgs=1,          # Only musiccoca CFG (batch 2x instead of 3x)
    output_name='mrt2_base_fast',
    output_dir='/Users/e.baena/Documents/Magenta/magenta-rt-v2/models',
)

print("\n✅ Export complete! Model saved as mrt2_base_fast")
