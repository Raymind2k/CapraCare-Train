"""
CapraCare - Publication-Quality Training Curves
Generates per-model accuracy + loss curves for the research paper.
Style matches standard ML papers (e.g., JATIT 2025 EfficientNet-B4 paper).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# Configuration
# ============================================================
RESULTS_JSON = Path("runs/classification/comparison/comparison_results.json")
OUTPUT_DIR = Path("runs/classification/paper_figures")

# Display order in figure (best model last, like the reference paper)
MODEL_ORDER = ["MobileNetV2", "DenseNet121", "ResNet50", "EfficientNet-B0"]


# ============================================================
# Paper-style plot settings
# ============================================================
def setup_paper_style():
    """Configure matplotlib for publication-quality output."""
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def plot_single_model(model_data, output_path):
    """Create one figure with accuracy + loss panels for a single model."""
    history = model_data['history']
    name = model_data['model_name']
    epochs = list(range(1, len(history['train_acc']) + 1))
    
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4))
    
    # ===== Accuracy panel =====
    ax_acc.plot(epochs, history['train_acc'], '-', linewidth=1.8,
                color='#1f77b4', label='Train Accuracy')
    ax_acc.plot(epochs, history['val_acc'], '-', linewidth=1.8,
                color='#ff7f0e', label='Validation Accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.set_title(f'{name} Model Accuracy')
    ax_acc.legend(loc='lower right', frameon=True)
    ax_acc.set_ylim(0, 105)
    
    # ===== Loss panel =====
    ax_loss.plot(epochs, history['train_loss'], '-', linewidth=1.8,
                 color='#1f77b4', label='Train Loss')
    ax_loss.plot(epochs, history['val_loss'], '-', linewidth=1.8,
                 color='#ff7f0e', label='Validation Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title(f'{name} Model Loss')
    ax_loss.legend(loc='upper right', frameon=True)
    
    # Caption underneath like the reference paper
    fig.text(0.5, -0.02, name, ha='center', fontsize=12,
             fontweight='bold', fontstyle='italic')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  ✅ {name}: saved {output_path.name}")


def plot_combined_figure(all_models_data, output_path):
    """
    Create one combined figure (like Figure 3 in the reference paper).
    Layout: 4 rows × 2 cols = 4 models, each row has [accuracy | loss].
    """
    n_models = len(all_models_data)
    fig, axes = plt.subplots(n_models, 2, figsize=(11, 3.2 * n_models))
    
    for row, model_data in enumerate(all_models_data):
        history = model_data['history']
        name = model_data['model_name']
        epochs = list(range(1, len(history['train_acc']) + 1))
        
        ax_acc = axes[row, 0]
        ax_loss = axes[row, 1]
        
        # Accuracy
        ax_acc.plot(epochs, history['train_acc'], '-', linewidth=1.8,
                    color='#1f77b4', label='Train Accuracy')
        ax_acc.plot(epochs, history['val_acc'], '-', linewidth=1.8,
                    color='#ff7f0e', label='Validation Accuracy')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.set_title(f'{name} Model Accuracy')
        ax_acc.legend(loc='lower right', frameon=True)
        ax_acc.set_ylim(0, 105)
        
        # Loss
        ax_loss.plot(epochs, history['train_loss'], '-', linewidth=1.8,
                     color='#1f77b4', label='Train Loss')
        ax_loss.plot(epochs, history['val_loss'], '-', linewidth=1.8,
                     color='#ff7f0e', label='Validation Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'{name} Model Loss')
        ax_loss.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\n✅ Combined figure saved: {output_path}")


def main():
    setup_paper_style()
    
    print("=" * 70)
    print("CapraCare - Publication-Quality Training Curves")
    print("=" * 70)
    
    if not RESULTS_JSON.exists():
        print(f"❌ Results file not found: {RESULTS_JSON}")
        return
    
    with open(RESULTS_JSON, 'r') as f:
        data = json.load(f)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Reorder models per MODEL_ORDER
    models_by_name = {r['model_name']: r for r in data['results']}
    ordered_models = []
    for name in MODEL_ORDER:
        if name in models_by_name:
            ordered_models.append(models_by_name[name])
        else:
            print(f"⚠️  {name} not found in results")
    
    if not ordered_models:
        print("❌ No models to plot!")
        return
    
    # 1. Individual figures (one per model) - useful for separate placement
    print("\n[1/2] Generating individual model figures...")
    for model_data in ordered_models:
        safe_name = model_data['model_name'].replace('/', '_').replace(' ', '_')
        out_path = OUTPUT_DIR / f"curves_{safe_name}.png"
        plot_single_model(model_data, out_path)
    
    # 2. Combined figure (like Figure 3 in the reference paper)
    print("\n[2/2] Generating combined figure (paper-style)...")
    plot_combined_figure(
        ordered_models,
        OUTPUT_DIR / "combined_training_curves.png"
    )
    
    print("\n" + "=" * 70)
    print(f"✅ All figures saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 70)
    print("\nFigures generated:")
    print(f"  • {len(ordered_models)} individual model figures (curves_<model>.png)")
    print(f"  • 1 combined figure (combined_training_curves.png)")
    print("\nUse the combined figure for the paper, or individual ones if you")
    print("prefer to lay them out manually in your document.")


if __name__ == '__main__':
    main()