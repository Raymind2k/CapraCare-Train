"""
CapraCare - Publication-Quality Confusion Matrices
Generates per-model confusion matrices for the research paper.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# Configuration
# ============================================================
RESULTS_JSON = Path("runs/classification/comparison/comparison_results.json")
OUTPUT_DIR = Path("runs/classification/paper_figures")

# Display order
MODEL_ORDER = ["MobileNetV2", "DenseNet121", "ResNet50", "EfficientNet-B0"]

# Short labels for cleaner axes (full names are too long)
CLASS_LABEL_MAP = {
    "parasitic_mange": "Mange",
    "parasitic_myiasis": "Myiasis",
    "skin_infection_abscess": "Abscess",
    "skin_infection_dermatitis": "Dermatitis",
    "skin_infection_dermatophilosis": "Dermatophilosis",
    "viral_orf": "Orf",
    "viral_ppr": "PPR",
    "wounds_footrot": "Footrot",
}


# ============================================================
# Paper-style settings
# ============================================================
def setup_paper_style():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def compute_confusion_matrix(preds, labels, num_classes):
    """Compute confusion matrix: rows = true, cols = predicted."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1
    return cm


def plot_single_confusion_matrix(cm, class_labels, model_name, test_acc,
                                  output_path, normalize=False):
    """Plot a single confusion matrix with annotations."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    if normalize:
        # Normalize per row (per true class)
        with np.errstate(all='ignore'):
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_display = np.nan_to_num(cm_display)
        fmt = '.2f'
        vmin, vmax = 0, 1
        cbar_label = 'Normalized Count'
    else:
        cm_display = cm
        fmt = 'd'
        vmin, vmax = 0, cm.max()
        cbar_label = 'Count'
    
    im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues',
                   vmin=vmin, vmax=vmax)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)
    
    # Annotate each cell
    threshold = (vmax + vmin) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm_display[i, j] > threshold else 'black'
            ax.text(j, i, format(cm_display[i, j], fmt),
                    ha='center', va='center', color=color, fontsize=9)
    
    # Labels
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_yticklabels(class_labels)
    ax.set_xlabel('Predicted Class', fontsize=11)
    ax.set_ylabel('True Class', fontsize=11)
    ax.set_title(f'{model_name}  |  Test Accuracy: {test_acc:.2f}%',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  ✅ {model_name}: saved {output_path.name}")


def plot_combined_confusion_matrices(all_models_data, class_labels,
                                      output_path, normalize=False):
    """
    Plot all 4 confusion matrices in a 2x2 grid for the paper.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, model_data in enumerate(all_models_data):
        ax = axes[idx]
        preds = model_data['test_preds']
        labels = model_data['test_labels']
        num_classes = len(class_labels)
        
        cm = compute_confusion_matrix(preds, labels, num_classes)
        
        if normalize:
            with np.errstate(all='ignore'):
                cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
                cm_display = np.nan_to_num(cm_display)
            fmt = '.2f'
            vmin, vmax = 0, 1
        else:
            cm_display = cm
            fmt = 'd'
            vmin, vmax = 0, cm.max()
        
        im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues',
                       vmin=vmin, vmax=vmax)
        
        # Annotate cells
        threshold = (vmax + vmin) / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm_display[i, j] > threshold else 'black'
                ax.text(j, i, format(cm_display[i, j], fmt),
                        ha='center', va='center', color=color, fontsize=8)
        
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(class_labels, fontsize=8)
        ax.set_xlabel('Predicted Class', fontsize=10)
        ax.set_ylabel('True Class', fontsize=10)
        ax.set_title(f"{model_data['model_name']}  |  Test Acc: {model_data['test_acc']:.2f}%",
                     fontsize=11, fontweight='bold')
        
        # Colorbar for each subplot
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\n✅ Combined figure saved: {output_path}")


def print_text_confusion_matrices(all_models_data, class_labels):
    """Also print confusion matrices in text form (for the paper's Table 3 equivalent)."""
    print("\n" + "=" * 70)
    print("CONFUSION MATRICES (text form for paper Table 3)")
    print("=" * 70)
    
    for model_data in all_models_data:
        preds = model_data['test_preds']
        labels = model_data['test_labels']
        cm = compute_confusion_matrix(preds, labels, len(class_labels))
        
        print(f"\n{model_data['model_name']} (Test Accuracy: {model_data['test_acc']:.2f}%)")
        print("-" * 70)
        # Header
        header = 'True / Pred'
        print(f"{header:<16}", end='')
        for c in class_labels:
            print(f"{c[:8]:>10}", end='')
        print()
        # Rows
        for i, c in enumerate(class_labels):
            print(f"{c:<16}", end='')
            for j in range(len(class_labels)):
                print(f"{cm[i][j]:>10d}", end='')
            print()


def main():
    setup_paper_style()
    
    print("=" * 70)
    print("CapraCare - Publication-Quality Confusion Matrices")
    print("=" * 70)
    
    if not RESULTS_JSON.exists():
        print(f"❌ Results file not found: {RESULTS_JSON}")
        return
    
    with open(RESULTS_JSON, 'r') as f:
        data = json.load(f)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get class labels (short versions for plotting)
    class_names_full = data['class_names']
    class_labels = [CLASS_LABEL_MAP.get(c, c) for c in class_names_full]
    
    # Reorder models
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
    
    # 1. Individual confusion matrices (raw counts)
    print("\n[1/4] Individual confusion matrices (raw counts)...")
    for model_data in ordered_models:
        preds = model_data['test_preds']
        labels = model_data['test_labels']
        cm = compute_confusion_matrix(preds, labels, len(class_labels))
        safe_name = model_data['model_name'].replace('/', '_').replace(' ', '_')
        out_path = OUTPUT_DIR / f"confmat_{safe_name}.png"
        plot_single_confusion_matrix(
            cm, class_labels, model_data['model_name'],
            model_data['test_acc'], out_path, normalize=False
        )
    
    # 2. Individual confusion matrices (normalized)
    print("\n[2/4] Individual confusion matrices (normalized)...")
    for model_data in ordered_models:
        preds = model_data['test_preds']
        labels = model_data['test_labels']
        cm = compute_confusion_matrix(preds, labels, len(class_labels))
        safe_name = model_data['model_name'].replace('/', '_').replace(' ', '_')
        out_path = OUTPUT_DIR / f"confmat_{safe_name}_normalized.png"
        plot_single_confusion_matrix(
            cm, class_labels, model_data['model_name'],
            model_data['test_acc'], out_path, normalize=True
        )
    
    # 3. Combined 2x2 figure (raw counts) - for paper
    print("\n[3/4] Combined 2x2 figure (raw counts)...")
    plot_combined_confusion_matrices(
        ordered_models, class_labels,
        OUTPUT_DIR / "combined_confusion_matrices.png",
        normalize=False
    )
    
    # 4. Combined 2x2 figure (normalized) - alternative
    print("\n[4/4] Combined 2x2 figure (normalized)...")
    plot_combined_confusion_matrices(
        ordered_models, class_labels,
        OUTPUT_DIR / "combined_confusion_matrices_normalized.png",
        normalize=True
    )
    
    # Also print text version
    print_text_confusion_matrices(ordered_models, class_labels)
    
    print("\n" + "=" * 70)
    print(f"✅ All confusion matrices saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 70)
    print("\nFigures generated:")
    print(f"  • 4 individual matrices (counts) - confmat_<model>.png")
    print(f"  • 4 individual matrices (normalized) - confmat_<model>_normalized.png")
    print(f"  • 1 combined 2x2 figure (counts) - combined_confusion_matrices.png")
    print(f"  • 1 combined 2x2 figure (normalized) - combined_confusion_matrices_normalized.png")
    print("\nUse the COMBINED figure in your paper. Counts version is most")
    print("informative; normalized version is useful for comparing per-class")
    print("accuracy across models with different class sizes.")


if __name__ == '__main__':
    main()