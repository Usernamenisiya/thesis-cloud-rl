"""
Generate Thesis Figures for Chapter 4
=====================================
Creates separate, high-quality figures as requested by adviser:
1. Figure 4.3.1: Overall Metrics Comparison (bar chart)
2. Figure 4.3.2: Thin Cloud Recall Comparison (bar chart)
3. Figure 4.3.3-4.3.5: Individual Patch Comparisons (enlarged, 4-panel each)

Run this in Google Colab with the data loaded.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Colab detection
IN_COLAB = ('google.colab' in sys.modules or 
            os.path.exists('/content') or 
            'COLAB_GPU' in os.environ)

if IN_COLAB:
    BASE_DIR = Path('/content/drive/MyDrive/Colab_Data')
    OUTPUT_DIR = Path('/content/drive/MyDrive/Colab_Data/thesis_figures')
else:
    BASE_DIR = Path('.')
    OUTPUT_DIR = Path('./results/thesis_figures')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FIGURE 4.3.1: OVERALL METRICS COMPARISON
# ============================================================================

def create_figure_4_3_1():
    """Create bar chart for Overall Metrics Comparison."""
    
    # Data from your results
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    cnn_values = [78.49, 81.20, 73.79, 72.03]
    ppo_values = [79.90, 80.00, 74.20, 72.79]
    dqn_values = [80.89, 79.70, 76.70, 73.38]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create bars
    bars1 = ax.bar(x - width, cnn_values, width, label='CNN Baseline', 
                   color='#3498db', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, ppo_values, width, label='PPO (720k steps)', 
                   color='#e74c3c', edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x + width, dqn_values, width, label='DQN (100k steps)', 
                   color='#2ecc71', edgecolor='black', linewidth=1.2)
    
    # Customize axes
    ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_title('Figure 4.3.1: Overall Performance Metrics Comparison\nCNN Baseline vs PPO vs DQN', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_ylim(65, 95)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save figure
    save_path = OUTPUT_DIR / 'Figure_4_3_1_Overall_Metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    
    plt.show()
    plt.close()


# ============================================================================
# FIGURE 4.3.2: THIN CLOUD RECALL COMPARISON
# ============================================================================

def create_figure_4_3_2():
    """Create focused bar chart for Thin Cloud Recall Comparison."""
    
    # Data from your results
    models = ['CNN Baseline', 'PPO\n(720k steps)', 'DQN\n(100k steps)']
    thin_recall = [63.28, 71.64, 77.00]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create bars
    bars = ax.bar(models, thin_recall, color=colors, edgecolor='black', linewidth=2, width=0.6)
    
    # Customize axes
    ax.set_ylabel('Thin Cloud Recall (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Detection Method', fontsize=14, fontweight='bold')
    ax.set_title('Figure 4.3.2: Thin Cloud Recall Comparison\n(Key Research Metric)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(50, 85)
    
    # Add value labels on bars
    for bar, val in zip(bars, thin_recall):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Add improvement annotations
    # PPO improvement
    ax.annotate('', xy=(1, 71.64), xytext=(0, 63.28),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(0.5, 67.5, '+8.36%', ha='center', fontsize=11, fontweight='bold', color='#e74c3c')
    
    # DQN improvement
    ax.annotate('', xy=(2, 77.00), xytext=(0, 63.28),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(1.0, 70.0, '+13.72%', ha='center', fontsize=11, fontweight='bold', color='#2ecc71')
    
    # Add horizontal line at baseline
    ax.axhline(y=63.28, color='#3498db', linestyle='--', linewidth=2, alpha=0.7, label='Baseline: 63.28%')
    
    # Add target line
    ax.axhline(y=77.00, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.7, label='Target Achieved: 77.00%')
    
    ax.legend(loc='lower right', fontsize=10)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Style x-axis labels
    ax.tick_params(axis='x', labelsize=12)
    
    plt.tight_layout()
    
    # Save figure
    save_path = OUTPUT_DIR / 'Figure_4_3_2_Thin_Cloud_Recall.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    
    plt.show()
    plt.close()


# ============================================================================
# INDIVIDUAL PATCH COMPARISON FIGURES (4.3.3, 4.3.4, 4.3.5)
# ============================================================================

def create_rgb_image(patch):
    """Create RGB image from patch for visualization."""
    rgb = patch[:, :, [2, 1, 0]]
    rgb = np.clip(rgb / 3000.0, 0, 1)
    rgb = np.power(rgb, 0.7)
    return rgb


def create_error_overlay(pred, gt, thin_mask=None):
    """Create color-coded error overlay.
    
    Colors:
    - Green: True Positive (correct cloud detection)
    - Red: False Negative (missed cloud)
    - Blue: False Positive (false alarm)
    - Cyan: Thin cloud correctly detected
    """
    overlay = np.zeros((*pred.shape, 3), dtype=np.float32)
    
    tp = (pred == 1) & (gt == 1)
    fn = (pred == 0) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    tn = (pred == 0) & (gt == 0)
    
    # Green for TP
    overlay[tp] = [0.2, 0.8, 0.2]
    
    # Red for FN (missed)
    overlay[fn] = [0.9, 0.2, 0.2]
    
    # Blue for FP (false alarm)
    overlay[fp] = [0.2, 0.4, 0.9]
    
    # Light gray for TN
    overlay[tn] = [0.85, 0.85, 0.85]
    
    # Highlight thin clouds detected correctly in cyan
    if thin_mask is not None:
        thin_tp = tp & thin_mask
        overlay[thin_tp] = [0.0, 0.9, 0.9]
    
    return overlay


def create_individual_patch_figure(patch, label, cnn_mask, dqn_mask, patch_num, 
                                   cnn_thin_recall, dqn_thin_recall, save_path):
    """Create 4-panel comparison for a single patch (enlarged view)."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    gt_binary = (label > 0).astype(np.uint8)
    thin_mask = (label == 2)
    
    # Panel 1: RGB Image
    rgb = create_rgb_image(patch)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('(a) Original RGB Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Panel 2: Ground Truth
    gt_display = np.zeros((*label.shape, 3))
    gt_display[label == 0] = [0.85, 0.85, 0.85]  # Clear - light gray
    gt_display[label == 1] = [0.8, 0.2, 0.2]      # Thick cloud - red
    gt_display[label == 2] = [0.0, 0.8, 0.8]      # Thin cloud - cyan
    gt_display[label == 3] = [0.3, 0.3, 0.3]      # Shadow - dark gray
    
    axes[0, 1].imshow(gt_display)
    axes[0, 1].set_title('(b) Ground Truth Labels', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Add legend for ground truth
    legend_elements = [
        mpatches.Patch(facecolor=[0.85, 0.85, 0.85], edgecolor='black', label='Clear Sky'),
        mpatches.Patch(facecolor=[0.8, 0.2, 0.2], edgecolor='black', label='Thick Cloud'),
        mpatches.Patch(facecolor=[0.0, 0.8, 0.8], edgecolor='black', label='Thin Cloud'),
        mpatches.Patch(facecolor=[0.3, 0.3, 0.3], edgecolor='black', label='Shadow'),
    ]
    axes[0, 1].legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # Panel 3: CNN Baseline (with error overlay)
    cnn_overlay = create_error_overlay(cnn_mask, gt_binary, thin_mask)
    axes[1, 0].imshow(cnn_overlay)
    axes[1, 0].set_title(f'(c) CNN Baseline\nThin Cloud Recall: {cnn_thin_recall:.1f}%', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Panel 4: DQN Refined (with error overlay)
    dqn_overlay = create_error_overlay(dqn_mask, gt_binary, thin_mask)
    axes[1, 1].imshow(dqn_overlay)
    axes[1, 1].set_title(f'(d) DQN Refined\nThin Cloud Recall: {dqn_thin_recall:.1f}%', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add common legend for error overlays
    error_legend = [
        mpatches.Patch(facecolor=[0.2, 0.8, 0.2], edgecolor='black', label='True Positive'),
        mpatches.Patch(facecolor=[0.9, 0.2, 0.2], edgecolor='black', label='False Negative (Missed)'),
        mpatches.Patch(facecolor=[0.2, 0.4, 0.9], edgecolor='black', label='False Positive'),
        mpatches.Patch(facecolor=[0.0, 0.9, 0.9], edgecolor='black', label='Thin Cloud Detected'),
    ]
    fig.legend(handles=error_legend, loc='lower center', ncol=4, fontsize=11, 
               bbox_to_anchor=(0.5, 0.02))
    
    # Main title
    fig.suptitle(f'Figure 4.3.{patch_num+2}: Patch #{patch_num} Detailed Comparison\n'
                 f'Improvement: +{dqn_thin_recall - cnn_thin_recall:.1f}% Thin Cloud Recovery',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    
    plt.show()
    plt.close()


def generate_patch_figures_from_data(patches, labels, cnn_masks, dqn_masks, 
                                     num_patches=3, selected_indices=None):
    """Generate individual patch comparison figures."""
    
    if selected_indices is None:
        # Select patches with good thin cloud examples
        selected_indices = [0, 1, 2]  # Adjust based on your data
    
    for idx, patch_idx in enumerate(selected_indices[:num_patches]):
        patch = patches[patch_idx]
        label = labels[patch_idx]
        cnn_mask = cnn_masks[patch_idx]
        dqn_mask = dqn_masks[patch_idx]
        
        # Calculate thin cloud recall for this patch
        thin_mask = (label == 2)
        thin_total = thin_mask.sum()
        
        if thin_total > 0:
            cnn_thin_recall = (cnn_mask[thin_mask].sum() / thin_total) * 100
            dqn_thin_recall = (dqn_mask[thin_mask].sum() / thin_total) * 100
        else:
            cnn_thin_recall = 0.0
            dqn_thin_recall = 0.0
        
        save_path = OUTPUT_DIR / f'Figure_4_3_{idx+3}_Patch_{idx+1}_Comparison.png'
        
        create_individual_patch_figure(
            patch, label, cnn_mask, dqn_mask,
            patch_num=idx+1,
            cnn_thin_recall=cnn_thin_recall,
            dqn_thin_recall=dqn_thin_recall,
            save_path=save_path
        )


# ============================================================================
# FIGURE FOR TRAINING EFFICIENCY COMPARISON
# ============================================================================

def create_training_efficiency_figure():
    """Create comparison figure for training efficiency (PPO vs DQN)."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Training steps comparison
    models = ['PPO', 'DQN']
    steps = [720000, 100000]
    colors = ['#e74c3c', '#2ecc71']
    
    bars1 = axes[0].bar(models, steps, color=colors, edgecolor='black', linewidth=2, width=0.5)
    axes[0].set_ylabel('Training Steps', fontsize=13, fontweight='bold')
    axes[0].set_title('(a) Training Steps Required', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 800000)
    
    # Add value labels
    for bar, val in zip(bars1, steps):
        axes[0].annotate(f'{val:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', fontsize=14, fontweight='bold')
    
    # Add efficiency annotation
    axes[0].annotate('7× fewer steps!', xy=(1, 100000), xytext=(0.5, 400000),
                    fontsize=12, fontweight='bold', color='#2ecc71',
                    arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2))
    
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Right panel: Performance achieved
    thin_recall = [71.64, 77.00]
    
    bars2 = axes[1].bar(models, thin_recall, color=colors, edgecolor='black', linewidth=2, width=0.5)
    axes[1].set_ylabel('Thin Cloud Recall (%)', fontsize=13, fontweight='bold')
    axes[1].set_title('(b) Performance Achieved', fontsize=14, fontweight='bold')
    axes[1].set_ylim(60, 85)
    
    # Add value labels
    for bar, val in zip(bars2, thin_recall):
        axes[1].annotate(f'{val:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', fontsize=14, fontweight='bold')
    
    # Add better performance annotation
    axes[1].annotate('+5.36% better!', xy=(1, 77.00), xytext=(0.5, 82),
                    fontsize=12, fontweight='bold', color='#2ecc71',
                    arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2))
    
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Main title
    fig.suptitle('Figure 4.2.3: Training Efficiency Comparison (PPO vs DQN)\n'
                 'DQN achieves better results with 7× fewer training steps',
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / 'Figure_4_2_3_Training_Efficiency.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {save_path}")
    
    plt.show()
    plt.close()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_bar_charts_only():
    """Generate only the bar chart figures (no data loading required)."""
    print("=" * 60)
    print("GENERATING THESIS FIGURES (Bar Charts)")
    print("=" * 60)
    
    print("\n📊 Creating Figure 4.3.1: Overall Metrics Comparison...")
    create_figure_4_3_1()
    
    print("\n📊 Creating Figure 4.3.2: Thin Cloud Recall Comparison...")
    create_figure_4_3_2()
    
    print("\n📊 Creating Figure 4.2.3: Training Efficiency Comparison...")
    create_training_efficiency_figure()
    
    print("\n" + "=" * 60)
    print("✅ BAR CHART FIGURES GENERATED SUCCESSFULLY!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("=" * 60)


def main():
    """Main function - run all figure generation."""
    print("=" * 60)
    print("THESIS FIGURE GENERATOR")
    print("=" * 60)
    
    # First generate bar charts (no data needed)
    generate_bar_charts_only()
    
    # For patch figures, we need to load data
    if IN_COLAB:
        print("\n" + "=" * 60)
        print("GENERATING PATCH COMPARISON FIGURES")
        print("=" * 60)
        print("\nLoading test data and models...")
        
        # Import required modules
        from visualize_algorithm_masks import load_test_data, apply_cnn_baseline
        from visualize_algorithm_masks import apply_dqn_model
        from stable_baselines3 import DQN
        
        # Load data
        patches, labels = load_test_data()
        
        # Load DQN model
        dqn_path = BASE_DIR / 'dqn_thin_cloud' / 'dqn_thin_cloud_100000_steps.zip'
        dqn_model = DQN.load(str(dqn_path))
        
        # Generate masks for selected patches
        selected = [0, 1, 4]  # Patches with good thin cloud examples
        
        cnn_masks = []
        dqn_masks = []
        
        for idx in selected:
            print(f"  Processing patch {idx+1}...")
            cnn_mask, cnn_prob = apply_cnn_baseline(patches[idx])
            dqn_mask = apply_dqn_model(patches[idx], dqn_model, cnn_prob, labels[idx])
            cnn_masks.append(cnn_mask)
            dqn_masks.append(dqn_mask)
        
        # Generate individual patch figures
        generate_patch_figures_from_data(
            [patches[i] for i in selected],
            [labels[i] for i in selected],
            cnn_masks,
            dqn_masks,
            num_patches=3
        )
        
        print("\n" + "=" * 60)
        print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
        print("=" * 60)
    else:
        print("\n⚠️  Patch comparison figures require running in Google Colab")
        print("    with access to the test data and trained models.")
        print("\n    To generate patch figures:")
        print("    1. Upload this script to Colab")
        print("    2. Mount Google Drive")
        print("    3. Run main()")


if __name__ == "__main__":
    # For quick testing, just generate bar charts
    generate_bar_charts_only()
    
    # For full generation including patches, uncomment:
    # main()
