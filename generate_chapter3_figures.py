"""
Generate Chapter 3 Methodology Figures
- Figure 3.2.1: Conceptual Framework
- Figure 3.3.1: System Architecture Pipeline

Creates colorful, publication-quality diagrams.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os

# Create output directory
output_dir = 'results/thesis_figures'
os.makedirs(output_dir, exist_ok=True)

# Color palette - modern, colorful scheme
COLORS = {
    'input': '#3498db',        # Blue - input/data
    'cnn': '#e74c3c',          # Red - CNN baseline
    'ppo': '#2ecc71',          # Green - PPO
    'dqn': '#9b59b6',          # Purple - DQN
    'feature': '#f39c12',      # Orange - feature extraction
    'output': '#1abc9c',       # Teal - output
    'metrics': '#34495e',      # Dark gray - metrics
    'arrow': '#7f8c8d',        # Gray - arrows
    'text': '#2c3e50',         # Dark text
    'white': '#ffffff',
    'light_bg': '#ecf0f1',     # Light background
}


def draw_rounded_box(ax, x, y, width, height, text, color, fontsize=10, 
                     text_color='white', bold=False, multiline=False):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color, edgecolor='white', linewidth=2,
        alpha=0.95
    )
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    if multiline and '\n' in text:
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                color=text_color, fontweight=weight, linespacing=1.3)
    else:
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                color=text_color, fontweight=weight)
    return box


def draw_arrow(ax, start, end, color=None, style='simple', connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    if color is None:
        color = COLORS['arrow']
    
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=15,
        lw=2,
        color=color,
        connectionstyle=connectionstyle
    )
    ax.add_patch(arrow)
    return arrow


def generate_figure_3_2_1():
    """
    Figure 3.2.1: Conceptual Framework
    Shows the experimental workflow with three detection branches.
    """
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Conceptual Framework: RL-Enhanced Cloud Detection Pipeline',
            ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['text'])
    
    # === ROW 1: Input Dataset ===
    draw_rounded_box(ax, 7, 8.3, 3.5, 0.7, 'Sentinel-2 Multispectral\nDataset (CloudSEN12)', 
                     COLORS['input'], fontsize=10, bold=True, multiline=True)
    
    # === ROW 2: Three Detection Branches ===
    # Arrows from dataset to three branches
    draw_arrow(ax, (5.5, 7.9), (3, 7.2))
    draw_arrow(ax, (7, 7.9), (7, 7.2))
    draw_arrow(ax, (8.5, 7.9), (11, 7.2))
    
    # Three branches
    draw_rounded_box(ax, 3, 6.8, 2.8, 0.7, 's2cloudless CNN\n(Baseline)', 
                     COLORS['cnn'], fontsize=10, bold=True, multiline=True)
    draw_rounded_box(ax, 7, 6.8, 2.8, 0.7, 'PPO-RL\nRefinement', 
                     COLORS['ppo'], fontsize=10, bold=True, multiline=True)
    draw_rounded_box(ax, 11, 6.8, 2.8, 0.7, 'DQN-RL\nRefinement', 
                     COLORS['dqn'], fontsize=10, bold=True, multiline=True)
    
    # === ROW 3: Spectral Feature Extraction ===
    # Arrows down
    draw_arrow(ax, (3, 6.4), (3, 5.7))
    draw_arrow(ax, (7, 6.4), (7, 5.7))
    draw_arrow(ax, (11, 6.4), (11, 5.7))
    
    # Feature extraction box (spans across)
    feature_box = FancyBboxPatch(
        (1.5, 4.9), 11, 0.8,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=COLORS['feature'], edgecolor='white', linewidth=2,
        alpha=0.95
    )
    ax.add_patch(feature_box)
    ax.text(7, 5.3, 'Spectral Feature Extraction', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Feature details below
    features = ['Cirrus Band (B10)', 'Blue/Red Ratio', 'NDSI']
    for i, feat in enumerate(features):
        x_pos = 3 + i * 4
        ax.text(x_pos, 4.5, feat, ha='center', va='center', fontsize=9, 
                color=COLORS['feature'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['feature'], linewidth=1.5))
    
    # === ROW 4: Refinement Phase ===
    draw_arrow(ax, (7, 4.1), (7, 3.5))
    
    refinement_box = FancyBboxPatch(
        (2.5, 2.5), 9, 1,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=COLORS['light_bg'], edgecolor=COLORS['text'], linewidth=2,
        alpha=0.95
    )
    ax.add_patch(refinement_box)
    ax.text(7, 3.2, 'Refinement Phase', 
            ha='center', va='center', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax.text(7, 2.75, 'Local Threshold Adjustment  •  Thin Cloud Confidence Boost', 
            ha='center', va='center', fontsize=9, color=COLORS['text'])
    
    # === ROW 5: Output Masks ===
    draw_arrow(ax, (4, 2.4), (3, 1.8))
    draw_arrow(ax, (7, 2.4), (7, 1.8))
    draw_arrow(ax, (10, 2.4), (11, 1.8))
    
    draw_rounded_box(ax, 3, 1.4, 2.5, 0.7, 'Baseline Mask\n(Static)', 
                     COLORS['cnn'], fontsize=9, bold=False, multiline=True)
    draw_rounded_box(ax, 7, 1.4, 2.5, 0.7, 'RL-Enhanced\n(PPO)', 
                     COLORS['ppo'], fontsize=9, bold=False, multiline=True)
    draw_rounded_box(ax, 11, 1.4, 2.5, 0.7, 'RL-Enhanced\n(DQN)', 
                     COLORS['dqn'], fontsize=9, bold=False, multiline=True)
    
    # === ROW 6: Metrics Evaluation ===
    draw_arrow(ax, (3, 1.0), (5.5, 0.5))
    draw_arrow(ax, (7, 1.0), (7, 0.5))
    draw_arrow(ax, (11, 1.0), (8.5, 0.5))
    
    metrics_box = FancyBboxPatch(
        (4, -0.2), 6, 0.7,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=COLORS['metrics'], edgecolor='white', linewidth=2,
        alpha=0.95
    )
    ax.add_patch(metrics_box)
    ax.text(7, 0.15, 'Performance Metrics: IoU  •  F1-Score  •  Recall  •  Precision', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['input'], label='Input Data'),
        mpatches.Patch(facecolor=COLORS['cnn'], label='CNN Baseline'),
        mpatches.Patch(facecolor=COLORS['ppo'], label='PPO Agent'),
        mpatches.Patch(facecolor=COLORS['dqn'], label='DQN Agent'),
        mpatches.Patch(facecolor=COLORS['feature'], label='Feature Extraction'),
        mpatches.Patch(facecolor=COLORS['metrics'], label='Evaluation'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, 
              framealpha=0.9, ncol=3, bbox_to_anchor=(0.05, -0.02))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_3_2_1_Conceptual_Framework.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {filepath}")
    return filepath


def generate_figure_3_3_1():
    """
    Figure 3.3.1: System Architecture Pipeline
    Shows the linear modular workflow from input to output.
    """
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(8, 7.5, 'RL-Enhanced Cloud Detection System Architecture',
            ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['text'])
    
    # === MAIN PIPELINE (centered) ===
    y_main = 5.0
    box_width = 2.8
    box_height = 1.2
    
    # Stage 1: Input Dataset
    draw_rounded_box(ax, 2, y_main, box_width, box_height, 
                     'INPUT\n-----------\nSentinel-2\nDataset', 
                     COLORS['input'], fontsize=10, bold=True, multiline=True)
    
    # Arrow 1→2
    draw_arrow(ax, (3.5, y_main), (4.7, y_main), color=COLORS['arrow'])
    
    # Stage 2: CNN Module
    draw_rounded_box(ax, 6, y_main, box_width, box_height,
                     'CNN MODULE\n-----------\ns2cloudless\nProbability Map',
                     COLORS['cnn'], fontsize=10, bold=True, multiline=True)
    
    # Arrow 2→3
    draw_arrow(ax, (7.5, y_main), (8.7, y_main), color=COLORS['arrow'])
    
    # Stage 3: RL Refinement
    # Create a compound box for RL
    rl_x = 10
    rl_box = FancyBboxPatch(
        (rl_x - box_width/2 - 0.3, y_main - box_height/2 - 0.4), 
        box_width + 0.6, box_height + 0.8,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor=COLORS['light_bg'], edgecolor=COLORS['text'], linewidth=2,
        alpha=0.9
    )
    ax.add_patch(rl_box)
    ax.text(rl_x, y_main + 0.8, 'RL REFINEMENT', 
            ha='center', va='center', fontsize=10, fontweight='bold', color=COLORS['text'])
    
    # PPO and DQN sub-boxes
    draw_rounded_box(ax, rl_x - 0.8, y_main - 0.1, 1.4, 0.7, 'PPO\nAgent',
                     COLORS['ppo'], fontsize=9, bold=True, multiline=True)
    draw_rounded_box(ax, rl_x + 0.8, y_main - 0.1, 1.4, 0.7, 'DQN\nAgent',
                     COLORS['dqn'], fontsize=9, bold=True, multiline=True)
    
    # Arrow 3→4
    draw_arrow(ax, (11.8, y_main), (13.0, y_main), color=COLORS['arrow'])
    
    # Stage 4: Output
    draw_rounded_box(ax, 14.2, y_main, box_width, box_height,
                     'OUTPUT\n-----------\nEnhanced\nCloud Mask',
                     COLORS['output'], fontsize=10, bold=True, multiline=True)
    
    # === DETAIL SECTION (below main pipeline) ===
    y_detail = 2.2
    
    # Section headers
    sections = [
        (2, 'Spectral Bands', ['• 13 Sentinel-2 bands', '• 512×512 patches', '• CloudSEN12 format']),
        (6, 'Baseline Processing', ['• Static threshold (0.5)', '• Probability mapping', '• Initial cloud mask']),
        (10, 'Dynamic Refinement', ['• Threshold adjustment', '• Thin cloud boost', '• Patch-based (64×64)']),
        (14.2, 'Final Product', ['• High-fidelity mask', '• Improved thin cloud', '• Geospatial-ready']),
    ]
    
    for x, title, details in sections:
        # Detail box
        detail_box = FancyBboxPatch(
            (x - 1.3, y_detail - 0.9), 2.6, 1.6,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor='white', edgecolor=COLORS['text'], linewidth=1.5,
            alpha=0.95
        )
        ax.add_patch(detail_box)
        
        # Arrow from main to detail
        draw_arrow(ax, (x, y_main - box_height/2 - 0.1), (x, y_detail + 0.8), 
                   color=COLORS['arrow'])
        
        # Title
        ax.text(x, y_detail + 0.45, title, ha='center', va='center', 
                fontsize=9, fontweight='bold', color=COLORS['text'])
        
        # Details
        for i, detail in enumerate(details):
            ax.text(x, y_detail + 0.1 - i*0.35, detail, ha='center', va='center',
                    fontsize=8, color=COLORS['text'])
    
    # === OBSERVATION SPACE BOX (bottom) ===
    obs_y = 0.5
    obs_box = FancyBboxPatch(
        (3.5, obs_y - 0.3), 9, 0.6,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=COLORS['feature'], edgecolor='white', linewidth=2,
        alpha=0.95
    )
    ax.add_patch(obs_box)
    ax.text(8, obs_y, 'Observation Space: CNN Stats  •  Cirrus Band (B10)  •  Blue/Red Ratio  •  NDSI  •  Spatial Context',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Arrow from observation to RL
    draw_arrow(ax, (10, obs_y + 0.4), (10, y_detail - 1.0), color=COLORS['feature'])
    
    # Add phase labels at bottom
    phase_y = -0.3
    phases = [
        (2, 'Phase 1', COLORS['input']),
        (6, 'Phase 1', COLORS['cnn']),
        (10, 'Phase 2', COLORS['ppo']),
        (14.2, 'Phase 3', COLORS['output']),
    ]
    for x, label, color in phases:
        ax.text(x, phase_y, label, ha='center', va='center', fontsize=8,
                fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor=color, linewidth=1.5))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_3_3_1_System_Architecture.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {filepath}")
    return filepath


def main():
    print("="*60)
    print("Generating Chapter 3 Methodology Figures")
    print("="*60)
    
    # Generate both figures
    fig1 = generate_figure_3_2_1()
    fig2 = generate_figure_3_3_1()
    
    print("\n" + "="*60)
    print("✅ All figures generated successfully!")
    print(f"📂 Output directory: {output_dir}")
    print("="*60)
    
    return [fig1, fig2]


if __name__ == "__main__":
    main()
