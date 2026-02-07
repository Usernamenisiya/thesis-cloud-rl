"""
Generate Chapter 3 Methodology Figures (Version 3)
- Compact layout with minimal spacing
- Large, clear patch visualizations
- All text highly readable without zooming
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os

# Create output directory
output_dir = 'results/thesis_figures'
os.makedirs(output_dir, exist_ok=True)

# Color palette
COLORS = {
    'input': '#3498db',        # Blue
    'cnn': '#e74c3c',          # Red
    'ppo': '#2ecc71',          # Green
    'dqn': '#9b59b6',          # Purple
    'feature': '#f39c12',      # Orange
    'output': '#1abc9c',       # Teal
    'metrics': '#34495e',      # Dark gray
    'arrow': '#555555',        # Dark arrow
    'text': '#2c3e50',
    'light_bg': '#ecf0f1',
}


def load_sample_data():
    """Load sample satellite data for visualizations."""
    try:
        import rasterio
        
        image_path = 'data/sentinel2_image.tif'
        mask_path = 'data/ground_truth.tif'
        
        if os.path.exists(image_path) and os.path.exists(mask_path):
            with rasterio.open(image_path) as src:
                img = src.read()
                if img.shape[0] >= 4:
                    rgb = np.stack([img[3], img[2], img[1]], axis=-1)
                else:
                    rgb = np.stack([img[0], img[1], img[2]], axis=-1)
                
                rgb = rgb.astype(np.float32)
                for i in range(3):
                    band = rgb[:,:,i]
                    p2, p98 = np.percentile(band[band > 0], [2, 98]) if np.any(band > 0) else (0, 1)
                    rgb[:,:,i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
            
            with rasterio.open(mask_path) as src:
                gt = src.read(1)
            
            print(f"✅ Loaded sample data")
            return rgb, gt
        else:
            return generate_synthetic_data()
            
    except:
        return generate_synthetic_data()


def generate_synthetic_data():
    """Generate synthetic satellite-like data."""
    np.random.seed(42)
    size = 128
    
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    rgb[:,:,0] = 0.3 + np.random.rand(size, size) * 0.1
    rgb[:,:,1] = 0.4 + np.random.rand(size, size) * 0.1
    rgb[:,:,2] = 0.2 + np.random.rand(size, size) * 0.1
    
    y, x = np.ogrid[:size, :size]
    thick_cloud = ((x - 30)**2 + (y - 40)**2 < 400)
    rgb[thick_cloud] = [0.95, 0.95, 0.95]
    
    thin_cloud_mask = ((x - 90)**2 + (y - 80)**2 < 600)
    rgb[thin_cloud_mask] = rgb[thin_cloud_mask] * 0.6 + np.array([0.9, 0.9, 0.95]) * 0.4
    
    gt = np.zeros((size, size), dtype=np.uint8)
    gt[thick_cloud] = 1
    gt[thin_cloud_mask] = 2
    
    return np.clip(rgb, 0, 1), gt


def create_patches(rgb, gt, patch_size=80):
    """Extract patches for visualization."""
    h, w = rgb.shape[:2]
    
    thin_cloud_pixels = np.where(gt == 2)
    if len(thin_cloud_pixels[0]) > 0:
        cy = int(np.mean(thin_cloud_pixels[0]))
        cx = int(np.mean(thin_cloud_pixels[1]))
    else:
        cy, cx = h // 2, w // 2
    
    y1 = max(0, cy - patch_size // 2)
    y2 = min(h, y1 + patch_size)
    x1 = max(0, cx - patch_size // 2)
    x2 = min(w, x1 + patch_size)
    
    rgb_patch = rgb[y1:y2, x1:x2]
    gt_patch = gt[y1:y2, x1:x2]
    
    # Baseline mask (misses thin clouds)
    baseline_mask = np.zeros_like(gt_patch, dtype=np.float32)
    baseline_mask[gt_patch == 1] = 1.0
    baseline_mask[gt_patch == 2] = 0.25
    
    # Refined mask (detects all clouds)
    refined_mask = np.zeros_like(gt_patch, dtype=np.float32)
    refined_mask[gt_patch >= 1] = 1.0
    
    return rgb_patch, gt_patch, baseline_mask, refined_mask


def draw_box(ax, x, y, w, h, text, color, fontsize=14, text_color='white'):
    """Draw a compact rounded box with text."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor='white', linewidth=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', linespacing=1.1)


def draw_arrow(ax, start, end, color='#555555'):
    """Draw arrow."""
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=20,
                            lw=2.5, color=color)
    ax.add_patch(arrow)


def add_image(ax, img, x, y, zoom=1.0, border_color=None, title=None):
    """Add image at coordinates with optional border and title."""
    if len(img.shape) == 2:
        img = plt.cm.gray(img)[:,:,:3]
    
    imagebox = OffsetImage(img, zoom=zoom)
    
    if border_color:
        ab = AnnotationBbox(imagebox, (x, y), frameon=True,
                           bboxprops=dict(boxstyle='round,pad=0.05', 
                                         facecolor='white', edgecolor=border_color, linewidth=3))
    else:
        ab = AnnotationBbox(imagebox, (x, y), frameon=True,
                           bboxprops=dict(boxstyle='round,pad=0.05', 
                                         facecolor='white', edgecolor='#333', linewidth=2))
    ax.add_artist(ab)
    
    if title:
        ax.text(x, y - 1.1, title, ha='center', va='top', fontsize=12, 
                fontweight='bold', color=COLORS['text'])


def generate_figure_3_2_1(rgb_patch, gt_patch, baseline_mask, refined_mask):
    """Figure 3.2.1: Conceptual Framework - Compact version with large images."""
    
    fig, ax = plt.subplots(figsize=(14, 9), facecolor='white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(7, 8.7, 'Conceptual Framework: RL-Enhanced Cloud Detection', 
            ha='center', fontsize=20, fontweight='bold', color=COLORS['text'])
    
    # === ROW 1: Input ===
    draw_box(ax, 7, 7.8, 5, 0.7, 'Sentinel-2 Dataset (CloudSEN12)', COLORS['input'], fontsize=14)
    
    # === ROW 2: Three branches ===
    draw_arrow(ax, (4.8, 7.4), (2.5, 6.8))
    draw_arrow(ax, (7, 7.4), (7, 6.8))
    draw_arrow(ax, (9.2, 7.4), (11.5, 6.8))
    
    draw_box(ax, 2.5, 6.4, 3, 0.7, 's2cloudless (Baseline)', COLORS['cnn'], fontsize=13)
    draw_box(ax, 7, 6.4, 3, 0.7, 'PPO Refinement', COLORS['ppo'], fontsize=13)
    draw_box(ax, 11.5, 6.4, 3, 0.7, 'DQN Refinement', COLORS['dqn'], fontsize=13)
    
    # === ROW 3: Feature Extraction ===
    draw_arrow(ax, (2.5, 6.0), (2.5, 5.5))
    draw_arrow(ax, (7, 6.0), (7, 5.5))
    draw_arrow(ax, (11.5, 6.0), (11.5, 5.5))
    
    # Feature box spanning across
    feat_box = FancyBboxPatch((1, 4.9), 12, 0.6, boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=COLORS['feature'], edgecolor='white', linewidth=2)
    ax.add_patch(feat_box)
    ax.text(7, 5.2, 'Feature Extraction:  Cirrus Band (B10)  •  Blue/Red Ratio  •  NDSI',
            ha='center', fontsize=13, fontweight='bold', color='white')
    
    # === ROW 4: Refinement Phase ===
    draw_arrow(ax, (7, 4.85), (7, 4.4))
    
    ref_box = FancyBboxPatch((2, 3.85), 10, 0.5, boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=COLORS['light_bg'], edgecolor=COLORS['text'], linewidth=2)
    ax.add_patch(ref_box)
    ax.text(7, 4.1, 'Refinement: Threshold Adjustment + Thin Cloud Boost',
            ha='center', fontsize=13, fontweight='bold', color=COLORS['text'])
    
    # === ROW 5: Output Images (LARGE) ===
    draw_arrow(ax, (3.5, 3.8), (2.5, 3.3))
    draw_arrow(ax, (7, 3.8), (7, 3.3))
    draw_arrow(ax, (10.5, 3.8), (11.5, 3.3))
    
    # Create colored masks
    baseline_vis = np.stack([baseline_mask, baseline_mask*0.3, baseline_mask*0.3], axis=-1)
    ppo_vis = np.stack([refined_mask*0.2, refined_mask, refined_mask*0.4], axis=-1)
    dqn_vis = np.stack([refined_mask*0.6, refined_mask*0.3, refined_mask], axis=-1)
    
    # Large images with titles
    add_image(ax, baseline_vis, 2.5, 2.0, zoom=1.0, border_color=COLORS['cnn'], title='Baseline Mask')
    add_image(ax, ppo_vis, 7, 2.0, zoom=1.0, border_color=COLORS['ppo'], title='PPO Enhanced')
    add_image(ax, dqn_vis, 11.5, 2.0, zoom=1.0, border_color=COLORS['dqn'], title='DQN Enhanced')
    
    # === ROW 6: Metrics ===
    metrics_box = FancyBboxPatch((3, 0.15), 8, 0.5, boxstyle="round,pad=0.02,rounding_size=0.1",
                                  facecolor=COLORS['metrics'], edgecolor='white', linewidth=2)
    ax.add_patch(metrics_box)
    ax.text(7, 0.4, 'Metrics: IoU  •  F1-Score  •  Recall  •  Precision',
            ha='center', fontsize=13, fontweight='bold', color='white')
    
    # Arrows to metrics
    draw_arrow(ax, (2.5, 0.85), (4.5, 0.7))
    draw_arrow(ax, (7, 0.85), (7, 0.7))
    draw_arrow(ax, (11.5, 0.85), (9.5, 0.7))
    
    plt.tight_layout(pad=0.5)
    filepath = os.path.join(output_dir, 'Figure_3_2_1_Conceptual_Framework.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {filepath}")


def generate_figure_3_3_1(rgb_patch, gt_patch, baseline_mask, refined_mask):
    """Figure 3.3.1: System Architecture - Compact with large images."""
    
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(8, 7.7, 'System Architecture: RL-Enhanced Cloud Detection Pipeline',
            ha='center', fontsize=20, fontweight='bold', color=COLORS['text'])
    
    # === MAIN PIPELINE BOXES ===
    y_box = 6.2
    
    # Box 1: Input
    draw_box(ax, 2, y_box, 2.8, 1.0, 'INPUT\nSentinel-2', COLORS['input'], fontsize=13)
    
    # Box 2: CNN
    draw_arrow(ax, (3.5, y_box), (4.5, y_box))
    draw_box(ax, 6, y_box, 2.8, 1.0, 'CNN MODULE\ns2cloudless', COLORS['cnn'], fontsize=13)
    
    # Box 3: RL Refinement (compound)
    draw_arrow(ax, (7.5, y_box), (8.5, y_box))
    
    rl_box = FancyBboxPatch((8.6, y_box - 0.7), 3.4, 1.4, 
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=COLORS['light_bg'], edgecolor=COLORS['text'], linewidth=2)
    ax.add_patch(rl_box)
    ax.text(10.3, y_box + 0.4, 'RL REFINEMENT', ha='center', fontsize=12, 
            fontweight='bold', color=COLORS['text'])
    
    draw_box(ax, 9.5, y_box - 0.15, 1.2, 0.6, 'PPO', COLORS['ppo'], fontsize=12)
    draw_box(ax, 11.1, y_box - 0.15, 1.2, 0.6, 'DQN', COLORS['dqn'], fontsize=12)
    
    # Box 4: Output
    draw_arrow(ax, (12.1, y_box), (13.1, y_box))
    draw_box(ax, 14.3, y_box, 2.8, 1.0, 'OUTPUT\nEnhanced Mask', COLORS['output'], fontsize=13)
    
    # === LARGE IMAGE VISUALIZATIONS ===
    y_img = 3.2
    
    # Create visualizations
    baseline_vis = np.stack([baseline_mask, baseline_mask*0.3, baseline_mask*0.3], axis=-1)
    refined_vis = np.stack([refined_mask*0.2, refined_mask*0.8, refined_mask*0.6], axis=-1)
    
    # Images with borders
    add_image(ax, rgb_patch, 2, y_img, zoom=1.2, border_color=COLORS['input'], title='Raw Image')
    add_image(ax, baseline_vis, 6, y_img, zoom=1.2, border_color=COLORS['cnn'], title='Baseline Mask')
    add_image(ax, refined_vis, 14.3, y_img, zoom=1.2, border_color=COLORS['output'], title='Refined Mask')
    
    # Arrows connecting images to boxes
    draw_arrow(ax, (2, y_box - 0.55), (2, y_img + 1.2))
    draw_arrow(ax, (6, y_box - 0.55), (6, y_img + 1.2))
    draw_arrow(ax, (14.3, y_box - 0.55), (14.3, y_img + 1.2))
    
    # === OBSERVATION SPACE ===
    obs_box = FancyBboxPatch((3.5, 0.9), 9, 0.6, boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=COLORS['feature'], edgecolor='white', linewidth=2)
    ax.add_patch(obs_box)
    ax.text(8, 1.2, 'Observation: CNN Stats • Cirrus (B10) • Blue/Red • NDSI • Position',
            ha='center', fontsize=12, fontweight='bold', color='white')
    
    # Arrow from observation to RL
    draw_arrow(ax, (10.3, 1.55), (10.3, y_box - 0.75), color=COLORS['feature'])
    
    # === PHASE LABELS ===
    phases = [(2, 'PHASE 1', COLORS['input']), 
              (6, 'PHASE 1', COLORS['cnn']),
              (10.3, 'PHASE 2', COLORS['ppo']), 
              (14.3, 'PHASE 3', COLORS['output'])]
    
    for x, label, color in phases:
        ax.text(x, 0.35, label, ha='center', fontsize=11, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=color, linewidth=2))
    
    plt.tight_layout(pad=0.5)
    filepath = os.path.join(output_dir, 'Figure_3_3_1_System_Architecture.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {filepath}")


def main():
    print("="*60)
    print("Generating Chapter 3 Figures (v3 - Compact & Large)")
    print("="*60)
    
    rgb, gt = load_sample_data()
    rgb_patch, gt_patch, baseline_mask, refined_mask = create_patches(rgb, gt, patch_size=80)
    
    print(f"📊 Patch size: {rgb_patch.shape}")
    
    generate_figure_3_2_1(rgb_patch, gt_patch, baseline_mask, refined_mask)
    generate_figure_3_3_1(rgb_patch, gt_patch, baseline_mask, refined_mask)
    
    print("\n" + "="*60)
    print("✅ All figures generated!")
    print(f"📂 Location: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
