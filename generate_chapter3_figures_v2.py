"""
Generate Chapter 3 Methodology Figures (Version 2)
- Figure 3.2.1: Conceptual Framework with embedded visualizations
- Figure 3.3.1: System Architecture Pipeline with embedded visualizations

Features:
- Larger text for better readability
- Actual satellite imagery visualizations embedded
- Colorful, publication-quality diagrams
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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


def load_sample_data():
    """Load sample satellite data for visualizations."""
    try:
        import rasterio
        
        # Try to load the local sample data
        image_path = 'data/sentinel2_image.tif'
        mask_path = 'data/ground_truth.tif'
        
        if os.path.exists(image_path) and os.path.exists(mask_path):
            # Load RGB bands
            with rasterio.open(image_path) as src:
                # Sentinel-2: B4=Red, B3=Green, B2=Blue (bands 4, 3, 2 in 1-indexed)
                # Assume bands are in order, take first 3 as RGB-like
                img = src.read()
                if img.shape[0] >= 4:
                    # Use B4, B3, B2 for RGB
                    rgb = np.stack([img[3], img[2], img[1]], axis=-1)  # R, G, B
                else:
                    rgb = np.stack([img[0], img[1], img[2]], axis=-1)
                
                # Normalize to 0-1
                rgb = rgb.astype(np.float32)
                for i in range(3):
                    band = rgb[:,:,i]
                    p2, p98 = np.percentile(band[band > 0], [2, 98]) if np.any(band > 0) else (0, 1)
                    rgb[:,:,i] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
            
            # Load ground truth
            with rasterio.open(mask_path) as src:
                gt = src.read(1)
            
            print(f"✅ Loaded sample data: {image_path}")
            return rgb, gt
        else:
            print("⚠️ Sample data not found, generating synthetic data")
            return generate_synthetic_data()
            
    except ImportError:
        print("⚠️ Rasterio not available, generating synthetic data")
        return generate_synthetic_data()
    except Exception as e:
        print(f"⚠️ Error loading data: {e}, generating synthetic data")
        return generate_synthetic_data()


def generate_synthetic_data():
    """Generate synthetic satellite-like data for visualization."""
    np.random.seed(42)
    
    # Create a 128x128 synthetic image
    size = 128
    
    # Create RGB with some cloud-like features
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    
    # Background (land-like green/brown)
    rgb[:,:,0] = 0.3 + np.random.rand(size, size) * 0.1  # R
    rgb[:,:,1] = 0.4 + np.random.rand(size, size) * 0.1  # G
    rgb[:,:,2] = 0.2 + np.random.rand(size, size) * 0.1  # B
    
    # Add some cloud features (white, bright)
    # Thick cloud region
    y, x = np.ogrid[:size, :size]
    thick_cloud = ((x - 30)**2 + (y - 40)**2 < 400)
    rgb[thick_cloud] = [0.95, 0.95, 0.95]
    
    # Thin cloud region (semi-transparent, harder to detect)
    thin_cloud_mask = ((x - 90)**2 + (y - 80)**2 < 600)
    rgb[thin_cloud_mask] = rgb[thin_cloud_mask] * 0.6 + np.array([0.9, 0.9, 0.95]) * 0.4
    
    # Another thin wispy cloud
    wispy = ((x - 60)**2 + (y - 20)**2 < 200)
    rgb[wispy] = rgb[wispy] * 0.7 + np.array([0.85, 0.85, 0.9]) * 0.3
    
    # Ground truth mask (0=clear, 1=thick cloud, 2=thin cloud)
    gt = np.zeros((size, size), dtype=np.uint8)
    gt[thick_cloud] = 1  # Thick cloud
    gt[thin_cloud_mask] = 2  # Thin cloud
    gt[wispy] = 2  # Thin cloud
    
    return np.clip(rgb, 0, 1), gt


def create_visualization_patches(rgb, gt, patch_size=64):
    """Extract a patch and create baseline/refined mask visualizations."""
    h, w = rgb.shape[:2]
    
    # Find a region with thin clouds
    # Look for area with thin clouds (class 2)
    thin_cloud_pixels = np.where(gt == 2)
    if len(thin_cloud_pixels[0]) > 0:
        # Center on thin cloud region
        cy = int(np.mean(thin_cloud_pixels[0]))
        cx = int(np.mean(thin_cloud_pixels[1]))
    else:
        cy, cx = h // 2, w // 2
    
    # Extract patch
    y1 = max(0, cy - patch_size // 2)
    y2 = min(h, y1 + patch_size)
    x1 = max(0, cx - patch_size // 2)
    x2 = min(w, x1 + patch_size)
    
    # Adjust if patch is too small
    if y2 - y1 < patch_size:
        y1 = max(0, y2 - patch_size)
    if x2 - x1 < patch_size:
        x1 = max(0, x2 - patch_size)
    
    rgb_patch = rgb[y1:y2, x1:x2]
    gt_patch = gt[y1:y2, x1:x2]
    
    # Resize if needed
    if rgb_patch.shape[0] != patch_size or rgb_patch.shape[1] != patch_size:
        from scipy.ndimage import zoom
        scale_y = patch_size / rgb_patch.shape[0]
        scale_x = patch_size / rgb_patch.shape[1]
        rgb_patch = zoom(rgb_patch, (scale_y, scale_x, 1), order=1)
        gt_patch = zoom(gt_patch, (scale_y, scale_x), order=0).astype(np.uint8)
    
    # Create baseline mask (misses thin clouds - only detects class 1)
    baseline_mask = np.zeros_like(gt_patch, dtype=np.float32)
    baseline_mask[gt_patch == 1] = 1.0  # Only thick clouds
    baseline_mask[gt_patch == 2] = 0.3  # Partially detect thin clouds
    
    # Create refined mask (better thin cloud detection)
    refined_mask = np.zeros_like(gt_patch, dtype=np.float32)
    refined_mask[gt_patch >= 1] = 1.0  # Detect all clouds
    
    return rgb_patch, gt_patch, baseline_mask, refined_mask


def add_image_to_plot(ax, img, x, y, zoom_factor=0.15, title=None, title_fontsize=11):
    """Add an image at specific coordinates on the plot."""
    if len(img.shape) == 2:
        # Grayscale - create a colored version
        img_display = plt.cm.Reds(img)[:,:,:3]  # Use red colormap for masks
    else:
        img_display = img
    
    # Create the image
    imagebox = OffsetImage(img_display, zoom=zoom_factor)
    imagebox.image.axes = ax
    
    ab = AnnotationBbox(imagebox, (x, y),
                        frameon=True,
                        bboxprops=dict(boxstyle='round,pad=0.1', 
                                      facecolor='white', 
                                      edgecolor=COLORS['text'],
                                      linewidth=2))
    ax.add_artist(ab)
    
    if title:
        ax.text(x, y - 0.9, title, ha='center', va='top', 
                fontsize=title_fontsize, fontweight='bold', color=COLORS['text'])


def draw_rounded_box(ax, x, y, width, height, text, color, fontsize=12, 
                     text_color='white', bold=False, multiline=False):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color, edgecolor='white', linewidth=2.5,
        alpha=0.95
    )
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    if multiline and '\n' in text:
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                color=text_color, fontweight=weight, linespacing=1.2)
    else:
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                color=text_color, fontweight=weight)
    return box


def draw_arrow(ax, start, end, color=None, style='simple', lw=2.5):
    """Draw an arrow between two points."""
    if color is None:
        color = COLORS['arrow']
    
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=18,
        lw=lw,
        color=color,
    )
    ax.add_patch(arrow)
    return arrow


def generate_figure_3_2_1(rgb_patch, gt_patch, baseline_mask, refined_mask):
    """
    Figure 3.2.1: Conceptual Framework
    Shows the experimental workflow with embedded visualizations.
    """
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'Conceptual Framework: RL-Enhanced Cloud Detection Pipeline',
            ha='center', va='center', fontsize=18, fontweight='bold', color=COLORS['text'])
    
    # === ROW 1: Input Dataset with sample image ===
    draw_rounded_box(ax, 8, 10.2, 4.5, 1.0, 'Sentinel-2 Multispectral\nDataset (CloudSEN12)', 
                     COLORS['input'], fontsize=13, bold=True, multiline=True)
    
    # Add sample RGB patch next to input
    add_image_to_plot(ax, rgb_patch, 12.5, 10.2, zoom_factor=0.5, title='Sample Patch', title_fontsize=10)
    
    # === ROW 2: Three Detection Branches ===
    # Arrows from dataset to three branches
    draw_arrow(ax, (5.8, 9.6), (3.5, 8.5), lw=2.5)
    draw_arrow(ax, (8, 9.6), (8, 8.5), lw=2.5)
    draw_arrow(ax, (10.2, 9.6), (12.5, 8.5), lw=2.5)
    
    # Three branches
    draw_rounded_box(ax, 3.5, 8.0, 3.2, 0.9, 's2cloudless CNN\n(Baseline)', 
                     COLORS['cnn'], fontsize=12, bold=True, multiline=True)
    draw_rounded_box(ax, 8, 8.0, 3.2, 0.9, 'PPO-RL\nRefinement', 
                     COLORS['ppo'], fontsize=12, bold=True, multiline=True)
    draw_rounded_box(ax, 12.5, 8.0, 3.2, 0.9, 'DQN-RL\nRefinement', 
                     COLORS['dqn'], fontsize=12, bold=True, multiline=True)
    
    # === ROW 3: Spectral Feature Extraction ===
    draw_arrow(ax, (3.5, 7.5), (3.5, 6.8), lw=2.5)
    draw_arrow(ax, (8, 7.5), (8, 6.8), lw=2.5)
    draw_arrow(ax, (12.5, 7.5), (12.5, 6.8), lw=2.5)
    
    # Feature extraction box (spans across)
    feature_box = FancyBboxPatch(
        (1.5, 5.8), 13, 1.0,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=COLORS['feature'], edgecolor='white', linewidth=2.5,
        alpha=0.95
    )
    ax.add_patch(feature_box)
    ax.text(8, 6.3, 'Spectral Feature Extraction', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Feature details below
    features = ['Cirrus Band (B10)', 'Blue/Red Ratio', 'NDSI']
    for i, feat in enumerate(features):
        x_pos = 3.5 + i * 4.5
        ax.text(x_pos, 5.3, feat, ha='center', va='center', fontsize=11, 
                color=COLORS['feature'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=COLORS['feature'], linewidth=2))
    
    # === ROW 4: Refinement Phase ===
    draw_arrow(ax, (8, 4.9), (8, 4.3), lw=2.5)
    
    refinement_box = FancyBboxPatch(
        (2.5, 3.0), 11, 1.3,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=COLORS['light_bg'], edgecolor=COLORS['text'], linewidth=2.5,
        alpha=0.95
    )
    ax.add_patch(refinement_box)
    ax.text(8, 3.9, 'Refinement Phase', 
            ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.text(8, 3.35, 'Local Threshold Adjustment  •  Thin Cloud Confidence Boost', 
            ha='center', va='center', fontsize=11, color=COLORS['text'])
    
    # === ROW 5: Output Masks with visualizations ===
    draw_arrow(ax, (4.5, 2.9), (3.5, 2.3), lw=2.5)
    draw_arrow(ax, (8, 2.9), (8, 2.3), lw=2.5)
    draw_arrow(ax, (11.5, 2.9), (12.5, 2.3), lw=2.5)
    
    # Output boxes (smaller, to make room for images)
    draw_rounded_box(ax, 3.5, 1.9, 2.8, 0.7, 'Baseline Mask', 
                     COLORS['cnn'], fontsize=11, bold=True, multiline=True)
    draw_rounded_box(ax, 8, 1.9, 2.8, 0.7, 'RL-Enhanced (PPO)', 
                     COLORS['ppo'], fontsize=11, bold=True, multiline=True)
    draw_rounded_box(ax, 12.5, 1.9, 2.8, 0.7, 'RL-Enhanced (DQN)', 
                     COLORS['dqn'], fontsize=11, bold=True, multiline=True)
    
    # Add mask visualizations below output boxes
    # Create colorized masks
    baseline_vis = np.zeros((*baseline_mask.shape, 3))
    baseline_vis[:,:,0] = baseline_mask  # Red channel
    
    refined_vis = np.zeros((*refined_mask.shape, 3))
    refined_vis[:,:,1] = refined_mask  # Green channel for PPO
    
    refined_vis_dqn = np.zeros((*refined_mask.shape, 3))
    refined_vis_dqn[:,:,0] = refined_mask * 0.6  # Purple-ish for DQN
    refined_vis_dqn[:,:,2] = refined_mask
    
    add_image_to_plot(ax, baseline_vis, 3.5, 0.7, zoom_factor=0.4, title='', title_fontsize=9)
    add_image_to_plot(ax, refined_vis, 8, 0.7, zoom_factor=0.4, title='', title_fontsize=9)
    add_image_to_plot(ax, refined_vis_dqn, 12.5, 0.7, zoom_factor=0.4, title='', title_fontsize=9)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['input'], label='Input Data'),
        mpatches.Patch(facecolor=COLORS['cnn'], label='CNN Baseline'),
        mpatches.Patch(facecolor=COLORS['ppo'], label='PPO Agent'),
        mpatches.Patch(facecolor=COLORS['dqn'], label='DQN Agent'),
        mpatches.Patch(facecolor=COLORS['feature'], label='Feature Extraction'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10, 
              framealpha=0.9, ncol=5, bbox_to_anchor=(0.02, -0.02))
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_3_2_1_Conceptual_Framework.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {filepath}")
    return filepath


def generate_figure_3_3_1(rgb_patch, gt_patch, baseline_mask, refined_mask):
    """
    Figure 3.3.1: System Architecture Pipeline
    Shows the linear modular workflow with embedded visualizations.
    """
    fig, ax = plt.subplots(figsize=(18, 10), facecolor='white')
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(9, 9.5, 'RL-Enhanced Cloud Detection System Architecture',
            ha='center', va='center', fontsize=18, fontweight='bold', color=COLORS['text'])
    
    # === MAIN PIPELINE (centered at y=6.5) ===
    y_main = 6.5
    box_width = 3.2
    box_height = 1.4
    
    # Stage 1: Input Dataset
    draw_rounded_box(ax, 2.5, y_main, box_width, box_height, 
                     'INPUT\n\nSentinel-2\nDataset', 
                     COLORS['input'], fontsize=12, bold=True, multiline=True)
    
    # Add RGB image below input
    add_image_to_plot(ax, rgb_patch, 2.5, y_main - 1.8, zoom_factor=0.5, 
                      title='Raw Image', title_fontsize=10)
    
    # Arrow 1→2
    draw_arrow(ax, (4.2, y_main), (5.3, y_main), lw=3)
    
    # Stage 2: CNN Module
    draw_rounded_box(ax, 6.8, y_main, box_width, box_height,
                     'CNN MODULE\n\ns2cloudless\nProbability Map',
                     COLORS['cnn'], fontsize=12, bold=True, multiline=True)
    
    # Add baseline mask below CNN
    baseline_vis = np.zeros((*baseline_mask.shape, 3))
    baseline_vis[:,:,0] = baseline_mask
    baseline_vis[:,:,1] = baseline_mask * 0.3
    baseline_vis[:,:,2] = baseline_mask * 0.3
    add_image_to_plot(ax, baseline_vis, 6.8, y_main - 1.8, zoom_factor=0.5, 
                      title='Baseline Mask', title_fontsize=10)
    
    # Arrow 2→3
    draw_arrow(ax, (8.5, y_main), (9.6, y_main), lw=3)
    
    # Stage 3: RL Refinement (compound box)
    rl_x = 11.5
    rl_box = FancyBboxPatch(
        (rl_x - box_width/2 - 0.4, y_main - box_height/2 - 0.5), 
        box_width + 0.8, box_height + 1.0,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor=COLORS['light_bg'], edgecolor=COLORS['text'], linewidth=2.5,
        alpha=0.9
    )
    ax.add_patch(rl_box)
    ax.text(rl_x, y_main + 0.95, 'RL REFINEMENT', 
            ha='center', va='center', fontsize=13, fontweight='bold', color=COLORS['text'])
    
    # PPO and DQN sub-boxes
    draw_rounded_box(ax, rl_x - 0.9, y_main - 0.1, 1.6, 0.9, 'PPO\nAgent',
                     COLORS['ppo'], fontsize=11, bold=True, multiline=True)
    draw_rounded_box(ax, rl_x + 0.9, y_main - 0.1, 1.6, 0.9, 'DQN\nAgent',
                     COLORS['dqn'], fontsize=11, bold=True, multiline=True)
    
    # Arrow 3→4
    draw_arrow(ax, (13.5, y_main), (14.6, y_main), lw=3)
    
    # Stage 4: Output
    draw_rounded_box(ax, 16, y_main, box_width, box_height,
                     'OUTPUT\n\nEnhanced\nCloud Mask',
                     COLORS['output'], fontsize=12, bold=True, multiline=True)
    
    # Add refined mask below output
    refined_vis = np.zeros((*refined_mask.shape, 3))
    refined_vis[:,:,0] = refined_mask * 0.2
    refined_vis[:,:,1] = refined_mask * 0.8
    refined_vis[:,:,2] = refined_mask * 0.6
    add_image_to_plot(ax, refined_vis, 16, y_main - 1.8, zoom_factor=0.5, 
                      title='Refined Mask', title_fontsize=10)
    
    # === OBSERVATION SPACE BOX (bottom) ===
    obs_y = 2.3
    obs_box = FancyBboxPatch(
        (3.5, obs_y - 0.4), 11, 0.8,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=COLORS['feature'], edgecolor='white', linewidth=2.5,
        alpha=0.95
    )
    ax.add_patch(obs_box)
    ax.text(9, obs_y, 'Observation Space: CNN Stats  •  Cirrus Band (B10)  •  Blue/Red Ratio  •  NDSI  •  Spatial Context',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # Arrow from observation to RL
    draw_arrow(ax, (11.5, obs_y + 0.5), (11.5, y_main - 1.3), color=COLORS['feature'], lw=2.5)
    
    # === PHASE LABELS ===
    phase_y = 1.2
    phases = [
        (2.5, 'PHASE 1', COLORS['input']),
        (6.8, 'PHASE 1', COLORS['cnn']),
        (11.5, 'PHASE 2', COLORS['ppo']),
        (16, 'PHASE 3', COLORS['output']),
    ]
    for x, label, color in phases:
        ax.text(x, phase_y, label, ha='center', va='center', fontsize=11,
                fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=color, linewidth=2))
    
    # Phase descriptions
    descriptions = [
        (2.5, 'Data Input'),
        (6.8, 'Baseline Detection'),
        (11.5, 'RL Training'),
        (16, 'Evaluation'),
    ]
    for x, desc in descriptions:
        ax.text(x, phase_y - 0.5, desc, ha='center', va='center', 
                fontsize=10, color=COLORS['text'])
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'Figure_3_3_1_System_Architecture.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ Saved: {filepath}")
    return filepath


def main():
    print("="*60)
    print("Generating Chapter 3 Methodology Figures (v2)")
    print("With larger text and embedded visualizations")
    print("="*60)
    
    # Load or generate sample data
    rgb, gt = load_sample_data()
    
    # Create visualization patches
    rgb_patch, gt_patch, baseline_mask, refined_mask = create_visualization_patches(rgb, gt)
    
    print(f"📊 Patch size: {rgb_patch.shape}")
    print(f"📊 Unique GT values: {np.unique(gt_patch)}")
    
    # Generate figures
    fig1 = generate_figure_3_2_1(rgb_patch, gt_patch, baseline_mask, refined_mask)
    fig2 = generate_figure_3_3_1(rgb_patch, gt_patch, baseline_mask, refined_mask)
    
    print("\n" + "="*60)
    print("✅ All figures generated successfully!")
    print(f"📂 Output directory: {output_dir}")
    print("="*60)
    
    return [fig1, fig2]


if __name__ == "__main__":
    main()
