import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter
from typing import List

def save_3d_topology(conf_matrix: np.ndarray, labels: List[str], accuracy: float, filename: str) -> None:
    """
    Generates a high-end 3D topological relief of the Confusion Matrix
    in the "Matrix Rainbow" visual identity.
    """
    plt.style.use('dark_background')
    
    # The Matrix visual identity
    matrix_green = '#00ff41'
    bg_color = '#0d0d0d'

    # Upscale and Smooth for high-end "Plastic" rainbow look
    scaling = 10
    z_upscaled = zoom(conf_matrix, scaling)
    z_smooth = gaussian_filter(z_upscaled, sigma=2.2)

    x = np.linspace(0, 9, z_smooth.shape[1])
    y = np.linspace(0, 9, z_smooth.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(16, 11), facecolor=bg_color)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg_color)

    # Main Surface Rendering ('jet' creates the classic rainbow effect)
    surf = ax.plot_surface(X, Y, z_smooth, cmap='jet',
                           edgecolor='black', linewidth=0.1, antialiased=True,
                           rstride=1, cstride=1, alpha=0.9)

    # Dynamic Shadow Contours (Topographical lines)
    # Automatically calculates the bottom floor based on data max height
    z_max = z_smooth.max()
    shadow_offset = - (z_max * 0.2) if z_max > 0 else -1
    ax.contour(X, Y, z_smooth, zdir='z', offset=shadow_offset, cmap='jet', levels=25, alpha=0.4)
    ax.set_zlim(shadow_offset, z_max)

    # Labeling & Axis Calibration
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    
    # Applying Matrix Green to all text and axes
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10, color=matrix_green, family='monospace')
    ax.set_yticklabels(labels, fontsize=10, color=matrix_green, family='monospace')
    
    ax.xaxis.set_tick_params(colors=matrix_green)
    ax.yaxis.set_tick_params(colors=matrix_green)
    ax.zaxis.set_tick_params(colors=matrix_green)
    
    ax.xaxis.line.set_color(matrix_green)
    ax.yaxis.line.set_color(matrix_green)
    ax.zaxis.line.set_color(matrix_green)

    # Precise Axis Titles
    ax.set_xlabel('Predicted Category (Output)', fontsize=12, fontweight='bold', labelpad=15, color=matrix_green, family='monospace')
    ax.set_ylabel('True Category (Actual)', fontsize=12, fontweight='bold', labelpad=15, color=matrix_green, family='monospace')
    ax.set_zlabel('Classification Frequency', fontsize=12, fontweight='bold', labelpad=15, color=matrix_green, family='monospace')
    
    plt.title(f"3D TOPOLOGICAL ANALYSIS OF MODEL INFERENCE\nCalculated Accuracy: {accuracy:.2f}%", 
              fontsize=18, fontweight='bold', color=matrix_green, family='monospace', pad=30)

    # Gradient/Temperature Explanation (Colorbar)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Inference Density (Low to High)', fontsize=11, fontweight='bold', color=matrix_green, family='monospace')
    cbar.ax.yaxis.set_tick_params(color=matrix_green, labelcolor=matrix_green)

    # Explanatory Legend Box
    textstr = '\n'.join((
        r'>> TOPOLOGY LOGIC_ ',
        r'Peaks   = High Confidence',
        r'Valleys = Low Error',
        r'Red     = Logic Concentration',
        r'Blue    = Statistical Sparsity'
    ))
    ax.text2D(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11, color=matrix_green, family='monospace',
              verticalalignment='top', bbox=dict(boxstyle='square,pad=0.5', facecolor=bg_color, edgecolor=matrix_green, alpha=0.8))

    # Clean UI
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='#1a4d1a', linestyle='--', linewidth=0.5) # Faint green grid

    # Optimal perspective
    ax.view_init(elev=22, azim=45)
    
    output_path = f"results/{filename}"
    plt.savefig(output_path, dpi=300, facecolor=bg_color, bbox_inches='tight')
    print(f"[SYSTEM] Visual inference exported to: {output_path}")
    
    # Opens the interactive window
    plt.show()