import sys
import os
import numpy as np
import plotly.graph_objects as go
import webbrowser
from scipy.ndimage import zoom, gaussian_filter

# Adjust system path to allow imports from the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.neural_network import neuralNetwork

# =================================================================
# 1. SETUP & TRAINING PHASE
# =================================================================
n = neuralNetwork(784, 200, 10, 0.1)
fashion_labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

print("[SYSTEM] Phase 1: Initializing Model Training...")
with open("data/fashion/fashion-mnist_train.csv", 'r') as f:
    training_data = f.readlines()

for e in range(5):
    print(f" > Processing Epoch {e+1}/5...")
    for record in training_data[1:]:
        vals = record.split(',')
        inputs = (np.asarray(vals[1:], dtype=float) / 255.0 * 0.99) + 0.01
        targets = np.zeros(10) + 0.01
        targets[int(vals[0])] = 0.99
        n.train(inputs, targets)

# =================================================================
# 2. TESTING & MATRIX INFERENCE
# =================================================================
conf_matrix = np.zeros((10, 10))
print("[SYSTEM] Phase 2: Testing & Generating Topology...")
with open("data/fashion/fashion-mnist_test.csv", 'r') as f:
    test_data = f.readlines()

for record in test_data[1:]:
    vals = record.split(',')
    target = int(vals[0])
    inputs = (np.asarray(vals[1:], dtype=float) / 255.0 * 0.99) + 0.01
    prediction = np.argmax(n.query(inputs))
    conf_matrix[target, prediction] += 1

accuracy = (sum(conf_matrix.diagonal()) / 10000) * 100

# =================================================================
# 3. INTERPOLATION ENGINE (Latent Space Smoothing)
# =================================================================
print("[SYSTEM] Interpolating Latent Space for Topological Relief...")
scaling = 10
z_upscaled = zoom(conf_matrix, scaling)
z_smooth = gaussian_filter(z_upscaled, sigma=2.2)

x_smooth = np.linspace(0, 9, z_smooth.shape[1])
y_smooth = np.linspace(0, 9, z_smooth.shape[0])

# =================================================================
# 4. HIGH-FIDELITY INTERACTIVE RENDER (Plotly Engine)
# =================================================================
matrix_green = '#00ff41'
bg_color = '#0d0d0d'
floor_level = -150  

# --- Robust Custom Data Array for Hover Tooltips ---
# We create a 3D array to securely pass [Predicted Label, Actual Label, Density] to the JS engine
custom_data = np.empty((z_smooth.shape[0], z_smooth.shape[1], 3), dtype=object)

for i in range(z_smooth.shape[0]):
    for j in range(z_smooth.shape[1]):
        # Safely constrain indices between 0 and 9
        actual_idx = max(0, min(9, int(round(y_smooth[i]))))
        pred_idx = max(0, min(9, int(round(x_smooth[j]))))
        
        # Populate the data pipeline
        custom_data[i, j, 0] = fashion_labels[pred_idx]
        custom_data[i, j, 1] = fashion_labels[actual_idx]
        custom_data[i, j, 2] = f"{z_smooth[i, j]:.1f}"

# LAYER 1: Main topographical surface
surface_layer = go.Surface(
    z=z_smooth, x=x_smooth, y=y_smooth,
    colorscale='Jet',
    customdata=custom_data,  # Inject the secure data matrix here
    contours=dict(
        x=dict(show=True, color='rgba(0,0,0,0.5)', width=1), 
        y=dict(show=True, color='rgba(0,0,0,0.5)', width=1)
    ),
    lighting=dict(ambient=0.7, diffuse=0.7, specular=0.1, roughness=0.5),
    colorbar=dict(
        title=dict(text='Inference Density', font=dict(color=matrix_green, family='monospace', size=13)),
        tickfont=dict(color=matrix_green, family='monospace'),
        thickness=20, len=0.6, x=0.95
    ),
    # Map the custom data channels directly into the HTML template
    hovertemplate=(
        "<b>Predicted:</b> %{customdata[0]}<br>"
        "<b>Actual:</b> %{customdata[1]}<br>"
        "<b>Density:</b> %{customdata[2]}<extra></extra>"
    )
)

# LAYER 2: Heatmap projection on the floor
floor_z = np.full_like(z_smooth, floor_level)
heatmap_floor_layer = go.Surface(
    z=floor_z, x=x_smooth, y=y_smooth,
    surfacecolor=z_smooth, 
    colorscale='Jet',
    showscale=False,       
    opacity=0.6,           
    hoverinfo='skip'       
)

fig = go.Figure(data=[surface_layer, heatmap_floor_layer])

legend_html = (
    "&gt;&gt; TOPOLOGY LOGIC_<br>"
    "Peaks&nbsp;&nbsp;&nbsp;= High Confidence<br>"
    "Valleys = Low Error<br>"
    "Red&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= Logic Concentration<br>"
    "Blue&nbsp;&nbsp;&nbsp;&nbsp;= Statistical Sparsity"
)

fig.update_layout(
    title=dict(
        text=f'3D TOPOLOGICAL ANALYSIS OF MODEL INFERENCE<br><span style="font-size:16px">Calculated Accuracy: {accuracy:.2f}%</span>',
        font=dict(color=matrix_green, family='monospace', size=22),
        x=0.5, y=0.95
    ),
    template="plotly_dark",
    paper_bgcolor=bg_color,
    scene=dict(
        xaxis=dict(title='Predicted Category', ticktext=fashion_labels, tickvals=list(range(10)), color=matrix_green, gridcolor='#1a4d1a', backgroundcolor=bg_color),
        yaxis=dict(title='True Category', ticktext=fashion_labels, tickvals=list(range(10)), color=matrix_green, gridcolor='#1a4d1a', backgroundcolor=bg_color),
        zaxis=dict(title='Classification Frequency', color=matrix_green, gridcolor='#1a4d1a', backgroundcolor=bg_color, range=[floor_level, np.max(z_smooth)+50]),
        
        # --- NEW DRAMATICAL ANGLE ---
        camera=dict(eye=dict(x=2.2, y=-1.5, z=1.2)) 
    ),
    annotations=[dict(
        text=legend_html, align='left',
        x=0.02, y=0.95, xref='paper', yref='paper', showarrow=False, 
        font=dict(color=matrix_green, family='monospace', size=12),
        bgcolor='rgba(13,13,13,0.8)', bordercolor=matrix_green, borderwidth=1, borderpad=8
    )],
    margin=dict(l=0, r=0, t=80, b=0)
)

# Export interactive HTML
html_output_path = "results/fashion_interactive_topology.html"
fig.write_html(html_output_path, include_plotlyjs='cdn')

# Export static high-res PNG fallback for GitHub README
png_output_path = "results/fashion_full_topology.png"
fig.write_image(png_output_path, width=1200, height=800, scale=2)

webbrowser.open('file://' + os.path.abspath(html_output_path))
print(f"[SUCCESS] High-fidelity Interactive Matrix generated: {html_output_path}")
print(f"[SUCCESS] Static PNG Preview generated: {png_output_path}")
