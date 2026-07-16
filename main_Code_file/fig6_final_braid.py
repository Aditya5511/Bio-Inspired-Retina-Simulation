import numpy as np
import matplotlib.pyplot as plt

# Aesthetics matching the paper
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

fig, axes = plt.subplots(1, 4, figsize=(16, 7.5))
bit_widths = [18, 16, 14, 10]

# Continuous sweep domain up to 0.21 and back
x_up = np.linspace(0.0, 0.21, 5000)
x_down = np.linspace(0.21, 0.0, 5000)
x_full = np.concatenate([x_up, x_down])

for i, bits in enumerate(bit_widths):
    # Fixed point scale mapping
    frac_bits = bits - 2 
    scale = 2.0 ** frac_bits
    
    if bits > 10:
        # 18, 16, and 14-bit strictly preserve full dynamic range linearly
        y_ideal = x_full
    else:
        # Strictly mapped exact coordinates from the original paper's 10-bit plot
        top_sat = 34.0 / 256.0  # ~0.1328
        shelf = 13.0 / 256.0    # ~0.0508
        
        # Upper curve (Leftmost boundary)
        y_ideal_up = np.clip(x_up, 0, top_sat)
        
        # Lower curve (Rightmost boundary) mapped exactly to the paper's coordinates
        y_ideal_down = np.zeros_like(x_down)
        for j, x in enumerate(x_down):
            if x > 0.1928:
                # Loop closes early at the top; both paths merge flat here
                y_ideal_down[j] = top_sat
            elif x > 0.1108:
                # Upper return diagonal
                y_ideal_down[j] = x - 0.06
            elif x > 0.0658:
                # The exact mid-level shelf
                y_ideal_down[j] = shelf
            elif x > 0.015:
                # Lower return diagonal
                y_ideal_down[j] = x - 0.015
            else:
                # Bottom flat zero
                y_ideal_down[j] = 0.0
                
        # Merge trajectories for continuous plotting
        y_ideal = np.concatenate([y_ideal_up, y_ideal_down])

    # Quantization to generate the mathematically exact staircases
    y_approx = np.floor(y_ideal * scale) / scale
    
    # Plotting
    axes[i].plot(x_full, y_approx, color='magenta', linewidth=1.5)
    
    # Enforce exact paper domain/range limits
    axes[i].set_xlim([-0.02, 0.25])
    axes[i].set_ylim([-0.02, 0.23])
    
    # Enforce exact paper tick marks
    axes[i].set_xticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
    axes[i].set_yticks([0.00, 0.05, 0.10, 0.15, 0.20])
    
    # Enforce exact paper labeling
    axes[i].set_xlabel('Full precision', fontsize=14, labelpad=10)
    axes[i].set_ylabel(f'{bits}-bit approximation', fontsize=14, labelpad=10)
    
    # Lock precise bounding box aspect ratio to match tall panels
    axes[i].set_box_aspect(2.2)

plt.tight_layout(pad=3.0)
plt.savefig('exact_paper_match.png', dpi=300)
plt.show()