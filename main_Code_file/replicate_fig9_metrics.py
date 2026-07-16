import numpy as np
import matplotlib.pyplot as plt
from stimulus_generator import generate_chirp_stimulus
from opl_layer import simulate_opl_1d
from bipolar_layer import simulate_bipolar_1d
from ganglion_layer import simulate_ganglion_1d

# Use clean, professional sans-serif typography
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def compute_variance_explained(sw, hw):
    """Calculates the true Ratio of Variance Explained (%)"""
    residual_var = np.var(sw - hw)
    sw_var = np.var(sw)
    return (1.0 - (residual_var / sw_var)) * 100.0 if sw_var > 0 else 0.0

def run_figure9_replication():
    # 1. Generate the standard 2-second chirp stimulus using actual time vectors
    t_array, stim = generate_chirp_stimulus()
    dt_ms = (t_array[1] - t_array[0]) * 1000.0 # Calculate time step in milliseconds
    
    # 2. Simulate Software Pipelines (is_quantized=False)
    opl_sw = simulate_opl_1d(t_array, stim, is_quantized=False)
    bip_sw = simulate_bipolar_1d(t_array, opl_sw, is_quantized=False)
    g_sw, _ = simulate_ganglion_1d(t_array, bip_sw, is_quantized=False, xi=1)
    
    # 3. Simulate Hardware Pipelines (is_quantized=True)
    opl_hw = simulate_opl_1d(t_array, stim, is_quantized=True)
    bip_hw = simulate_bipolar_1d(t_array, opl_hw, is_quantized=True)
    g_hw, _ = simulate_ganglion_1d(t_array, bip_hw, is_quantized=True, xi=1)

    # 4. TRUE METHODOLOGY: Calculate Real Variance Explained dynamically
    var_opl = compute_variance_explained(opl_sw, opl_hw)
    var_bip = compute_variance_explained(bip_sw, bip_hw)
    var_gang = compute_variance_explained(g_sw, g_hw)
    variance_values = [var_opl, var_bip, var_gang]

    # 5. TRUE METHODOLOGY: Mathematically Scaled Cross-Correlation
    sw_norm = g_sw - np.mean(g_sw)
    hw_norm = g_hw - np.mean(g_hw)

    # Standard full cross correlation
    cross_corr = np.correlate(hw_norm, sw_norm, mode='full')
    
    # Create lag array and convert directly to milliseconds
    lags_indices = np.arange(-len(t_array) + 1, len(t_array))
    lags_ms = lags_indices * dt_ms

    # True Pearson normalization factor
    norm_factor = np.sqrt(np.sum(sw_norm**2) * np.sum(hw_norm**2))
    cross_corr_normalized = cross_corr / norm_factor if norm_factor > 0 else cross_corr

    # Find the REAL maximum peak and its corresponding lag in ms
    max_idx = np.argmax(cross_corr_normalized)
    real_max_lag = lags_ms[max_idx]
    real_max_corr = cross_corr_normalized[max_idx]

    # =====================================================================
    # PLOTTING MASTER ASSEMBLY
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Plot Fig 9a: Variance Ratio Bars ---
    layers = ['OPL', 'Bipolar', 'Ganglion']
    colors = ['#FFF9C4', '#E0F7FA', '#F8BBD0']  # Cream-Yellow, Cyan, Light-Pink
    edgecolors = ['#CBC6A0', '#A8D3D6', '#D19DB0']

    bars = ax1.bar(layers, variance_values, color=colors, edgecolor=edgecolors, width=0.6, linewidth=1.2)
    ax1.set_title("Fig. 9(a): Ratio of Variance Explained", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Ratio of Variance (%)", fontsize=10)
    ax1.set_ylim(0, 115)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.5)

    # Add dynamic value annotations exactly on top of bars
    for bar in bars:
        height = bar.get_height()
        # Format to 2 decimal places to match the paper style
        ax1.annotate(f'{height:.2f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    # --- Plot Fig 9b: Cross Correlation Lags ---
    ax2.plot(lags_ms, cross_corr_normalized, color='#4FC3F7', label='Cross-correlation', linewidth=1.5)
    
    # Plot the true calculated peak dynamically
    ax2.scatter([real_max_lag], [real_max_corr], color='blue', s=20, zorder=5, label='Maximum Peak')
    ax2.text(real_max_lag + 80, real_max_corr - 0.05, f"x={real_max_lag:.1f}, y={real_max_corr:.3f}", 
             color='blue', fontsize=10, fontweight='bold')

    ax2.set_title("Fig. 9(b): Hardware & Software Time Difference", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Time Lags (ms)", fontsize=10)
    ax2.set_ylabel("Correlation (Norm)", fontsize=10)
    
    # Bound the x-axis to a reasonable window (e.g., +/- 2000 ms) to keep it readable
    ax2.set_xlim(-2200, 2200)
    
    # Dynamically adjust Y limits based on the actual correlation range
    min_corr = np.min(cross_corr_normalized)
    ax2.set_ylim(min_corr - 0.1, 1.1)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_figure9_replication()