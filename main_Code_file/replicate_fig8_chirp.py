import numpy as np
import matplotlib.pyplot as plt
from stimulus_generator import generate_chirp_stimulus
from opl_layer import simulate_opl_1d
from bipolar_layer import simulate_bipolar_1d
from ganglion_layer import simulate_ganglion_1d

# Pristine academic typography
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def run_figure8_replication():
    # 1. Generate the standard 2-second chirp stimulus
    t_array, stim = generate_chirp_stimulus()
    time_ms = t_array * 1000.0
    
    # 2. Simulate Pipelines
    sw_iopl = simulate_opl_1d(t_array, stim, is_quantized=False)
    sw_vbip = simulate_bipolar_1d(t_array, sw_iopl, is_quantized=False)
    sw_igang, sw_spikes_on = simulate_ganglion_1d(t_array, sw_vbip, is_quantized=False, xi=1)
    _, sw_spikes_off = simulate_ganglion_1d(t_array, sw_vbip, is_quantized=False, xi=-1)
    
    hw_iopl = simulate_opl_1d(t_array, stim, is_quantized=True)
    hw_vbip = simulate_bipolar_1d(t_array, hw_iopl, is_quantized=True)
    hw_igang, hw_spikes_on = simulate_ganglion_1d(t_array, hw_vbip, is_quantized=True, xi=1)
    _, hw_spikes_off = simulate_ganglion_1d(t_array, hw_vbip, is_quantized=True, xi=-1)

    # 3. Calibrate output levels to match paper "au" peaks
    max_sw_opl = np.max(np.abs(sw_iopl))
    scale_opl = 450.0 / max_sw_opl if max_sw_opl > 0 else 1400.0

    max_sw_bip = np.max(np.abs(sw_vbip))
    scale_bip = 325.0 / max_sw_bip if max_sw_bip > 0 else 900.0

    max_sw_gang = np.max(sw_igang)
    scale_gang = 930.0 / max_sw_gang if max_sw_gang > 0 else 60000.0

    sw_opl_scaled = sw_iopl * scale_opl
    hw_opl_scaled = hw_iopl * scale_opl
    
    sw_bip_scaled = sw_vbip * scale_bip
    hw_bip_scaled = hw_vbip * scale_bip
    
    sw_gang_scaled = sw_igang * scale_gang
    hw_gang_scaled = hw_igang * scale_gang

    # FIX: A tall, balanced canvas size perfectly matches a 4x2 grid of squares
    fig, axes = plt.subplots(4, 2, figsize=(6.8, 11.0))

    # Clean styling helper for perfect squares
    def style_square_axis(ax, x_label, y_label, sub_label):
        ax.set_box_aspect(1.0)  # Maintain strict square geometry matching the paper
        ax.tick_params(
            direction='in', 
            top=True, 
            right=True, 
            left=True, 
            bottom=True, 
            labelsize=8.5, 
            length=3.5, 
            width=0.7,
            pad=4
        )
        ax.set_xlabel(x_label, fontsize=9.5, labelpad=4)
        ax.set_ylabel(y_label, fontsize=9.5, labelpad=4)
        ax.text(0.5, -0.32, sub_label, transform=ax.transAxes, 
                fontsize=10, fontweight='bold', ha='center')
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)

    # Ultra-clean internal legend helper to eliminate white space clutter
    def place_clean_legend(ax, loc='upper left'):
        ax.legend(
            loc=loc,
            frameon=True,
            framealpha=0.85,
            edgecolor='none',
            facecolor='#ffffff',
            fontsize=7.5,
            handlelength=1.2,
            handletextpad=0.5
        )

    # =========================================================================
    # ROW 1: OPL LAYER
    # =========================================================================
    axes[0, 0].plot(time_ms, sw_opl_scaled, color='black', linewidth=0.9, label='SW')
    axes[0, 0].plot(time_ms, hw_opl_scaled, color='red', linewidth=0.7, alpha=0.85, label='HW')
    axes[0, 0].fill_between(time_ms, sw_opl_scaled, hw_opl_scaled, color='gold', alpha=0.35, label='Div')
    axes[0, 0].set_xlim(0, 2000)
    axes[0, 0].set_ylim(-600, 600)
    axes[0, 0].set_xticks([0, 1000, 2000])
    axes[0, 0].set_yticks([-500, 0, 500])
    style_square_axis(axes[0, 0], 'Time (ms)', r'$I_{\mathrm{OPL}}$ (au)', '(a)')
    place_clean_legend(axes[0, 0], loc='upper left')

    axes[0, 1].plot([-500, 500], [-500, 500], color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    axes[0, 1].plot(sw_opl_scaled, hw_opl_scaled, color='fuchsia', linewidth=0.7)
    axes[0, 1].set_xlim(-500, 500)
    axes[0, 1].set_ylim(-600, 600)
    axes[0, 1].set_xticks([-400, 0, 400])
    axes[0, 1].set_yticks([-500, 0, 500])
    style_square_axis(axes[0, 1], r'Software $I_{\mathrm{OPL}}$ (au)', r'Hardware $I_{\mathrm{OPL}}$ (au)', '(b)')

    # =========================================================================
    # ROW 2: BIPOLAR LAYER
    # =========================================================================
    axes[1, 0].plot(time_ms, sw_bip_scaled, color='black', linewidth=0.9, label='SW')
    axes[1, 0].plot(time_ms, hw_bip_scaled, color='red', linewidth=0.7, alpha=0.85, label='HW')
    axes[1, 0].fill_between(time_ms, sw_bip_scaled, hw_bip_scaled, color='gold', alpha=0.35, label='Div')
    axes[1, 0].set_xlim(0, 2000)
    axes[1, 0].set_ylim(-350, 400)
    axes[1, 0].set_xticks([0, 1000, 2000])
    axes[1, 0].set_yticks([-300, 0, 300])
    style_square_axis(axes[1, 0], 'Time (ms)', r'$V_{\mathrm{Bip}}$ (au)', '(c)')
    place_clean_legend(axes[1, 0], loc='upper left')

    axes[1, 1].plot([-350, 400], [-350, 400], color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    axes[1, 1].plot(sw_bip_scaled, hw_bip_scaled, color='fuchsia', linewidth=0.7)
    axes[1, 1].set_xlim(-350, 400)
    axes[1, 1].set_ylim(-350, 400)
    axes[1, 1].set_xticks([-300, 0, 300])
    axes[1, 1].set_yticks([-300, 0, 300])
    style_square_axis(axes[1, 1], r'Software $V_{\mathrm{Bip}}$ (au)', r'Hardware $V_{\mathrm{Bip}}$ (au)', '(d)')

    # =========================================================================
    # ROW 3: GANGLION CURRENT
    # =========================================================================
    axes[2, 0].plot(time_ms, sw_gang_scaled, color='black', linewidth=0.9, label='SW')
    axes[2, 0].plot(time_ms, hw_gang_scaled, color='red', linewidth=0.7, alpha=0.85, label='HW')
    axes[2, 0].fill_between(time_ms, sw_gang_scaled, hw_gang_scaled, color='gold', alpha=0.35, label='Div')
    axes[2, 0].set_xlim(0, 2000)
    axes[2, 0].set_ylim(-300, 1100)
    axes[2, 0].set_xticks([0, 1000, 2000])
    axes[2, 0].set_yticks([0, 500, 1000])
    style_square_axis(axes[2, 0], 'Time (ms)', r'$I_{\mathrm{Gang}}$ (au)', '(e)')
    place_clean_legend(axes[2, 0], loc='upper left')

    axes[2, 1].plot([-100, 1100], [-100, 1100], color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    axes[2, 1].plot(sw_gang_scaled, hw_gang_scaled, color='fuchsia', linewidth=0.7)
    axes[2, 1].set_xlim(-100, 1100)
    axes[2, 1].set_ylim(-100, 1100)
    axes[2, 1].set_xticks([0, 500, 1000])
    axes[2, 1].set_yticks([0, 500, 1000])
    style_square_axis(axes[2, 1], r'Software $I_{\mathrm{Gang}}$ (au)', r'Hardware $I_{\mathrm{Gang}}$ (au)', '(f)')

    # =========================================================================
    # ROW 4: BIOLOGICAL SPIKES
    # =========================================================================
    mismatch_on = np.abs(sw_spikes_on - hw_spikes_on) > 0.5
    mismatch_off = np.abs(sw_spikes_off - hw_spikes_off) > 0.5

    # (g) Spikes ON
    axes[3, 0].plot(time_ms, sw_spikes_on, color='black', linewidth=0.5, alpha=0.8, label='SW')
    axes[3, 0].plot(time_ms, hw_spikes_on, color='red', linewidth=0.5, alpha=0.7, label='HW')
    axes[3, 0].fill_between(time_ms, -0.2, 1.1, where=mismatch_on, color='orange', alpha=0.3, step='mid', label='Mis')
    axes[3, 0].set_xlim(0, 2000)
    axes[3, 0].set_ylim(-0.2, 1.2)
    axes[3, 0].set_xticks([0, 1000, 2000])
    axes[3, 0].set_yticks([0.0, 1.0])
    style_square_axis(axes[3, 0], 'Time (ms)', 'Spikes ON (au)', '(g)')
    place_clean_legend(axes[3, 0], loc='upper right')

    # (h) Spikes OFF
    axes[3, 1].plot(time_ms, sw_spikes_off, color='black', linewidth=0.5, alpha=0.8, label='SW')
    axes[3, 1].plot(time_ms, hw_spikes_off, color='red', linewidth=0.5, alpha=0.7, label='HW')
    axes[3, 1].fill_between(time_ms, -0.2, 1.1, where=mismatch_off, color='orange', alpha=0.3, step='mid', label='Mis')
    axes[3, 1].set_xlim(0, 2000)
    axes[3, 1].set_ylim(-0.2, 1.2)
    axes[3, 1].set_xticks([0, 1000, 2000])
    axes[3, 1].set_yticks([0.0, 1.0])
    style_square_axis(axes[3, 1], 'Time (ms)', 'Spikes OFF (au)', '(h)')
    place_clean_legend(axes[3, 1], loc='upper right')

    # Tight alignment mapping to eliminate structural whitespace padding
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.48, wspace=0.22, bottom=0.05, top=0.98, left=0.11, right=0.96)
    plt.show()

if __name__ == "__main__":
    run_figure8_replication()