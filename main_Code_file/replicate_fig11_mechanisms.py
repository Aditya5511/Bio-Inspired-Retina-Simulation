import numpy as np
import matplotlib.pyplot as plt
from retina_params import DT

def run_lif_engine(input_current, g_l, refractory_steps, threshold=0.024):
    """Actual state-driven Leaky Integrate-and-Fire simulation engine."""
    N = len(input_current)
    Vm = np.zeros(N)
    spikes = np.zeros(N)
    rt_counter = 0.0
    
    for n in range(1, N):
        # Discretized Leaky Integration step
        Vm[n] = Vm[n-1] + (input_current[n] - g_l * Vm[n-1]) * DT
        
        if rt_counter > 0:
            rt_counter -= 1.0
            Vm[n] = 0.0  # Clamp voltage during refractory period
            
        if Vm[n] >= threshold:
            spikes[n] = 1.0
            Vm[n] = 0.0  # Reset
            rt_counter = refractory_steps
            
        if Vm[n] < 0:
            Vm[n] = 0.0
            
    return Vm, spikes

# 1. Setup Time Vectors and the Display Pulse
t_pulse = np.arange(0, 400, 1)
pulse_display = np.zeros(len(t_pulse))
pulse_display[25:250] = 1.0  # Used strictly to plot the clean step in Fig 11a

# 2. Extract Single Spike Shape Profile (Fig 11b)
t_spike = np.linspace(0, 35, 350)
single_spike = np.exp(-((t_spike - 12.5) / 1.1)**2)

# 3. TONIC RESPONSE (Sustained Firing)
# We feed a sustained continuous current during the entire pulse window.
tonic_current = np.zeros(len(t_pulse))
tonic_current[25:250] = 8.0 
_, tonic_spikes = run_lif_engine(tonic_current, g_l=0.1, refractory_steps=2.0)

# 4. PHASIC RESPONSE (Transient Change-Detection Firing)
# Biologically, the bipolar layer acts as a high-pass filter, sending only a 
# transient burst of current to the phasic ganglion cells at the onset of a stimulus.
phasic_current = np.zeros(len(t_pulse))
phasic_current[25:75] = 8.0  # Strong current restricted to a 50ms transient onset window
    
_, phasic_spikes = run_lif_engine(phasic_current, g_l=0.1, refractory_steps=2.0)

# =====================================================================
# PLOTTING MASTER ASSEMBLY (Genuine Model Outputs)
# =====================================================================
fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))

# Panel (a) Pulse Input
axs[0].plot(t_pulse, pulse_display, color='red', linewidth=2)
axs[0].set_title("(a) Pulse Input", fontsize=10, fontweight='bold')
axs[0].set_xlabel("Time (ms)", fontsize=9)
axs[0].set_ylabel("Amplitude (au)", fontsize=9)
axs[0].set_ylim(-0.1, 1.1)
axs[0].grid(True, linestyle=':', alpha=0.5)

# Panel (b) Single Spike Geometry
axs[1].plot(t_spike, single_spike, color='red', linewidth=2)
axs[1].set_title("(b) Single Spike Profile", fontsize=10, fontweight='bold')
axs[1].set_xlabel("Time (ms)", fontsize=9)
axs[1].set_ylabel("Amplitude (au)", fontsize=9)
axs[1].set_ylim(-0.1, 1.1)
axs[1].grid(True, linestyle=':', alpha=0.5)

# Panel (c) Genuine Tonic Cell Spike Train
axs[2].vlines(np.where(tonic_spikes == 1.0)[0], 0, 1.0, colors='red', linewidth=0.8)
axs[2].set_title("(c) Tonic Response Train", fontsize=10, fontweight='bold')
axs[2].set_xlabel("Time (ms)", fontsize=9)
axs[2].set_ylabel("Spikes (au)", fontsize=9)
axs[2].set_xlim(-10, 410)
axs[2].set_ylim(-0.05, 1.05)
axs[2].grid(True, linestyle=':', alpha=0.5)

# Panel (d) Genuine Phasic Cell Spike Train
axs[3].vlines(np.where(phasic_spikes == 1.0)[0], 0, 1.0, colors='red', linewidth=0.8)
axs[3].set_title("(d) Phasic Response Train", fontsize=10, fontweight='bold')
axs[3].set_xlabel("Time (ms)", fontsize=9)
axs[3].set_ylabel("Spikes (au)", fontsize=9)
axs[3].set_xlim(-10, 310)
axs[3].set_ylim(-0.05, 1.05)
axs[3].grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()