import numpy as np
import matplotlib.pyplot as plt
from retina_params import b1, a1, b2, a2

# Setup time arrays
t_impulse = np.arange(0, 300, 1)  # 300ms window for Figure 7a
t_chirp = np.arange(0, 2000, 1)   # 2000ms window for Figure 7b

# =====================================================================
# GENERATE FIG 7(a): IMPULSE RESPONSE
# =====================================================================
impulse_stim = np.zeros(len(t_impulse))
impulse_stim[100] = 1.0  # Delta function strike at 100ms

E_C = np.zeros(len(t_impulse))
Tw = np.zeros(len(t_impulse))

# Pass through OPL center low-pass + high-pass cascade
for n in range(1, len(t_impulse)):
    E_C[n] = b1 * impulse_stim[n] - a1 * E_C[n-1]
    Tw[n] = b2 * (E_C[n] - E_C[n-1]) - a2 * Tw[n-1]

# Scale to match the paper's arbitrary unit (au) layout
Tw_scaled = (Tw / np.max(Tw)) * 240.0

# =====================================================================
# GENERATE FIG 7(b): CHIRP STIMULUS PROFILE
# =====================================================================
chirp_stim = np.zeros(len(t_chirp))
# ON-OFF Pulses
chirp_stim[200:450] = 1000.0
# Frequency/Amplitude Sweep
for t in range(700, 1800):
    f = 0.000015 * (t - 700)**2
    chirp_stim[t] = 1000.0 * np.sin(2 * np.pi * f * (t / 1000.0))

# =====================================================================
# PLOTTING MASTER ASSEMBLY
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Plot Fig 7a
ax1.plot(t_impulse, Tw_scaled, color='red', linewidth=1.5)
ax1.set_title("Fig. 7(a): Photoreceptor Impulse Response", fontsize=11, fontweight='bold')
ax1.set_xlabel("Time (ms)", fontsize=10)
ax1.set_ylabel("Amplitude (au)", fontsize=10)
ax1.set_ylim(-100, 260)
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot Fig 7b
ax2.plot(t_chirp, chirp_stim, color='blue', linewidth=1.5)
ax2.set_title("Fig. 7(b): Oscillatory 'Chirp' Stimulus", fontsize=11, fontweight='bold')
ax2.set_xlabel("Time (ms)", fontsize=10)
ax2.set_ylabel("Stimulus (au)", fontsize=10)
ax2.set_ylim(-1200, 1200)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()