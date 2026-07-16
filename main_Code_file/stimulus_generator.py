import numpy as np

def generate_chirp_stimulus():
    num_samples = 2000
    dt = 0.001  
    t = np.arange(num_samples) * dt
    
    baseline = 0.5
    luminance = np.ones(num_samples) * baseline
    
    # 1. OFF-ON-OFF Pulse Train Sequence
    luminance[(t >= 0.1) & (t < 0.3)] = 0.2  
    luminance[(t >= 0.3) & (t < 0.5)] = 0.8  
    luminance[(t >= 0.5) & (t < 0.75)] = 0.2 
    
    # 2. Sinusoidal Frequency and Amplitude Sweep (Chirp Phase)
    chirp_mask = (t >= 0.8)
    t_chirp = t[chirp_mask] - 0.8
    T_max = t_chirp[-1]
    
    amplitude = 0.05 + (0.35 * (t_chirp / T_max))
    f0, f1 = 2.0, 20.0   
    k = (f1 - f0) / T_max
    phase = 2.0 * np.pi * (f0 * t_chirp + 0.5 * k * (t_chirp ** 2))
    
    luminance[chirp_mask] = baseline + amplitude * np.sin(phase)
    return t, luminance