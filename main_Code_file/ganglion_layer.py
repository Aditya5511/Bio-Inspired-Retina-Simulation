import numpy as np
from retina_params import a5, b5, I0_G, V0_G, LAMBDA_G, GL, apply_19bit_fxp

def simulate_ganglion_1d(time_array, vbip_input, is_quantized=False, xi=1):
    N = len(time_array)
    
    # State tracking arrays
    I_Gang = np.zeros(N)
    V_G = np.zeros(N)
    Spikes = np.zeros(N)
    
    # CRITICAL FIX: Quantize the constants for hardware mode
    if is_quantized:
        q_a5, q_b5 = apply_19bit_fxp(a5), apply_19bit_fxp(b5)
        q_I0_G = apply_19bit_fxp(I0_G)
        q_V0_G = apply_19bit_fxp(V0_G)
        q_LAMBDA_G = apply_19bit_fxp(LAMBDA_G)
        q_GL = apply_19bit_fxp(GL)
    else:
        q_a5, q_b5 = a5, b5
        q_I0_G = I0_G
        q_V0_G = V0_G
        q_LAMBDA_G = LAMBDA_G
        q_GL = GL

    for n in range(1, N):
        x = vbip_input[n]
        if is_quantized: x = apply_19bit_fxp(x)
        
        # 1. Ganglion Input Current (Line 15)
        # xi determines ON (+1) or OFF (-1) pathway
        if is_quantized:
            term1 = apply_19bit_fxp(xi * x)
            I_Gang[n] = apply_19bit_fxp(q_I0_G + term1)
        else:
            I_Gang[n] = q_I0_G + xi * x
        
        # 2. Ganglion High-Pass Filter Stage (Line 17)
        if is_quantized:
            diff = apply_19bit_fxp(I_Gang[n] - I_Gang[n-1])
            term1 = apply_19bit_fxp(q_b5 * diff)
            term2 = apply_19bit_fxp(q_a5 * V_G[n-1])
            V_G[n] = apply_19bit_fxp(term1 - term2)
        else:
            V_G[n] = q_b5 * (I_Gang[n] - I_Gang[n-1]) - q_a5 * V_G[n-1]
        
        # 3. Simple Spiking Threshold (Approximating Eq 18/19 dynamics)
        if V_G[n] > q_V0_G:
            Spikes[n] = 1.0
        else:
            Spikes[n] = 0.0
            
    return I_Gang, Spikes

# Alias to match imports in other files
simulate_ganglion = simulate_ganglion_1d