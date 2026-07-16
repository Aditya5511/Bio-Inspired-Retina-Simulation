import numpy as np
from retina_params import a1, b1, a2, b2, a3, b3, apply_19bit_fxp

def simulate_opl_1d(time_array, luminance_input, is_quantized=False):
    N = len(time_array)
    
    # State tracking arrays
    E_tau_C = np.zeros(N)
    Tw_tau = np.zeros(N)
    E_tau_S = np.zeros(N)
    IOPL = np.zeros(N)
    
    # CRITICAL FIX: If in hardware mode, the filter coefficients 
    # must be loaded into the quantized registers first.
    if is_quantized:
        q_a1, q_b1 = apply_19bit_fxp(a1), apply_19bit_fxp(b1)
        q_a2, q_b2 = apply_19bit_fxp(a2), apply_19bit_fxp(b2)
        q_a3, q_b3 = apply_19bit_fxp(a3), apply_19bit_fxp(b3)
    else:
        q_a1, q_b1 = a1, b1
        q_a2, q_b2 = a2, b2
        q_a3, q_b3 = a3, b3
        
    for n in range(1, N):
        x = luminance_input[n]
        if is_quantized: 
            x = apply_19bit_fxp(x)
        
        # 1. Center Low-Pass Filter Stage (Line 3)
        if is_quantized:
            term1 = apply_19bit_fxp(q_b1 * x)
            term2 = apply_19bit_fxp(q_a1 * E_tau_C[n-1])
            E_tau_C[n] = apply_19bit_fxp(term1 - term2)
        else:
            E_tau_C[n] = q_b1 * x - q_a1 * E_tau_C[n-1]
        
        # 2. Photoreceptor High-Pass Filter Stage (Line 4)
        if is_quantized:
            diff = apply_19bit_fxp(E_tau_C[n] - E_tau_C[n-1])
            term1 = apply_19bit_fxp(q_b2 * diff)
            term2 = apply_19bit_fxp(q_a2 * Tw_tau[n-1])
            Tw_tau[n] = apply_19bit_fxp(term1 - term2)
        else:
            Tw_tau[n] = q_b2 * (E_tau_C[n] - E_tau_C[n-1]) - q_a2 * Tw_tau[n-1]
        
        # 3. Surround Delayed Low-Pass Filter Stage (Line 6)
        if is_quantized:
            term1 = apply_19bit_fxp(q_b3 * Tw_tau[n])
            term2 = apply_19bit_fxp(q_a3 * E_tau_S[n-1])
            E_tau_S[n] = apply_19bit_fxp(term1 - term2)
        else:
            E_tau_S[n] = q_b3 * Tw_tau[n] - q_a3 * E_tau_S[n-1]
        
        # 4. Final Antagonism Subtraction (Line 9)
        if is_quantized:
            IOPL[n] = apply_19bit_fxp(Tw_tau[n] - E_tau_S[n])
        else:
            IOPL[n] = Tw_tau[n] - E_tau_S[n]
        
    return IOPL

# Alias to match imports in other files
simulate_opl = simulate_opl_1d