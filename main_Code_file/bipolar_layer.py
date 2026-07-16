import numpy as np
from retina_params import a4, b4, G0_A, LAMBDA_A, apply_19bit_fxp

def simulate_bipolar_1d(time_array, iopl_input, is_quantized=False):
    N = len(time_array)
    
    # State tracking arrays
    V_Bip = np.zeros(N)
    V_A = np.zeros(N)
    
    # CRITICAL FIX: Quantize the constants for hardware mode
    if is_quantized:
        q_a4, q_b4 = apply_19bit_fxp(a4), apply_19bit_fxp(b4)
        q_G0_A = apply_19bit_fxp(G0_A)
        q_LAMBDA_A = apply_19bit_fxp(LAMBDA_A)
    else:
        q_a4, q_b4 = a4, b4
        q_G0_A = G0_A
        q_LAMBDA_A = LAMBDA_A

    for n in range(1, N):
        x = iopl_input[n]
        if is_quantized: x = apply_19bit_fxp(x)
        
        # 1. Bipolar Voltage (Line 10)
        if is_quantized:
            V_Bip[n] = apply_19bit_fxp(x - V_A[n-1])
        else:
            V_Bip[n] = x - V_A[n-1]
        
        # 2. Amacrine Shunting Activation (Line 12)
        f_V_Bip = max(0, V_Bip[n])  # Rectification
        if is_quantized: f_V_Bip = apply_19bit_fxp(f_V_Bip)
        
        # 3. Amacrine Low-Pass Filter Stage (Line 13)
        if is_quantized:
            term1 = apply_19bit_fxp(q_b4 * f_V_Bip)
            term2 = apply_19bit_fxp(q_a4 * V_A[n-1])
            V_A[n] = apply_19bit_fxp(term1 - term2)
        else:
            V_A[n] = q_b4 * f_V_Bip - q_a4 * V_A[n-1]
        
    return V_Bip

# Alias to match imports in other files
simulate_bipolar = simulate_bipolar_1d