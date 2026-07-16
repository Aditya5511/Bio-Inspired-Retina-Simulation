import numpy as np

# =====================================================================
# STRICT COMPLIANCE: SPATIAL CONSTANTS (Table VII)
# =====================================================================
SIGMA_C = 0.05      # Center spatial constant (deg)
SIGMA_S = 0.15      # Surround spatial constant (deg)
LAMBDA_OPL = 1.0    # OPL scalar weight
OMEGA_OPL = 0.5     # High-pass weight

# =====================================================================
# STRICT COMPLIANCE: TEMPORAL CONSTANTS (Table VII)
# =====================================================================
TAU_C = 0.010       # Center low-pass time constant
TAU_HPF = 0.020     # Photoreceptor high-pass time constant
TAU_S = 0.022       # Surround low-pass delay time constant
TAU_A = 0.005       # Amacrine feedback integration time constant
TAU_G = 0.020       # Ganglion high-pass filter time constant

DT = 0.001          # Step size (1ms execution rate)

# =====================================================================
# STRICT COMPLIANCE: IIR COEFFICIENT CORES
# =====================================================================
def standard_lpf_coeffs(tau, dt=DT):
    alpha = np.exp(-dt / tau)
    return -alpha, 1.0 - alpha

def standard_hpf_coeffs(tau, dt=DT):
    alpha = np.exp(-dt / tau)
    return -alpha, 1.0  # Literal high-pass differential numerator mapping

# Generate the explicit coefficient matrix matching Table I to IV
a1, b1 = standard_lpf_coeffs(TAU_C)
a2, b2 = standard_hpf_coeffs(TAU_HPF)
a3, b3 = standard_lpf_coeffs(TAU_S)
a4, b4 = standard_lpf_coeffs(TAU_A)
a5, b5 = standard_hpf_coeffs(TAU_G)

# Constant Gains & Shunting Multipliers (Table VII & Table III)
G0_A = 10.0
LAMBDA_A = 50.0
I0_G = 0.008
V0_G = 0.0
LAMBDA_G = 5.0
GL = 0.1
INPUT_AMP = 1.0

# =====================================================================
# STRICT COMPLIANCE: 19-BIT LOGIC FORMAT (Table VIII)
# =====================================================================
def apply_19bit_fxp(value):
    """Exact 19-bit signed fixed-point simulation using a strict Q8.11 datapath."""
    scale = 2 ** 11
    quantized = np.round(value * scale)
    quantized = np.clip(quantized, -(2**18), (2**18) - 1)
    return quantized / scale