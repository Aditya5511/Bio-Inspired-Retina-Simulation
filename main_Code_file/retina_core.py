import numpy as np
from scipy.signal import convolve2d
import cv2  

# =============================================================================
# 1. GLOBAL SIMULATION PARAMETERS (From Table VII)
# =============================================================================
DT = 0.001  

SIGMA_C, SIGMA_S, SIGMA_A = 1.0, 4.0, 1.0
TAU_C, TAU_U, TAU_S = 0.010, 0.010, 0.010
TAU_A, TAU_G = 0.005, 0.020

LAMBDA_OPL, OMEGA_OPL = 1.0, 0.5      
G_A_0, LAMBDA_A = 50.0, 0.0        
LAMBDA_G, I_G_0, V_G_0, G_L = 5.0, 0.008, 0.0, 0.1             

# =============================================================================
# 2. 19-BIT FPGA FIXED-POINT CONFIGURATION (From Research Paper Specs)
# =============================================================================
# Total Word Length = 19 bits (1 Sign bit + 4 Integer bits + 14 Fractional bits)
FRAC_BITS = 14
MULT = 2 ** FRAC_BITS

# Strict 19-bit signed integer boundaries to prevent wrap-around noise
MIN_19BIT = -(2 ** 18)  # -262144
MAX_19BIT = (2 ** 18) - 1  # 262143

def calc_iir_coeffs_fixed(tau, dt):
    a_float = tau / (tau + dt)
    b_float = dt / (tau + dt)
    a_int = int(np.round(a_float * MULT))
    b_int = int(np.round(b_float * MULT))
    return a_int, b_int

a1_int, b1_int = calc_iir_coeffs_fixed(TAU_C, DT)
a2_int, b2_int = calc_iir_coeffs_fixed(TAU_U, DT)
a3_int, b3_int = calc_iir_coeffs_fixed(TAU_S, DT)
a4_int, b4_int = calc_iir_coeffs_fixed(TAU_A, DT)
a5_int, b5_int = calc_iir_coeffs_fixed(TAU_G, DT)

# =============================================================================
# 3. FIXED-POINT HARDWARE-EMULATED FILTERS
# =============================================================================
def iir_lpf(current_input, prev_output, a_int, b_int):
    # Quantize incoming floating point data arrays to integers
    curr_in_int = np.round(current_input * MULT).astype(np.int64)
    prev_out_int = np.round(prev_output * MULT).astype(np.int64)
    
    # FPGA Math: Wide accumulator multiplication and addition
    accumulator = (a_int * prev_out_int) + (b_int * curr_in_int)
    
    # Drop fractional bits down to target resolution
    shifted = accumulator >> FRAC_BITS
    
    # Enforce strict hardware saturation to suppress background noise
    out_int = np.clip(shifted, MIN_19BIT, MAX_19BIT)
    
    return out_int / MULT

def iir_hpf(current_input, prev_input, prev_output, a_int, b_int):
    curr_in_int = np.round(current_input * MULT).astype(np.int64)
    prev_in_int = np.round(prev_input * MULT).astype(np.int64)
    prev_out_int = np.round(prev_output * MULT).astype(np.int64)
    
    # FPGA High-Pass Difference Accumulation
    accumulator = (a_int * prev_out_int) + (a_int * (curr_in_int - prev_in_int))
    shifted = accumulator >> FRAC_BITS
    
    out_int = np.clip(shifted, MIN_19BIT, MAX_19BIT)
    
    return out_int / MULT

# =============================================================================
# 4. SPATIAL KERNELS
# =============================================================================
def generate_gaussian_kernel(size, sigma):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

K1 = generate_gaussian_kernel(size=7, sigma=SIGMA_C)
K2 = generate_gaussian_kernel(size=21, sigma=SIGMA_S) 
K3 = generate_gaussian_kernel(size=7, sigma=SIGMA_A)

# =============================================================================
# 5. NEURAL LAYERS 
# =============================================================================
def OPLlayer(x_frame, prev_E_tauC, prev_prev_E_tauC, prev_T_w_tau, prev_E_tauS):
    G_C = convolve2d(x_frame, K1, mode='same', boundary='symm')
    
    E_tauC = iir_lpf(G_C, prev_E_tauC, a1_int, b1_int)
    T_w_tau = iir_hpf(E_tauC, prev_prev_E_tauC, prev_T_w_tau, a2_int, b2_int)
    
    G_S = convolve2d(T_w_tau, K2, mode='same', boundary='symm')
    E_tauS = iir_lpf(G_S, prev_E_tauS, a3_int, b3_int)
    
    I_OPL_output = LAMBDA_OPL * (T_w_tau - (OMEGA_OPL * E_tauS))
    return I_OPL_output, E_tauC, T_w_tau, E_tauS

def Bipolar(I_OPL, prev_V_Bip, prev_att_map, prev_E_A, inputamp=1.0, steps=0.001):
    g_A = G_A_0 + prev_E_A
    att_map = np.exp(-steps * g_A)
    E_inf = inputamp * I_OPL
    V_Bip = ((prev_V_Bip - E_inf) * att_map) + E_inf
    
    term1_int = np.round((LAMBDA_A * np.square(prev_V_Bip)) * MULT).astype(np.int64)
    prev_E_A_int = np.round(prev_E_A * MULT).astype(np.int64)
    
    accumulator = (term1_int * b4_int) - (prev_E_A_int * a4_int)
    shifted = accumulator >> FRAC_BITS
    E_A_temp = np.clip(shifted, MIN_19BIT, MAX_19BIT) / MULT
    
    E_A = convolve2d(E_A_temp, K3, mode='same', boundary='symm')
    return V_Bip, att_map, E_A

def ganglion_current_dual(V_Bip, prev_V_Bip, prev_T_G, xi=1.0):
    T_G = iir_hpf(V_Bip, prev_V_Bip, prev_T_G, a5_int, b5_int)
    
    x_on = xi * T_G
    n_on = np.zeros_like(x_on)
    mask_less_on = x_on < V_G_0
    n_on[mask_less_on] = (I_G_0 * I_G_0) / (I_G_0 - (LAMBDA_G * (x_on[mask_less_on] - V_G_0)))
    mask_greater_on = x_on >= V_G_0
    n_on[mask_greater_on] = I_G_0 + (LAMBDA_G * (x_on[mask_greater_on] - V_G_0))
    
    x_off = -xi * T_G
    n_off = np.zeros_like(x_off)
    mask_less_off = x_off < V_G_0
    n_off[mask_less_off] = (I_G_0 * I_G_0) / (I_G_0 - (LAMBDA_G * (x_off[mask_less_off] - V_G_0)))
    mask_greater_off = x_off >= V_G_0
    n_off[mask_greater_off] = I_G_0 + (LAMBDA_G * (x_off[mask_greater_off] - V_G_0))
    
    return n_on, n_off, T_G

def spiking(I_Gang, V_m, rt, refractory_time=2.0, integration_rate=0.05):
    V_m = V_m + ((I_Gang - (G_L * V_m)) * integration_rate)
    rt = rt - 1.0 
    V_m[rt > 0.0] = 0.0
    Spikes = V_m > 1.0
    rt[rt < 0.0] = 0.0
    V_m[Spikes] = 0.0
    rt[Spikes] = refractory_time
    return Spikes, V_m, rt

# =============================================================================
# 6. LIVE WEBCAM EXPERIMENT RUNNER
# =============================================================================
def run_webcam_experiment():
    print("🔬 Initializing Live Webcam Retina Simulation with 19-bit Saturated IIR Filters...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    spatial_size = 128
    
    prev_E_tauC = np.zeros((spatial_size, spatial_size))
    prev_prev_E_tauC = np.zeros((spatial_size, spatial_size))
    prev_T_w_tau = np.zeros((spatial_size, spatial_size))
    prev_E_tauS = np.zeros((spatial_size, spatial_size))
    prev_V_Bip = np.zeros((spatial_size, spatial_size))
    prev_att_map = np.ones((spatial_size, spatial_size))
    prev_E_A = np.zeros((spatial_size, spatial_size))
    prev_T_G = np.zeros((spatial_size, spatial_size))
    
    V_m_ON, rt_ON = np.zeros((spatial_size, spatial_size)), np.zeros((spatial_size, spatial_size))
    V_m_OFF, rt_OFF = np.zeros((spatial_size, spatial_size)), np.zeros((spatial_size, spatial_size))
    
    print("👀 Camera active! Wave your hand in front of it.")
    print("⌨️ Press 'q' in the video window to stop.")

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        gray_frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1) 
        input_frame = cv2.resize(gray_frame, (spatial_size, spatial_size)) / 255.0
        
        I_OPL, E_tauC, T_w_tau, E_tauS = OPLlayer(input_frame, prev_E_tauC, prev_prev_E_tauC, prev_T_w_tau, prev_E_tauS)
        V_Bip, att_map, E_A = Bipolar(I_OPL, prev_V_Bip, prev_att_map, prev_E_A)
        I_Gang_ON, I_Gang_OFF, T_G = ganglion_current_dual(V_Bip, prev_V_Bip, prev_T_G, xi=15.0) 
        
        Spikes_ON, V_m_ON, rt_ON = spiking(I_Gang_ON, V_m_ON, rt_ON)
        Spikes_OFF, V_m_OFF, rt_OFF = spiking(I_Gang_OFF, V_m_OFF, rt_OFF)
        
        viz_ON = (Spikes_ON * 255).astype(np.uint8)
        viz_OFF = (Spikes_OFF * 255).astype(np.uint8)
        
        viz_ON_colored = cv2.merge([np.zeros_like(viz_ON), viz_ON, np.zeros_like(viz_ON)])
        viz_OFF_colored = cv2.merge([np.zeros_like(viz_OFF), np.zeros_like(viz_OFF), viz_OFF])
        orig_colored = cv2.cvtColor((input_frame * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        combined_view = np.hstack((orig_colored, viz_ON_colored, viz_OFF_colored))
        display_view = cv2.resize(combined_view, (spatial_size * 3 * 3, spatial_size * 3), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow("Live 19-bit Retina Simulation: [Webcam] | [ON Cells (Green)] | [OFF Cells (Red)]", display_view)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        prev_prev_E_tauC, prev_E_tauC = prev_E_tauC, E_tauC
        prev_T_w_tau, prev_E_tauS = T_w_tau, E_tauS
        prev_V_Bip, prev_att_map, prev_E_A = V_Bip, att_map, E_A
        prev_T_G = T_G

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_experiment()