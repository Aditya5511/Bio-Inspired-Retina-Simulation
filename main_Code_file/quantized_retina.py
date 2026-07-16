import numpy as np
import matplotlib.pyplot as plt
from stimulus_generator import generate_chirp_stimulus
from opl_layer import simulate_opl

def apply_fixed_point(value, total_bits=16, frac_bits=10):
    """Simulates fixed-point hardware truncation and clipping."""
    scale = 2 ** frac_bits
    scaled_value = value * scale
    
    # Round to nearest integer (mimicking digital registers)
    quantized = np.round(scaled_value)
    
    # Clip the values to fit within signed bit-width limits
    min_val = - (2 ** (total_bits - 1))
    max_val = (2 ** (total_bits - 1)) - 1
    quantized = np.clip(quantized, min_val, max_val)
    
    # Scale back to floating point representation for analysis
    return quantized / scale

def simulate_quantized_opl(time_array, stimulus, dt=0.001):
    # Standard OPL parameters
    tau_p = 0.01  
    tau_h = 0.08  
    w = 0.7       
    
    p = np.zeros_like(stimulus)  
    h = np.zeros_like(stimulus)  
    opl_output = np.zeros_like(stimulus)
    
    # Set initial states and quantize them immediately
    p[0] = apply_fixed_point(stimulus[0])
    h[0] = apply_fixed_point(stimulus[0])
    opl_output[0] = apply_fixed_point(p[0] - apply_fixed_point(w * h[0]))
    
    for i in range(1, len(time_array)):
        # Hardware step 1: Photoreceptor update with quantization
        dp = (stimulus[i-1] - p[i-1]) / tau_p
        p[i] = apply_fixed_point(p[i-1] + apply_fixed_point(dp * dt))
        
        # Hardware step 2: Horizontal cell update with quantization
        dh = (p[i-1] - h[i-1]) / tau_h
        h[i] = apply_fixed_point(h[i-1] + apply_fixed_point(dh * dt))
        
        # Hardware step 3: Subtractive output with quantization
        weighted_h = apply_fixed_point(w * h[i])
        opl_output[i] = apply_fixed_point(p[i] - weighted_h)
        
    return opl_output

if __name__ == "__main__":
    # 1. Generate stimulus
    time_array, luminance_input = generate_chirp_stimulus()
    
    # 2. Run high-precision floating-point software model
    clean_opl = simulate_opl(time_array, luminance_input)
    
    # 3. Run restricted fixed-point hardware emulation model
    hardware_opl = simulate_quantized_opl(time_array, luminance_input)
    
    # 4. Plot them together to visualize the comparison
    plt.figure(figsize=(10, 5))
    plt.plot(time_array, clean_opl, label='Ideal Software Model (Float64)', color='blue', alpha=0.7)
    plt.plot(time_array, hardware_opl, label='Emulated Hardware Model (Fixed-Point 16-bit)', color='orange', linestyle='--')
    
    plt.title('Comparison: Ideal Software vs. Quantized Hardware OPL')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()