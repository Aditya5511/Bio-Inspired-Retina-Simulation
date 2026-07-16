import numpy as np
import matplotlib.pyplot as plt
# Import previous stages cleanly
from stimulus_generator import generate_chirp_stimulus
from opl_layer import simulate_opl
from bipolar_layer import simulate_bipolar
from quantized_retina import simulate_quantized_opl, apply_fixed_point

def simulate_quantized_bipolar(time_array, quantized_opl_output, dt=0.001):
    tau_b = 0.015  
    b = np.zeros_like(quantized_opl_output)
    
    # Initialize initial state with hardware quantization
    b[0] = apply_fixed_point(quantized_opl_output[0])
    
    for i in range(1, len(time_array)):
        # Hardware step: Calculate derivative and quantize
        db = (quantized_opl_output[i-1] - b[i-1]) / tau_b
        # Update register with fixed-point math
        b[i] = apply_fixed_point(b[i-1] + apply_fixed_point(db * dt))
    
    # Hardware rectification (splitting channels)
    on_bipolar = np.maximum(0, b)
    off_bipolar = np.maximum(0, -b)
    
    return on_bipolar, off_bipolar

def extract_signals(data, target_len):
    """Deeply searches the function output to find data arrays of the correct length (e.g., 2000)"""
    found = []
    if isinstance(data, np.ndarray):
        if data.size == target_len:
            found.append(data.flatten())
        elif data.ndim == 2 and data.shape[0] == target_len:
            found.append(data[:, 0])
            found.append(data[:, 1])
        return found
    if isinstance(data, list):
        if len(data) == target_len and not isinstance(data[0], (list, np.ndarray)):
            found.append(np.array(data))
            return found
    if isinstance(data, (tuple, list)):
        for item in data:
            found.extend(extract_signals(item, target_len))
    return found

if __name__ == "__main__":
    # 1. Generate core stimulus
    time_array, luminance_input = generate_chirp_stimulus()
    target_len = len(time_array)  # This will be exactly 2000
    
    # 2. Run High-Precision Floats Pipeline
    opl_raw = simulate_opl(time_array, luminance_input)
    # Extract only arrays of length 2000, grab the last one (ignores time arrays/scalars)
    opl_arrays = extract_signals(opl_raw, target_len)
    ideal_opl = opl_arrays[-1] 

    bipolar_raw = simulate_bipolar(time_array, ideal_opl)
    bipolar_arrays = extract_signals(bipolar_raw, target_len)
    # Bipolar layer returns ON and OFF, so we grab the last two 2000-point arrays
    ideal_on = bipolar_arrays[-2] if len(bipolar_arrays) >= 2 else bipolar_arrays[0]
    ideal_off = bipolar_arrays[-1] if len(bipolar_arrays) >= 2 else bipolar_arrays[0]
    
    # 3. Run Emulated Hardware Fixed-Point Pipeline
    q_opl_raw = simulate_quantized_opl(time_array, luminance_input)
    q_opl_arrays = extract_signals(q_opl_raw, target_len)
    q_opl = q_opl_arrays[-1]

    q_on, q_off = simulate_quantized_bipolar(time_array, q_opl)
    
    # Final safety squeeze to guarantee 1D shape for Matplotlib
    ideal_on = np.squeeze(ideal_on)
    q_on = np.squeeze(q_on)
    
    # 4. Plot the comparison for the ON channel
    plt.figure(figsize=(10, 5))
    plt.plot(time_array, ideal_on, label='Ideal ON Bipolar (Float64)', color='green', alpha=0.7)
    plt.plot(time_array, q_on, label='Hardware Emulated ON Bipolar (19-bit Signed Q8.11)', color='darkorange', linestyle='--')
    plt.title('Bipolar Layer Hardware Emulation Comparison (ON Channel)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Signal Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()