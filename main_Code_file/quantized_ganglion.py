import numpy as np
import matplotlib.pyplot as plt
# Import previous clean and quantized stages
from stimulus_generator import generate_chirp_stimulus
from opl_layer import simulate_opl
from bipolar_layer import simulate_bipolar
from ganglion_layer import simulate_ganglion
from quantized_retina import simulate_quantized_opl, apply_fixed_point
from quantized_bipolar import simulate_quantized_bipolar

def simulate_quantized_ganglion(time_array, q_on_bipolar, q_off_bipolar, dt=0.001):
    # Quantized LIF parameters
    tau_m = 0.03    
    v_thresh = 0.6  
    v_reset = 0.0   
    gain = 12.0     
    
    v_on = np.zeros_like(q_on_bipolar)
    v_off = np.zeros_like(q_off_bipolar)
    spikes_on = np.zeros_like(q_on_bipolar)
    spikes_off = np.zeros_like(q_off_bipolar)
    
    # Initialize hardware registers
    v_on[0] = apply_fixed_point(0.0)
    v_off[0] = apply_fixed_point(0.0)
    
    for i in range(1, len(time_array)):
        # 1. Quantized ON Ganglion Integration
        gain_input_on = apply_fixed_point(gain * q_on_bipolar[i])
        dv_on = apply_fixed_point((-v_on[i-1] + gain_input_on) * apply_fixed_point(dt / tau_m))
        v_on[i] = apply_fixed_point(v_on[i-1] + dv_on)
        
        if v_on[i] >= v_thresh:
            spikes_on[i] = 1
            v_on[i] = v_reset
            
        # 2. Quantized OFF Ganglion Integration
        gain_input_off = apply_fixed_point(gain * q_off_bipolar[i])
        dv_off = apply_fixed_point((-v_off[i-1] + gain_input_off) * apply_fixed_point(dt / tau_m))
        v_off[i] = apply_fixed_point(v_off[i-1] + dv_off)
        
        if v_off[i] >= v_thresh:
            spikes_off[i] = 1
            v_off[i] = v_reset
            
    return spikes_on, spikes_off

if __name__ == "__main__":
    # 1. Run the entire upstream pipelines
    time_array, luminance_input = generate_chirp_stimulus()
    
    # === REPLACE THE OLD IDEAL PIPELINE WITH THIS ===
    # Ideal Pipeline
    ideal_opl = simulate_opl(time_array, luminance_input)
    
    # 1. Bipolar layer returns a single unrectified voltage array
    ideal_bip = simulate_bipolar(time_array, ideal_opl)
    
    # 2. Extract ON and OFF spikes separately by setting the xi pathways (+1 and -1)
    _, ideal_on_spikes = simulate_ganglion(time_array, ideal_bip, xi=1)
    _, ideal_off_spikes = simulate_ganglion(time_array, ideal_bip, xi=-1)
    # ===============================================
    
    # Hardware Emulated Pipeline
    q_opl = simulate_quantized_opl(time_array, luminance_input)
    q_on_b, q_off_b = simulate_quantized_bipolar(time_array, q_opl)
    q_on_spikes, q_off_spikes = simulate_quantized_ganglion(time_array, q_on_b, q_off_b)
    
    # 2. Extract Timestamps
    ideal_on_times = time_array[ideal_on_spikes == 1]
    q_on_times = time_array[q_on_spikes == 1]
    
    # 3. Plot Comparison
    plt.figure(figsize=(10, 5))
    
    plt.subplot(2, 1, 1)
    plt.eventplot(ideal_on_times, orientation='horizontal', colors='green', lineoffsets=1, linelengths=0.5)
    plt.title('Ganglion Spike Output Comparison (ON Channel)')
    plt.ylabel('Ideal Float64')
    plt.xlim(0, time_array[-1])
    plt.grid(True, axis='x')
    plt.gca().get_yaxis().set_ticks([])
    
    plt.subplot(2, 1, 2)
    plt.eventplot(q_on_times, orientation='horizontal', colors='darkorange', lineoffsets=1, linelengths=0.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('16-bit Quantized')
    plt.xlim(0, time_array[-1])
    plt.grid(True, axis='x')
    plt.gca().get_yaxis().set_ticks([])
    
    plt.tight_layout()
    plt.show()
