import numpy as np
import matplotlib.pyplot as plt
import retina_core

def run_temporal_step_experiment():
    print("🔬 Starting Simulation: Full-Field Temporal Step Response...")
    
    # 1. Setup Simulation Time (1000 milliseconds)
    total_time_ms = 1000  
    dt = retina_core.DT
    time_steps = int(total_time_ms / (dt * 1000))
    time_axis = np.linspace(0, total_time_ms, time_steps)
    
    # 2. Setup Spatial Area (A small 16x16 pixel patch is enough for a full-field flash)
    spatial_size = 16
    
    # 3. Initialize State Variables mathematically
    prev_E_tauC = np.zeros((spatial_size, spatial_size))
    prev_prev_E_tauC = np.zeros((spatial_size, spatial_size))
    prev_T_w_tau = np.zeros((spatial_size, spatial_size))
    prev_E_tauS = np.zeros((spatial_size, spatial_size))
    
    prev_V_Bip = np.zeros((spatial_size, spatial_size))
    prev_att_map = np.ones((spatial_size, spatial_size))
    prev_E_A = np.zeros((spatial_size, spatial_size))
    
    prev_T_G = np.zeros((spatial_size, spatial_size))
    
    V_m_ON = np.zeros((spatial_size, spatial_size))
    rt_ON = np.zeros((spatial_size, spatial_size))
    V_m_OFF = np.zeros((spatial_size, spatial_size))
    rt_OFF = np.zeros((spatial_size, spatial_size))
    
    # Arrays to log the data for our graphs
    log_stimulus = []
    log_OPL = []
    log_Bip = []
    log_Spikes_ON = []
    log_Spikes_OFF = []
    
    print("⏳ Running simulation loop. This will take a few seconds...")
    # 4. Run the Simulation Loop strictly over time
    for t in range(time_steps):
        # Create the Stimulus: Dark -> Light (at 200ms) -> Dark (at 600ms)
        if 200 <= time_axis[t] <= 600:
            intensity = 1.0  # Light ON
        else:
            intensity = 0.0  # Light OFF
            
        input_frame = np.full((spatial_size, spatial_size), intensity)
        log_stimulus.append(intensity)
        
        # --- Pass through Retina Core Simulator ---
        I_OPL, E_tauC, T_w_tau, E_tauS = retina_core.OPLlayer(
            input_frame, prev_E_tauC, prev_prev_E_tauC, prev_T_w_tau, prev_E_tauS
        )
        V_Bip, att_map, E_A = retina_core.Bipolar(
            I_OPL, prev_V_Bip, prev_att_map, prev_E_A
        )
        I_Gang_ON, I_Gang_OFF, T_G = retina_core.ganglion_current_dual(
            V_Bip, prev_V_Bip, prev_T_G, xi=15.0
        )
        Spikes_ON, V_m_ON, rt_ON = retina_core.spiking(I_Gang_ON, V_m_ON, rt_ON)
        Spikes_OFF, V_m_OFF, rt_OFF = retina_core.spiking(I_Gang_OFF, V_m_OFF, rt_OFF)
        
        # --- Log the center pixel data for the graph ---
        center = spatial_size // 2
        log_OPL.append(I_OPL[center, center])
        log_Bip.append(V_Bip[center, center])
        log_Spikes_ON.append(Spikes_ON[center, center])
        log_Spikes_OFF.append(Spikes_OFF[center, center])
        
        # --- State Updates ---
        prev_prev_E_tauC, prev_E_tauC = prev_E_tauC, E_tauC
        prev_T_w_tau, prev_E_tauS = T_w_tau, E_tauS
        prev_V_Bip, prev_att_map, prev_E_A = V_Bip, att_map, E_A
        prev_T_G = T_G

    print("📊 Plotting results...")
    # 5. Plot the Results strictly according to standard paper formats
    fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Simulation Result: Temporal Step Response", fontsize=14, fontweight='bold')
    
    axs[0].plot(time_axis, log_stimulus, 'k-', lw=2)
    axs[0].set_ylabel("Input\nLight")
    axs[0].grid(True)
    
    axs[1].plot(time_axis, log_OPL, 'b-', lw=2)
    axs[1].set_ylabel("OPL\nOutput")
    axs[1].grid(True)
    
    axs[2].plot(time_axis, log_Bip, 'g-', lw=2)
    axs[2].set_ylabel("Bipolar\nVoltage")
    axs[2].grid(True)
    
    # Raster plot for ON spikes
    spike_times_ON = [time_axis[i] for i, s in enumerate(log_Spikes_ON) if s]
    axs[3].vlines(spike_times_ON, 0, 1, colors='green', lw=1.5)
    axs[3].set_ylabel("ON\nSpikes")
    axs[3].set_yticks([])
    axs[3].grid(True)
    
    # Raster plot for OFF spikes
    spike_times_OFF = [time_axis[i] for i, s in enumerate(log_Spikes_OFF) if s]
    axs[4].vlines(spike_times_OFF, 0, 1, colors='red', lw=1.5)
    axs[4].set_ylabel("OFF\nSpikes")
    axs[4].set_yticks([])
    axs[4].set_xlabel("Time (ms)")
    axs[4].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_temporal_step_experiment()