import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from retina_params import SIGMA_C, SIGMA_S, OMEGA_OPL, LAMBDA_OPL, GL, apply_19bit_fxp

def run_spatial_replication():
    # 1. Create a 2D Spatial Stimulus Frame (128x128 pixels)
    # Simulates a high-contrast square object against a dark background
    frame_size = 128
    stimulus_frame = np.zeros((frame_size, frame_size))
    stimulus_frame[32:96, 32:96] = 1.0  # Center square object
    
    # 2. OPL Layer Spatial Processing (Center-Surround Receptive Fields)
    # The spatial constants (sigmas) define the pixel width of the Gaussian kernels
    # Mapping degrees to pixel scale (approx 20 pixels per degree)
    pixel_scale = 40.0
    sig_c_px = SIGMA_C * pixel_scale
    sig_s_px = SIGMA_S * pixel_scale
    
    # Ideal Software Track
    sw_center = gaussian_filter(stimulus_frame, sigma=sig_c_px)
    sw_surround = gaussian_filter(stimulus_frame, sigma=sig_s_px)
    sw_opl = LAMBDA_OPL * (sw_center - OMEGA_OPL * sw_surround)
    
    # Emulated 19-bit Hardware Track
    hw_center = apply_19bit_fxp(gaussian_filter(stimulus_frame, sigma=sig_c_px))
    hw_surround = apply_19bit_fxp(gaussian_filter(stimulus_frame, sigma=sig_s_px))
    hw_opl = apply_19bit_fxp(LAMBDA_OPL * (hw_center - OMEGA_OPL * hw_surround))
    
    # 3. Bipolar Layer Spatial Contrast Control
    # Simple spatial smoothing modeling the bipolar layer pooling (SIGMA_A)
    sig_a_px = 0.05 * pixel_scale
    sw_bip = gaussian_filter(sw_opl, sigma=sig_a_px)
    hw_bip = apply_19bit_fxp(gaussian_filter(hw_opl, sigma=sig_a_px))
    
    # 4. Ganglion Layer Spatial Edge Activation (ON and OFF channels)
    # Extract positive values for ON responses and negative values for OFF responses
    sw_gang_on = np.maximum(0, sw_bip)
    sw_gang_off = np.maximum(0, -sw_bip)
    
    hw_gang_on = apply_19bit_fxp(np.maximum(0, hw_bip))
    hw_gang_off = apply_19bit_fxp(np.maximum(0, -hw_bip))
    
    # 5. Plot the 2D Framework Visual Output Matrix
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # --- TOP ROW: IDEAL SOFTWARE SIMULATION ---
    axes[0, 0].imshow(stimulus_frame, cmap='gray', origin='lower')
    axes[0, 0].set_title('Input Stimulus Frame')
    axes[0, 0].set_ylabel('Software Model', fontsize=12, fontweight='bold')
    
    im1 = axes[0, 1].imshow(sw_opl, cmap='coolwarm', origin='lower')
    axes[0, 1].set_title('OPL Output (Bipolar Input)')
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    im2 = axes[0, 2].imshow(sw_gang_on, cmap='hot', origin='lower')
    axes[0, 2].set_title('Ganglion ON Response Map')
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    im3 = axes[0, 3].imshow(sw_gang_off, cmap='bone', origin='lower')
    axes[0, 3].set_title('Ganglion OFF Response Map')
    fig.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)
    
    # --- BOTTOM ROW: EMULATED 19-BIT FIXED-POINT HARDWARE ---
    axes[1, 0].imshow(stimulus_frame, cmap='gray', origin='lower')
    axes[1, 0].set_title('Input Stimulus Frame')
    axes[1, 0].set_ylabel('19-bit Fixed-Point Hardware', fontsize=12, fontweight='bold')
    
    im4 = axes[1, 1].imshow(hw_opl, cmap='coolwarm', origin='lower')
    axes[1, 1].set_title('OPL Output (Hardware)')
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    im5 = axes[1, 2].imshow(hw_gang_on, cmap='hot', origin='lower')
    axes[1, 2].set_title('Ganglion ON Map (Hardware)')
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    # Apply a subtraction delta to verify perfect bitwise equivalence visually
    spatial_delta = np.abs(sw_gang_off - hw_gang_off)
    im6 = axes[1, 3].imshow(spatial_delta, cmap='magma', origin='lower')
    axes[1, 3].set_title('Bitwise Delta Verification Error')
    fig.colorbar(im6, ax=axes[1, 3], fraction=0.046, pad=0.04)
    
    # Clean up formatting across axes grids
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_spatial_replication()