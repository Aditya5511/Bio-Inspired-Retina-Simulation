# Bio-Inspired Digital Retina Emulation Pipeline

This repository contains a comprehensive computational framework that replicates the multi-layered processing pathways of the biological vertebrate retina. The pipeline simulates visual processing across both standard software floating-point engines (`Float64`) and hardware-emulated configurations using custom 19-bit saturated fixed-point mathematics. 

The architecture simulates the outer plexiform layer (OPL), bipolar pathway, and ganglion cell spike generation across 1D time-series data and 2D spatial frameworks.

---

## 🚀 Features & Simulation Phases

The project runs a structured 5-phase replication pipeline orchestrated by a central dashboard controller:

*   **Phase 1: Bit-Quantization Limits (Fig. 6)** – Demonstrates exact staircase replication bounds under varying precision steps.
*   **Phase 2: Photoreceptor Transient & Chirp Responses (Fig. 7 & 8)** – Models full-field luminance, frequency sweeps, and amplitude modulation across Software vs. Hardware pathways.
*   **Phase 3: Mathematical Fidelity Metrics (Fig. 9)** – Dynamically calculates the true Ratio of Variance Explained (%) and evaluates hardware-to-software time lags using normalized Pearson cross-correlation profiles.
*   **Phase 4: 2D Spatial Frameworks (Fig. 10)** – Benchmarks spatial edge activations, center-surround receptive fields, and maps localized ON/OFF channel responses to compute bitwise delta verification matrices.
*   **Phase 5: Neural Spiking Mechanisms (Fig. 11 & Step Response)** – Utilizes a state-driven Leaky Integrate-and-Fire (LIF) simulation engine to evaluate transient (phasic) change-detection and sustained (tonic) spike train responses alongside full-field dark-to-light flash experiments.

### 🎥 Bonus Feature: Live Webcam Retina Simulation
Includes an active computer vision script that processes real-time webcam video arrays through the 19-bit saturated IIR filters, generating dynamic live visual matrices of ON and OFF ganglion spike channels.

---

## 📂 Repository Structure

### 🎛️ Master Control & Configuration
*   `main.py` - The primary orchestration file executing the complete pipeline.
*   `retina_params.py` - Core biophysical parameters (constants, time steps) and fixed-point Q8.11 quantization configurations.
*   `stimulus_generator.py` - Generates standardized time vectors, dark/light steps, and frequency/amplitude sweep chirp signals.

### 🧬 Biological Layer Processing Engines
*   `retina_core.py` - Main spatial processing library containing 2D convolutional layers, IIR high-pass/low-pass implementations, and the live camera engine.
*   `opl_layer.py` - 1D implementation of outer plexiform layer temporal cascades.
*   `bipolar_layer.py` - 1D implementation of amacrine shunting control and voltage profiles.
*   `ganglion_layer.py` - 1D implementation of dual-channel ganglion spiking dynamics.

### 📊 Plotting & Figure Drivers
*   `fig6_final_braid.py` - Replicates quantization staircases.
*   `replicate_fig7.py` - Models impulse tracking.
*   `replicate_fig8_chirp.py` - Renders the 8-panel software vs. hardware chirp comparison.
*   `replicate_fig9_metrics.py` - Evaluates statistical deviation and cross-correlation lags.
*   `replicate_fig10_spatial.py` - Maps 2D receptive filters.
*   `replicate_fig11_mechanisms.py` - Simulates tonic/phasic firing patterns via the LIF engine.
*   `replicate_temporal_response.py` - Logs center-pixel arrays during full-field flash protocols.

---

## 🛠️ Prerequisites & Installation

Ensure you have Python 3.8+ installed along with the required numerical and visualization dependencies:

```bash
pip install numpy matplotlib scipy opencv-python