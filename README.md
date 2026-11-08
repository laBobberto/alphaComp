# alphaComp

Hardware compiler and streaming inference engine for Spiking Neural Networks on Gowin FPGAs (Sipeed Tang Nano 1K).

## Architecture

- Model: Two-layer fully-connected SNN (784 inputs -> 12 LIF neurons -> 10 output classes)
- Quantization: 6-bit signed weights (INT6 in [-31, +31]), 10-bit biases (INT10)
- Memory packing: 3:1 packing of 6-bit weights into 18-bit BSRAM words (4 blocks, 100% capacity utilization)
- Neuron arithmetic: Leaky Integrate-and-Fire with shift-leak (beta = 0.75 via v - (v >> 2)), 0 DSP blocks used
- Latency detection: Time-to-First-Spike (TTFS) ramp
- I/O: Streaming UART receiver and transmitter (115200 baud, 8N1)

## Prerequisites

- Python 3.10+
- OSS CAD Suite (yosys, nextpnr-himbaechel, gowin_pack, openFPGALoader)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Compile Verilog and Build Bitstream

Compile pre-trained MNIST weights into Verilog RTL and synthesize bitstream:

```bash
./build.sh
```

Or run the compiler directly:

```bash
python3 compile.py --weights weights/digits_mnist_28x28.json --build
```

The resulting bitstream is written to `build/top.fs`.

### 2. Flash to FPGA

Connect Sipeed Tang Nano 1K via USB and run:

```bash
./flash.sh

./flash.sh -f
```

### 3. Test Inference via UART

Send a test image over UART and read the classification result:

```bash
python3 test_client.py --port /dev/ttyUSB1 --baud 115200
```

### 4. Train from Scratch (Optional)

To train a new SNN model on MNIST with surrogate gradient Heaviside and BatchNorm folding:

```bash
python3 train.py
```

The quantized weights will be exported to `weights/digits_mnist_28x28.json`.

## Hardware Utilization (Tang Nano 1K / GW1NZ-1)

- BSRAM: 4 / 4 blocks (100.0%)
- Logic LUT4: 993 / 1152 cells (86.2%)
- Flip-Flops (DFF): 277 / 864 registers (32.1%)
- Multipliers (DSP): 0 / 0 blocks (0.0%)
