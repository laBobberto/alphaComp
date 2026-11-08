#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from compiler.quantizer import SNNQuantizer
from compiler.codegen import VerilogGenerator
from compiler.builder import FPGABuilder
def get_demo_3x3_weights():
    weights = np.array([[25, -17, 26, 39, -40, 33, 33, -12, 28], [-39, 16, -41, -26, 50, -29, -32, 14, -40], [35, -7, 29, -40, 32, -47, 30, -6, 30], [29, -18, 35, -31, 40, 37, 23, -6, 30]], dtype=np.float32)
    biases = np.array([-40, 60, -22, -55], dtype=np.float32)
    return (weights, biases)
def get_demo_4x4_weights():
    np.random.seed(42)
    weights = np.random.randn(4, 16).astype(np.float32) * 20.0
    biases = np.zeros(4, dtype=np.float32)
    return (weights, biases)
def generate_host_test_script(out_dir: Path, num_inputs: int, num_classes: int):
    script_path = out_dir / 'test_client.py'
    content = f'''#!/usr/bin/env python3\n"""\nAuto-generated FPGA UART Test Client\nInput length: {num_inputs} bits, Number of classes: {num_classes}\n"""\n\nimport sys\nimport time\nimport re\nimport serial\n\nPORT = "/dev/ttyUSB1"\nBAUD = 115200\n\ndef test_inference(pattern_bits: str):\n    if len(pattern_bits) != {num_inputs}:\n        raise ValueError(f"Pattern must be exactly {num_inputs} bits long (got {{len(pattern_bits)}})")\n\n    print(f"Connecting to {{PORT}} @ {{BAUD}} baud...")\n    try:\n        ser = serial.Serial(PORT, BAUD, timeout=0.5)\n    except Exception as e:\n        print(f"Error opening serial port {{PORT}}: {{e}}")\n        return\n\n    ser.reset_input_buffer()\n    payload = (pattern_bits + "\\n").encode("ascii")\n    print(f"Sending ({{len(pattern_bits)}} bits): {{pattern_bits}}")\n    ser.write(payload)\n    ser.flush()\n\n    time.sleep(0.05)\n    resp = ser.read(ser.in_waiting or 64).decode("ascii", errors="replace").strip()\n    print(f"FPGA Raw Response: '{{resp}}'")\n\n    m = re.search(r"Class:\\s*(\\d+)\\s*\\(TTFS:\\s*(\\d+)\\)", resp)\n    if m:\n        c = int(m.group(1))\n        ttfs = int(m.group(2))\n        latency_ns = (ttfs + {num_inputs}) * (1000.0 / 27.0)\n        print(f"Prediction: Class {{c}} | TTFS: {{ttfs}} steps | Latency: ~{{latency_ns:.1f}} ns")\n    else:\n        print("Could not parse class/TTFS from response.")\n\n    ser.close()\n\nif __name__ == "__main__":\n    if len(sys.argv) > 1:\n        pattern = sys.argv[1]\n    else:\n        # Default test pattern (all ones)\n        pattern = "1" * {num_inputs}\n    test_inference(pattern)\n'''
    with open(script_path, 'w') as f:
        f.write(content)
    script_path.chmod(493)
    print(f'==> Generated host UART test script: {script_path}')
def main():
    parser = argparse.ArgumentParser(description='betaComp: SNN Compiler & Translator for Tang Nano 1K')
    parser.add_argument('--demo', choices=['3x3', '4x4'], default='3x3', help='Use built-in demo model')
    parser.add_argument('--weights', type=str, help='Path to JSON file containing weights and biases')
    parser.add_argument('--threshold', type=int, default=200, help='LIF firing threshold (default: 200)')
    parser.add_argument('--leak-shift', type=int, default=3, help='LIF leak shift (default: 3)')
    parser.add_argument('--max-steps', type=int, default=32, help='Max TTFS simulation cycles (default: 32)')
    parser.add_argument('--out-dir', type=str, default='output', help='Directory for generated RTL')
    parser.add_argument('--build-dir', type=str, default='build', help='Directory for synthesis/pnr artifacts')
    parser.add_argument('--build', action='store_true', help='Run full Yosys + NextPNR synthesis pipeline')
    parser.add_argument('--flash', action='store_true', help='Flash bitstream to connected Tang Nano 1K')
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / args.out_dir
    build_dir = base_dir / args.build_dir
    templates_dir = base_dir / 'rtl' if (base_dir / 'rtl').exists() else base_dir / 'templates'
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    quantizer = SNNQuantizer(target_max_weight=50, threshold=args.threshold, leak_shift=args.leak_shift, max_steps=args.max_steps)
    is_hidden = False
    if args.weights:
        print(f'==> Loading model weights from: {args.weights}')
        with open(args.weights, 'r') as f:
            data = json.load(f)
        if 'weights_l1' in data or data.get('architecture') == 'hidden_snn':
            is_hidden = True
            num_in = data.get('num_inputs', 64)
            num_hid = data.get('num_hidden', 16)
            num_cls = data.get('num_classes', 10)
            cfg = quantizer.quantize_hidden(data['weights_l1'], data.get('biases_l1'), data['weights_l2'], data.get('biases_l2'), num_inputs=num_in, num_hidden=num_hid, num_classes=num_cls)
        else:
            weights = np.array(data['weights'], dtype=np.float32)
            biases = np.array(data.get('biases', [0] * len(weights)), dtype=np.float32)
            cfg = quantizer.quantize(weights, biases)
    else:
        print(f'==> Using built-in demo dataset ({args.demo})')
        if args.demo == '3x3':
            weights, biases = get_demo_3x3_weights()
        else:
            weights, biases = get_demo_4x4_weights()
        cfg = quantizer.quantize(weights, biases)
    print('\n' + '=' * 55)
    print('STEP 1: SNN Quantization & Tang Nano 1K Resource Analysis')
    print('=' * 55)
    if is_hidden:
        print(f'• Architecture: 2-Layer Time-Multiplexed Spiking Network')
        print(f"• Sensory Layer: {cfg['num_inputs']} inputs -> {cfg['num_hidden']} hidden neurons ({cfg['num_inputs'] * cfg['num_hidden']} synapses)")
        print(f"• Action Potentials: {cfg['num_hidden']}-bit discrete spike latch")
        print(f"• Decision Layer: {cfg['num_hidden']} hidden -> {cfg['num_classes']} output classes ({cfg['num_hidden'] * cfg['num_classes']} synapses)")
        print(f"• Gowin BSRAM Usage: {cfg['num_bsrams']} / 4 blocks ({cfg['num_bsrams'] * 25}%)")
        print(f"• Estimated Latency: {cfg['num_inputs'] + cfg['num_hidden'] + 5} cycles @ 27 MHz (~{(cfg['num_inputs'] + cfg['num_hidden'] + 5) * (1000 / 27.0):.1f} ns)")
    else:
        print(f'• Architecture: Single-Layer Crossbar SNN')
        print(f"• Inputs: {cfg['num_inputs']}, Classes: {cfg['num_classes']}")
        print(f"• Synapses: {cfg['num_inputs'] * cfg['num_classes']}")
        print(f"• Quantization Scale: {cfg['scale']:.2f}")
    print(f"• Estimated LUTs: ~{cfg['estimated_luts']} / {cfg['max_luts']} ({cfg['estimated_luts'] * 100 / cfg['max_luts']:.1f}%)")
    if not cfg['fits_in_1k']:
        print(f'WARNING: Network exceeds standard LUT capacity of Tang Nano 1K!')
    else:
        print(f'Resource Check: Network fits comfortably inside Tang Nano 1K!')
    print('\n' + '=' * 55)
    print('STEP 2: Synthesizable Verilog & RTL Generation')
    print('=' * 55)
    generator = VerilogGenerator(cfg)
    snn_core_v = generator.generate_snn_core()
    top_v = generator.generate_top()
    snn_core_path = out_dir / 'snn_core.v'
    top_path = out_dir / 'top.v'
    with open(snn_core_path, 'w') as f:
        f.write(snn_core_v)
    with open(top_path, 'w') as f:
        f.write(top_v)
    for mod in ['lif_neuron.v', 'uart_rx.v', 'uart_tx.v']:
        if (templates_dir / mod).exists():
            shutil.copy(templates_dir / mod, out_dir / mod)
    cst_src = base_dir / 'tangnano1k.cst' if (base_dir / 'tangnano1k.cst').exists() else templates_dir / 'tangnano1k.cst'
    if cst_src.exists():
        shutil.copy(cst_src, out_dir / 'tangnano1k.cst')
    print(f'• Generated: {snn_core_path}')
    print(f'• Generated: {top_path}')
    print(f'• Copied templates: lif_neuron.v, uart_rx.v, uart_tx.v, tangnano1k.cst')
    generate_host_test_script(out_dir, cfg['num_inputs'], cfg['num_classes'])
    if args.build:
        print('\n' + '=' * 55)
        print('STEP 3: FPGA Synthesis & Place-and-Route (OSS CAD Suite)')
        print('=' * 55)
        builder = FPGABuilder(build_dir=build_dir)
        verilog_files = [str(top_path.resolve()), str(snn_core_path.resolve()), str((out_dir / 'uart_rx.v').resolve()), str((out_dir / 'uart_tx.v').resolve())]
        if not is_hidden:
            verilog_files.append(str((out_dir / 'lif_neuron.v').resolve()))
        cst_file = str((out_dir / 'tangnano1k.cst').resolve())
        bitstream_fs = builder.build(verilog_files, cst_file)
        if args.flash:
            print('\n' + '=' * 55)
            print('STEP 4: Program Tang Nano 1K Board')
            print('=' * 55)
            builder.flash(bitstream_fs)
    print('\n' + '=' * 55)
    print('COMPILATION FINISHED!')
    print(f'Artifacts located in: {out_dir.resolve()}')
    print('=' * 55)
if __name__ == '__main__':
    main()
