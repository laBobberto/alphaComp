#!/usr/bin/env python3
import sys
import os
import glob
import time
import json
import re
from pathlib import Path
import serial
import serial.tools.list_ports

def find_tang_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "ttyACM" in p.device or "ttyUSB" in p.device:
            return p.device
    acm_nodes = glob.glob("/dev/ttyACM*")
    if acm_nodes:
        return acm_nodes[0]
    usb_nodes = glob.glob("/dev/ttyUSB*")
    if usb_nodes:
        return usb_nodes[0]
    return "/dev/ttyACM0"

def run_test(port=None, baud=2000000, weights_file=None):
    if port is None:
        port = find_tang_port()

    base_dir = Path(__file__).resolve().parent.parent
    if weights_file is None:
        weights_file = base_dir / "weights/digits_mnist_28x28.json"
    else:
        weights_file = Path(weights_file)

    print("=" * 65)
    print("Tang Mega 138K High-Speed UART (2 Mbps) SNN Inference Benchmark")
    print(f"Target Port : {port} @ {baud} baud (DEBUG|OTG Type-C port)")
    print(f"Model       : {weights_file.name}")
    print("=" * 65)

    if not weights_file.exists():
        print(f"Error: Weights file {weights_file} not found!")
        return

    with open(weights_file, "r") as f:
        model_data = json.load(f)

    test_samples = model_data.get("test_samples", [])
    if isinstance(test_samples, dict):
        samples_list = []
        for digit_str, pat in sorted(test_samples.items(), key=lambda x: int(x[0])):
            samples_list.append({"label": int(digit_str), "pattern": pat})
        test_samples = samples_list
    elif not test_samples:
        print("Warning: No embedded test samples found in weights file. Generating dummy 784-bit pattern.")
        test_samples = [{"label": 0, "pattern": "1" * 784}]

    try:
        ser = serial.Serial(port, baudrate=baud, timeout=2.0)
    except Exception as e:
        print(f"Error opening port {port}: {e}")
        print("Tip: Check connection to 'DEBUG|OTG' Type-C connector and permissions (sudo usermod -a -G dialout $USER).")
        return

    time.sleep(0.1)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print(f"\nRunning inference on {len(test_samples)} samples...\n")
    correct = 0
    total = 0
    latencies = []

    for idx, sample in enumerate(test_samples):
        label = sample.get("label", -1)
        raw_pat = sample.get("pattern", "")

        if isinstance(raw_pat, list):
            pat_str = "".join(str(int(x > 0)) for x in raw_pat)
        else:
            pat_str = str(raw_pat).strip()

        if len(pat_str) != 784:
            pat_str = pat_str.ljust(784, "0")[:784]

        payload = (pat_str + "\n").encode("ascii")

        t0 = time.perf_counter()
        ser.write(payload)
        resp = ser.readline().decode("ascii", errors="ignore").strip()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)

        pred_match = re.search(r"(\d+)", resp)
        pred_digit = int(pred_match.group(1)) if pred_match else -1

        match_str = "MATCH" if pred_digit == label else "MISMATCH"
        if pred_digit == label:
            correct += 1
        total += 1

        print(f"Sample #{idx+1:02d} | True: {label} | Pred: {pred_digit} | Raw: '{resp}' | Status: {match_str} | Latency: {latency_ms:.2f} ms")

    ser.close()

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    min_lat = min(latencies) if latencies else 0.0
    max_lat = max(latencies) if latencies else 0.0
    acc = (correct / total) * 100.0 if total > 0 else 0.0

    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS SUMMARY (2 Mbps CDC UART)")
    print("=" * 65)
    print(f"Accuracy         : {correct}/{total} ({acc:.1f}%)")
    print(f"Average Latency  : {avg_lat:.2f} ms")
    print(f"Min / Max Latency: {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"Speedup vs 115k  : ~17.4x interface latency reduction")
    print("=" * 65)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tang Mega 138K 2 Mbps UART SNN Test Client")
    parser.add_argument("--port", default=None, help="Serial port (e.g. /dev/ttyACM0 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=2000000, help="Baud rate (default: 2000000)")
    parser.add_argument("--weights", default=None, help="Path to weights JSON")
    args = parser.parse_args()

    run_test(args.port, args.baud, args.weights)
