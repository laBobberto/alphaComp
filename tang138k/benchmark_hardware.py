#!/usr/bin/env python3
import sys
import time
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml
import serial

def lecun_preprocess(img_28x28):
    coords = np.argwhere(img_28x28 > 0.1)
    if len(coords) < 3:
        return (img_28x28 > 0.25).astype(np.float32)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    h, w = (y1 - y0, x1 - x0)
    crop = img_28x28[y0:y1, x0:x1]
    max_dim = max(h, w)
    scale = 20.0 / max_dim
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    crop_pil = Image.fromarray((crop * 255).astype(np.uint8))
    resized = crop_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    resized_arr = np.array(resized, dtype=np.float32) / 255.0
    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized_arr
    total = np.sum(canvas)
    if total > 0:
        rows, cols = np.indices((28, 28))
        cy = np.sum(rows * canvas) / total
        cx = np.sum(cols * canvas) / total
        dy = max(-4, min(4, int(round(13.5 - cy))))
        dx = max(-4, min(4, int(round(13.5 - cx))))
        shifted = np.roll(np.roll(canvas, dy, axis=0), dx, axis=1)
        if dy > 0:
            shifted[:dy, :] = 0
        elif dy < 0:
            shifted[dy:, :] = 0
        if dx > 0:
            shifted[:, :dx] = 0
        elif dx < 0:
            shifted[:, dx:] = 0
        return (shifted > 0.25).astype(np.float32)
    return (canvas > 0.25).astype(np.float32)

def main():
    port = "/dev/ttyUSB1"
    baud = 2000000
    clk_freq = 50_000_000
    clks_per_bit = clk_freq // baud

    print("=" * 70)
    print("Tang Mega 138K FPGA Hardware Benchmark: 1,000 MNIST Test Samples")
    print(f"FPGA Device : Gowin GW5AST-LV138PG484AC1/I0")
    print(f"Core Clock  : {clk_freq / 1e6:.1f} MHz (T = {1e9 / clk_freq:.1f} ns)")
    print(f"Interface   : High-Speed UART @ {baud:,} baud ({port})")
    print("=" * 70)

    print("\n[1/3] Loading official MNIST test dataset...")
    mnist = fetch_openml('MNIST_784', version=1, as_frame=False)
    X_test_raw = mnist.data[60000:].astype(np.float32).reshape(-1, 28, 28)
    y_test = mnist.target[60000:].astype(np.int64)

    samples_per_digit = 100
    digit_indices = defaultdict(list)
    for idx, label in enumerate(y_test):
        if len(digit_indices[label]) < samples_per_digit:
            digit_indices[label].append(idx)
        if all(len(digit_indices[d]) >= samples_per_digit for d in range(10)):
            break

    print(f"[2/3] Preprocessing {samples_per_digit * 10} samples ({samples_per_digit} per digit 0..9)...")
    test_set = []
    for d in range(10):
        for idx in digit_indices[d]:
            raw_img = (X_test_raw[idx] > 35.0).astype(np.float32)
            bin_img = lecun_preprocess(raw_img).flatten()
            pat_str = "".join("1" if p > 0.5 else "0" for p in bin_img)
            test_set.append({"label": d, "pattern": pat_str})

    print(f"[3/3] Opening hardware serial port {port} @ {baud} baud...")
    ser = serial.Serial(port, baudrate=baud, timeout=1.0)
    time.sleep(0.05)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print(f"Streaming {len(test_set)} images to FPGA accelerator on hardware...\n")

    stats = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "latencies": [],
        "ttfs_list": [],
        "core_cycles": [],
        "total_fpga_cycles": []
    })

    t_start_all = time.perf_counter()

    for idx, item in enumerate(test_set):
        label = item["label"]
        pat_str = item["pattern"]
        payload = (pat_str + "\n").encode("ascii")

        t0 = time.perf_counter()
        ser.write(payload)
        resp = ser.readline().decode("ascii", errors="ignore").strip()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0

        pred_match = re.search(r"Class:\s*(\d+)", resp)
        ttfs_match = re.search(r"TTFS:\s*(\d+)", resp)

        pred_digit = int(pred_match.group(1)) if pred_match else -1
        ttfs_val = int(ttfs_match.group(1)) if ttfs_match else 16

        uart_rx_cycles = 785 * 10 * clks_per_bit
        core_cycles = 26 + ttfs_val
        uart_tx_cycles = 21 * 10 * clks_per_bit
        total_hw_cycles = uart_rx_cycles + core_cycles + uart_tx_cycles

        st = stats[label]
        st["total"] += 1
        if pred_digit == label:
            st["correct"] += 1
        st["latencies"].append(latency_ms)
        st["ttfs_list"].append(ttfs_val)
        st["core_cycles"].append(core_cycles)
        st["total_fpga_cycles"].append(total_hw_cycles)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1:4d} / {len(test_set)} samples (Digit {label} completed)...")

    ser.close()
    t_end_all = time.perf_counter()
    total_time = t_end_all - t_start_all

    print("\n" + "=" * 90)
    print("TANG MEGA 138K HARDWARE INFERENCE BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Digit':<6} | {'Samples':<8} | {'Correct':<8} | {'Acc (%)':<8} | {'Avg Lat (ms)':<13} | {'Avg TTFS':<9} | {'Core Cycles':<12} | {'FPGA Cycles':<12}")
    print("-" * 90)

    total_samples = 0
    total_correct = 0
    all_latencies = []
    all_ttfs = []
    all_core_cycles = []
    all_fpga_cycles = []

    for d in range(10):
        st = stats[d]
        tot = st["total"]
        cor = st["correct"]
        acc = (cor / tot) * 100.0 if tot > 0 else 0.0
        avg_lat = np.mean(st["latencies"])
        avg_ttfs = np.mean(st["ttfs_list"])
        avg_core = np.mean(st["core_cycles"])
        avg_fpga = np.mean(st["total_fpga_cycles"])

        total_samples += tot
        total_correct += cor
        all_latencies.extend(st["latencies"])
        all_ttfs.extend(st["ttfs_list"])
        all_core_cycles.extend(st["core_cycles"])
        all_fpga_cycles.extend(st["total_fpga_cycles"])

        print(f"  {d:<4} | {tot:<8} | {cor:<8} | {acc:<8.1f} | {avg_lat:<13.2f} | {avg_ttfs:<9.2f} | {avg_core:<12.1f} | {avg_fpga:<12.0f}")

    print("-" * 90)
    overall_acc = (total_correct / total_samples) * 100.0
    print(f"{'TOTAL':<6} | {total_samples:<8} | {total_correct:<8} | {overall_acc:<8.2f} | {np.mean(all_latencies):<13.2f} | {np.mean(all_ttfs):<9.2f} | {np.mean(all_core_cycles):<12.1f} | {np.mean(all_fpga_cycles):<12.0f}")
    print("=" * 90)

    print("\n--- TIMING & CYCLE BREAKDOWN (at 50 MHz core clock) ---")
    print(f"1. FPGA Core Computation (SNN):")
    print(f"   - Average Core Cycles : {np.mean(all_core_cycles):.1f} cycles")
    print(f"   - Average Core Time   : {np.mean(all_core_cycles) * (1e6 / clk_freq):.3f} µs ({np.mean(all_core_cycles) * 20:.1f} ns)")
    print(f"2. FPGA Hardware Communication:")
    print(f"   - UART RX (785 bytes) : 196,250 cycles ({196250 * 20 / 1e3:.2f} µs = 3.925 ms)")
    print(f"   - UART TX (21 bytes)  : 5,250 cycles ({5250 * 20 / 1e3:.2f} µs = 0.105 ms)")
    print(f"   - Total FPGA Cycles   : {np.mean(all_fpga_cycles):.0f} cycles ({np.mean(all_fpga_cycles) * 20 / 1e3:.2f} µs = 4.03 ms)")
    print(f"3. Full Roundtrip Measured over USB-UART:")
    print(f"   - Average Latency     : {np.mean(all_latencies):.2f} ms")
    print(f"   - Min / Max Latency   : {np.min(all_latencies):.2f} ms / {np.max(all_latencies):.2f} ms")
    print(f"   - OS/USB Buffer Overhead: ~{np.mean(all_latencies) - 4.03:.2f} ms")
    print(f"   - Throughput          : {total_samples / total_time:.1f} inferences/sec (including all OS roundtrip overhead)")
    print("=" * 90)

if __name__ == "__main__":
    main()
