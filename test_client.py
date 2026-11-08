#!/usr/bin/env python3
import sys
import time
import argparse
import numpy as np
import serial
def test_fpga(port, baudrate, image_path=None):
    ser = serial.Serial(port, baudrate=baudrate, timeout=2.0)
    time.sleep(0.1)
    ser.reset_input_buffer()
    if image_path:
        data = np.load(image_path).flatten()
    else:
        data = np.zeros(784, dtype=np.uint8)
    raw_bytes = bytes([1 if v > 0 else 0 for v in data])
    start_t = time.perf_counter()
    ser.write(raw_bytes)
    resp = ser.readline().decode('utf-8', errors='ignore').strip()
    latency = time.perf_counter() - start_t
    print(f'FPGA Response: {resp}')
    print(f'Roundtrip Latency: {latency * 1000:.2f} ms')
    ser.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default='/dev/ttyUSB1')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--image', default=None)
    args = parser.parse_args()
    test_fpga(args.port, args.baud, args.image)
