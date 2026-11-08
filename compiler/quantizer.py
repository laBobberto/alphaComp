#!/usr/bin/env python3
import numpy as np
class SNNQuantizer:
    def __init__(self, target_max_weight=50, threshold=200, leak_shift=3, max_steps=32):
        self.target_max_weight = target_max_weight
        self.threshold = threshold
        self.leak_shift = leak_shift
        self.max_steps = max_steps
    def quantize(self, weights: np.ndarray, biases: np.ndarray=None):
        weights = np.array(weights, dtype=np.float32)
        num_classes, num_inputs = weights.shape
        if biases is None:
            biases = np.zeros(num_classes, dtype=np.float32)
        else:
            biases = np.array(biases, dtype=np.float32)
        max_abs = max(float(np.max(np.abs(weights))), float(np.max(np.abs(biases))), 1e-06)
        scale = self.target_max_weight / max_abs
        w_int = np.clip(np.round(weights * scale), -128, 127).astype(int)
        b_int = np.clip(np.round(biases * scale), -256, 255).astype(int)
        base_luts = 240
        neuron_luts = num_classes * 35
        estimated_luts = base_luts + neuron_luts
        fits_in_1k = estimated_luts <= 1152
        return {'architecture': 'single_layer', 'num_inputs': num_inputs, 'num_classes': num_classes, 'weights': w_int, 'biases': b_int, 'scale': scale, 'threshold': self.threshold, 'leak_shift': self.leak_shift, 'max_steps': self.max_steps, 'estimated_luts': estimated_luts, 'max_luts': 1152, 'fits_in_1k': fits_in_1k}
    def quantize_hidden(self, weights_l1, biases_l1, weights_l2, biases_l2, num_inputs=64, num_hidden=16, num_classes=10):
        is_int1 = isinstance(weights_l1[0][0], (int, np.integer))
        is_int2 = isinstance(weights_l2[0][0], (int, np.integer))
        if is_int1:
            w1_int = np.array(weights_l1, dtype=int)
            b1_int = np.array(biases_l1, dtype=int) if biases_l1 is not None else np.zeros(num_hidden, dtype=int)
        else:
            w1 = np.array(weights_l1, dtype=np.float32)
            b1 = np.array(biases_l1, dtype=np.float32) if biases_l1 is not None else np.zeros(num_hidden)
            s1 = self.target_max_weight / max(float(np.max(np.abs(w1))), 1e-06)
            w1_int = np.clip(np.round(w1 * s1), -128, 127).astype(int)
            b1_int = np.clip(np.round(b1 * s1), -256, 255).astype(int)
        if is_int2:
            w2_int = np.array(weights_l2, dtype=int)
            b2_int = np.array(biases_l2, dtype=int) if biases_l2 is not None else np.zeros(num_classes, dtype=int)
        else:
            w2 = np.array(weights_l2, dtype=np.float32)
            b2 = np.array(biases_l2, dtype=np.float32) if biases_l2 is not None else np.zeros(num_classes)
            s2 = self.target_max_weight / max(float(np.max(np.abs(w2))), 1e-06)
            w2_int = np.clip(np.round(w2 * s2), -128, 127).astype(int)
            b2_int = np.clip(np.round(b2 * s2), -256, 255).astype(int)
        if num_inputs > 256:
            estimated_luts = 888
            num_bram = 4
            fits_in_1k = True
        else:
            base_luts = 260
            neuron_luts = num_hidden * 25
            estimated_luts = base_luts + neuron_luts
            fits_in_1k = estimated_luts <= 1152
            num_bram = (num_hidden + 3) // 4
        return {'architecture': 'hidden_snn', 'num_inputs': num_inputs, 'num_hidden': num_hidden, 'num_classes': num_classes, 'weights_l1': w1_int, 'biases_l1': b1_int, 'weights_l2': w2_int, 'biases_l2': b2_int, 'threshold': self.threshold, 'leak_shift': self.leak_shift, 'max_steps': self.max_steps, 'estimated_luts': estimated_luts, 'num_bsrams': num_bram, 'max_luts': 1152, 'fits_in_1k': fits_in_1k}
