#!/usr/bin/env python3
import os
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
from sklearn.datasets import fetch_openml
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
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
class SurrHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0.0).float()
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sg = torch.sigmoid(5.0 * x) * (1.0 - torch.sigmoid(5.0 * x)) * 5.0
        return grad_output * sg
spike_fn = SurrHeaviside.apply
class SNN_28x28(nn.Module):
    def __init__(self, num_hidden=12):
        super().__init__()
        self.fc1 = nn.Linear(784, num_hidden, bias=True)
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.fc2 = nn.Linear(num_hidden, 10, bias=True)
    def forward(self, x):
        v1 = self.bn1(self.fc1(x))
        s1 = spike_fn(v1)
        out = self.fc2(s1)
        return (out, s1)
def train():
    mnist = fetch_openml('MNIST_784', version=1, as_frame=False)
    X_all = mnist.data.astype(np.float32)
    y_all = mnist.target.astype(np.int64)
    X_train_raw = X_all[:60000].reshape(-1, 28, 28)
    y_train_raw = y_all[:60000]
    X_test_raw = X_all[60000:].reshape(-1, 28, 28)
    y_test = y_all[60000:]
    X_aug = []
    y_aug = []
    for i in range(25000):
        img_bin = (X_train_raw[i] > 35.0).astype(np.float32)
        norm = lecun_preprocess(img_bin)
        X_aug.append(norm.flatten())
        y_aug.append(y_train_raw[i])
    X_train_arr = np.array(X_aug, dtype=np.float32)
    y_train_arr = np.array(y_aug, dtype=np.int64)
    X_test_arr = []
    for img in (X_test_raw > 35.0).astype(np.float32):
        X_test_arr.append(lecun_preprocess(img).flatten())
    X_test_arr = np.array(X_test_arr, dtype=np.float32)
    torch.manual_seed(42)
    np.random.seed(42)
    num_hidden = 12
    model = SNN_28x28(num_hidden=num_hidden)
    opt = optim.AdamW(model.parameters(), lr=0.008, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    X_tr_t = torch.tensor(X_train_arr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train_arr, dtype=torch.long)
    X_te_t = torch.tensor(X_test_arr, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)
    batch_size = 512
    n_batches = len(X_train_arr) // batch_size
    epochs = 15
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_arr))
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            opt.zero_grad()
            out, _ = model(X_tr_t[idx])
            loss = loss_fn(out, y_tr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out_te, _ = model(X_te_t)
            test_acc = (out_te.argmax(1) == y_te_t).float().mean().item()
        print(f'Epoch {ep + 1}/{epochs} - Test Acc: {test_acc * 100:.2f}%')
    gamma = model.bn1.weight.detach().numpy()
    beta = model.bn1.bias.detach().numpy()
    mean = model.bn1.running_mean.detach().numpy()
    var = model.bn1.running_var.detach().numpy()
    eps = model.bn1.eps
    scale_bn = gamma / np.sqrt(var + eps)
    W1 = model.fc1.weight.detach().numpy()
    b1 = model.fc1.bias.detach().numpy()
    W1_folded = W1 * scale_bn[:, None]
    B1_folded = (b1 - mean) * scale_bn + beta
    W2 = model.fc2.weight.detach().numpy()
    B2 = model.fc2.bias.detach().numpy()
    w1_max = np.max(np.abs(W1_folded))
    s1_factor = 31.0 / w1_max
    W1_int = np.clip(np.round(W1_folded * s1_factor), -31, 31).astype(int)
    B1_int = np.clip(np.round(B1_folded * s1_factor), -512, 511).astype(int)
    w2_max = np.max(np.abs(W2))
    s2_factor = 31.0 / w2_max
    W2_int = np.clip(np.round(W2 * s2_factor), -31, 31).astype(int)
    B2_int = np.clip(np.round(B2 * s2_factor), -256, 255).astype(int)
    out_dict = {'architecture': 'hidden_snn_streaming', 'num_inputs': 784, 'num_hidden': 12, 'num_classes': 10, 'weights_l1': W1_int.tolist(), 'biases_l1': B1_int.tolist(), 'weights_l2': W2_int.tolist(), 'biases_l2': B2_int.tolist(), 'threshold': 256, 'leak_shift': 2}
    out_path = Path('weights/digits_mnist_28x28.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_dict, indent=2))
    print(f'Model saved to {out_path}')
if __name__ == '__main__':
    train()
