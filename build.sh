#!/usr/bin/env bash
set -e
if [ -d "/home/bobbert/oss-cad-suite/bin" ]; then
    export PATH=/home/bobbert/oss-cad-suite/bin:$PATH
fi
MODEL="${1:-weights/digits_mnist_28x28.json}"
python3 compile.py --weights "$MODEL" --build
