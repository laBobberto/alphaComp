#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

if [ -d "/home/bobbert/oss-cad-suite/bin" ]; then
    export PATH=/home/bobbert/oss-cad-suite/bin:$PATH
fi

mkdir -p build

yosys -p "read_verilog -sv top_138k.v uart_rx.v uart_tx.v; synth_gowin -family gw5a -top top -json build/top_138k.json"

nextpnr-himbaechel \
    --json build/top_138k.json \
    --write build/top_138k_pnr.json \
    --device GW5AST-LV138PG484AC1/I0 \
    --vopt cst=tang_mega_138k.cst \
    --timing-allow-fail

gowin_pack -d GW5AST-138C -o build/top_138k.fs build/top_138k_pnr.json
