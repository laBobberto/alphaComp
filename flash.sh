#!/usr/bin/env bash
set -e
if [ -d "/home/bobbert/oss-cad-suite/bin" ]; then
    export PATH=/home/bobbert/oss-cad-suite/bin:$PATH
fi
BITSTREAM="build/top.fs"
if [ ! -f "$BITSTREAM" ]; then
    echo "Error: Bitstream $BITSTREAM not found. Run ./build.sh first."
    exit 1
fi
if [ "$1" == "-f" ] || [ "$1" == "--flash" ]; then
    openFPGALoader -b tangnano1k -f "$BITSTREAM"
else
    openFPGALoader -b tangnano1k "$BITSTREAM"
fi
