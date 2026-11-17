#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

if [ -d "/home/bobbert/oss-cad-suite/bin" ]; then
    export PATH=/home/bobbert/oss-cad-suite/bin:$PATH
fi

BITSTREAM="build/top_138k.fs"

if [ ! -f "$BITSTREAM" ]; then
    echo "Bitstream not found. Run ./build.sh first."
    exit 1
fi

if [ "$1" == "-f" ] || [ "$1" == "--flash" ]; then
    openFPGALoader -b tangmega138k -f "$BITSTREAM"
else
    openFPGALoader -b tangmega138k "$BITSTREAM"
fi
