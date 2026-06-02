#!/bin/bash
# Build script for Enzyme AD of the FGPA skewer transform.
# Built on https://github.com/pasteurlabs/tesseract-core/blob/9d2f490bfb95720194f8fe51a8d3161c18123a0a/demo/enzyme_thermal_2d/enzyme/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENZYME_LIB="${ENZYME_LIB:-/usr/local/lib/LLVMEnzyme-19.so}"
OUTPUT="${1:-${SCRIPT_DIR}/libfgpa_ad.so}"

echo "=== Step 1: LFortran -> LLVM IR ==="
lfortran --show-llvm --no-array-bounds-checking \
    "${SCRIPT_DIR}/fgpa_skewer.f90" > /tmp/fgpa_skewer.ll

echo "=== Step 1b: Canonicalize math symbols for Enzyme ==="
sed -i \
    -e 's/_lfortran_dexp/exp/g' \
    -e 's/_lfortran_dmax/fmax/g' \
    -e 's/_lfortran_pow_dd/pow/g' \
    /tmp/fgpa_skewer.ll

echo "=== Step 2: Optimize Fortran IR ==="
opt -O1 -S /tmp/fgpa_skewer.ll -o /tmp/fgpa_skewer_opt.ll

echo "=== Step 3: Compile C wrapper -> LLVM IR ==="
clang -emit-llvm -S -O1 "${SCRIPT_DIR}/wrapper.c" -o /tmp/wrapper.ll

echo "=== Step 4: Link IR modules ==="
llvm-link /tmp/wrapper.ll /tmp/fgpa_skewer_opt.ll -S -o /tmp/combined.ll

echo "=== Step 5: Enzyme AD pass ==="
opt --load-pass-plugin="${ENZYME_LIB}" -passes=enzyme \
    -S /tmp/combined.ll -o /tmp/ad.ll

echo "=== Step 6: Optimize post-Enzyme IR ==="
opt -O3 -S /tmp/ad.ll -o /tmp/ad_opt.ll

echo "=== Step 7: Compile shared library ==="
clang -shared -O3 /tmp/ad_opt.ll -o "${OUTPUT}" -lm

echo "=== Built ${OUTPUT} ==="
