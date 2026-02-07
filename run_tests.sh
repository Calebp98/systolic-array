#!/usr/bin/env bash
set -e

PASS=0
FAIL=0
FAILED_TESTS=""

run_test() {
    local name="$1"
    shift
    printf "\n========================================\n"
    printf " %s\n" "$name"
    printf "========================================\n"
    if "$@" 2>&1; then
        PASS=$((PASS + 1))
        printf "\n  -> PASSED\n"
    else
        FAIL=$((FAIL + 1))
        FAILED_TESTS="$FAILED_TESTS\n  - $name"
        printf "\n  -> FAILED\n"
    fi
}

cd "$(dirname "$0")"

printf "Running all systolic array tests...\n"

# Phase 0: Original C++ tests
run_test "C++ PE unit test"          make test-pe
run_test "C++ diagnostic test"       make test-diag
run_test "C++ integration test"      make test-full

# Phase 0: Original cocotb tests
run_test "Cocotb PE tests"           make test-cocotb-pe
run_test "Cocotb array tests"        make test-cocotb-array

# Phase 1: PyTorch quantization
run_test "PyTorch quantization"      .venv/bin/python -m pytest mnist/test_quantize.py -v

# Phase 2: Tiling driver
run_test "Cocotb tiling driver"      make test-cocotb-tiling

# Phase 3: SW MNIST inference
run_test "Cocotb MNIST inference"    make test-cocotb-mnist

# Phase 4-6: RTL modules
run_test "Cocotb ReLU"               make test-cocotb-relu
run_test "Cocotb bias-add"           make test-cocotb-bias
run_test "Cocotb requantize"         make test-cocotb-requantize

# Phase 7: Integrated pipeline
run_test "Cocotb inference pipeline" make test-cocotb-inference

printf "\n========================================\n"
printf " RESULTS: %d passed, %d failed\n" "$PASS" "$FAIL"
printf "========================================\n"

if [ "$FAIL" -gt 0 ]; then
    printf "\nFailed tests:%b\n" "$FAILED_TESTS"
    exit 1
else
    printf "\nAll tests passed.\n"
fi
