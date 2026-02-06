// Unit test for a single Processing Element (PE) — output-stationary version
// Tests: MAC (a_in * b_in), accumulation, clear, signed values, passthrough
#include <cstdio>
#include <cstdint>
#include "Vpe.h"
#include "verilated.h"

static Vpe *dut;
static int test_pass = 0, test_fail = 0;

void tick() {
    dut->clk = 0; dut->eval();
    dut->clk = 1; dut->eval();
}

void reset() {
    dut->rst_n = 0;
    dut->clear_acc = 0; dut->enable = 0;
    dut->a_in = 0; dut->b_in = 0;
    tick(); tick();
    dut->rst_n = 1;
    tick();
}

void check(const char *name, int32_t got, int32_t expected) {
    if (got == expected) {
        printf("  PASS: %s = %d\n", name, got);
        test_pass++;
    } else {
        printf("  FAIL: %s = %d, expected %d\n", name, got, expected);
        test_fail++;
    }
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vpe;

    // ============================================================
    printf("=== PE Test 1: Basic MAC (a_in * b_in) ===\n");
    reset();
    dut->a_in = 3;
    dut->b_in = 7;
    dut->enable = 1;
    tick();
    check("acc after 3*7", (int32_t)dut->acc_out, 21);
    check("a_out passthrough", (int8_t)dut->a_out, 3);
    check("b_out passthrough", (int8_t)dut->b_out, 7);

    // Another MAC: 2*5 = 10, acc = 21+10 = 31
    dut->a_in = 2;
    dut->b_in = 5;
    tick();
    check("acc after 3*7 + 2*5", (int32_t)dut->acc_out, 31);
    dut->enable = 0;

    // ============================================================
    printf("\n=== PE Test 2: Clear accumulator ===\n");
    dut->clear_acc = 1;
    tick();
    check("acc after clear", (int32_t)dut->acc_out, 0);
    dut->clear_acc = 0;

    // ============================================================
    printf("\n=== PE Test 3: Enable gating ===\n");
    dut->a_in = 5;
    dut->b_in = 10;
    dut->enable = 0;
    tick();
    check("acc unchanged when disabled", (int32_t)dut->acc_out, 0);

    // ============================================================
    printf("\n=== PE Test 4: Signed arithmetic ===\n");
    reset();
    dut->a_in = (uint8_t)(int8_t)-4;
    dut->b_in = (uint8_t)(int8_t)-3;
    dut->enable = 1;
    tick();
    check("acc for (-4)*(-3)", (int32_t)dut->acc_out, 12);

    dut->a_in = (uint8_t)(int8_t)5;
    dut->b_in = (uint8_t)(int8_t)-3;
    tick();
    check("acc for (-4)*(-3) + 5*(-3)", (int32_t)dut->acc_out, -3);

    // ============================================================
    printf("\n=== PE Test 5: Data passthrough timing ===\n");
    reset();
    dut->a_in = 42;
    dut->b_in = 99;
    dut->eval();
    check("a_out before tick (should be 0)", (int8_t)dut->a_out, 0);

    tick();
    check("a_out after 1 tick", (int8_t)dut->a_out, 42);
    check("b_out after 1 tick", (int8_t)dut->b_out, 99);

    dut->a_in = 10;
    dut->b_in = 20;
    tick();
    check("a_out shows prev input (10)", (int8_t)dut->a_out, 10);
    check("b_out shows prev input (20)", (int8_t)dut->b_out, 20);

    // ============================================================
    printf("\n=== PE Test 6: Dot product simulation ===\n");
    // Simulate PE(0,0) computing C[0][0] = sum_k A[0][k]*B[k][0]
    // A row: [1, 2, 3, 4], B col: [5, 6, 7, 8]
    // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
    reset();
    int8_t a_vals[] = {1, 2, 3, 4};
    int8_t b_vals[] = {5, 6, 7, 8};
    dut->enable = 1;
    for (int k = 0; k < 4; k++) {
        dut->a_in = (uint8_t)a_vals[k];
        dut->b_in = (uint8_t)b_vals[k];
        tick();
    }
    dut->enable = 0;
    check("dot product [1,2,3,4].[5,6,7,8]", (int32_t)dut->acc_out, 70);

    // ============================================================
    printf("\n========================================\n");
    printf("PE Tests: %d passed, %d failed\n", test_pass, test_fail);
    printf("========================================\n");

    delete dut;
    return test_fail > 0 ? 1 : 0;
}
