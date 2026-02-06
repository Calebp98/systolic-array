// Integration test for output-stationary systolic array
// Runs 3 full matrix multiplications and verifies against software reference.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "Vsystolic_array.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

#define N 4
#define DATA_WIDTH 8
#define ACC_WIDTH 32

static Vsystolic_array *dut;
static VerilatedVcdC *tfp;
static vluint64_t sim_time = 0;

void tick() {
    dut->clk = 0; dut->eval();
    if (tfp) tfp->dump(sim_time++);
    dut->clk = 1; dut->eval();
    if (tfp) tfp->dump(sim_time++);
}

void reset() {
    dut->rst_n = 0;
    dut->start = 0;
    for (int w = 0; w < 4; w++) { dut->a_data[w] = 0; dut->b_data[w] = 0; }
    for (int i = 0; i < 3; i++) tick();
    dut->rst_n = 1;
    tick();
}

void set_matrix(uint32_t *dest, int8_t mat[N][N]) {
    for (int w = 0; w < 4; w++) dest[w] = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int bit = (i * N + j) * DATA_WIDTH;
            dest[bit/32] |= ((uint32_t)(uint8_t)mat[i][j]) << (bit % 32);
        }
}

int32_t get_result(int i, int j) {
    return (int32_t)dut->result[i * N + j];
}

int run_test(const char *name, int8_t A[N][N], int8_t B[N][N]) {
    printf("=== %s ===\n", name);
    reset();
    set_matrix((uint32_t*)dut->a_data, A);
    set_matrix((uint32_t*)dut->b_data, B);

    dut->start = 1; tick(); dut->start = 0;
    int timeout = 30;
    while (dut->state_out != 0 && timeout-- > 0) tick();

    int32_t expected[N][N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            expected[i][j] = 0;
            for (int k = 0; k < N; k++)
                expected[i][j] += (int32_t)A[i][k] * (int32_t)B[k][j];
        }

    int pass = 1;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int32_t got = get_result(i, j);
            if (got != expected[i][j]) {
                printf("  FAIL C[%d][%d]: got %d, expected %d\n", i, j, got, expected[i][j]);
                pass = 0;
            } else {
                printf("  C[%d][%d] = %d OK\n", i, j, got);
            }
        }
    printf("%s: %s\n\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    dut = new Vsystolic_array;
    tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("systolic_array.vcd");

    int all_pass = 1;

    // Test 1: Identity
    {
        int8_t A[N][N] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
        int8_t B[N][N] = {{5,6,7,8},{1,2,3,4},{9,10,11,12},{13,14,15,16}};
        all_pass &= run_test("Test 1: Identity * B = B", A, B);
    }

    // Test 2: General
    {
        int8_t A[N][N] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
        int8_t B[N][N] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
        all_pass &= run_test("Test 2: General 4x4", A, B);
    }

    // Test 3: Signed
    {
        int8_t A[N][N] = {{1,-2,3,-4},{-5,6,-7,8},{9,-10,11,-12},{-1,2,-3,4}};
        int8_t B[N][N] = {{2,-1,0,3},{-4,5,-6,7},{8,-9,10,-11},{-3,4,-5,6}};
        all_pass &= run_test("Test 3: Signed values", A, B);
    }

    tfp->close();
    delete tfp;
    delete dut;

    printf("Simulation complete. Waveform saved to systolic_array.vcd\n");
    return all_pass ? 0 : 1;
}
