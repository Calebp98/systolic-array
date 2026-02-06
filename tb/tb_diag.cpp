// Diagnostic testbench for output-stationary systolic array
// Tests are ordered from simplest to most complex to isolate failures.
#include <cstdio>
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
static int test_pass = 0, test_fail = 0;
static int cycle_num = 0;

void tick() {
    dut->clk = 0; dut->eval();
    if (tfp) tfp->dump(sim_time++);
    dut->clk = 1; dut->eval();
    if (tfp) tfp->dump(sim_time++);
    cycle_num++;
}

void reset() {
    dut->rst_n = 0;
    dut->start = 0;
    for (int w = 0; w < 4; w++) { dut->a_data[w] = 0; dut->b_data[w] = 0; }
    tick(); tick(); tick();
    dut->rst_n = 1;
    tick();
    cycle_num = 0;
}

const char* state_name(int s) {
    switch(s) {
        case 0: return "IDLE";
        case 1: return "COMPUTE";
        case 2: return "DONE";
        default: return "???";
    }
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

void print_result_matrix() {
    for (int i = 0; i < N; i++) {
        printf("    [");
        for (int j = 0; j < N; j++) printf("%6d", get_result(i, j));
        printf(" ]\n");
    }
}

void check(const char *name, int32_t got, int32_t expected) {
    if (got == expected) { test_pass++; }
    else { printf("  FAIL: %s = %d, expected %d\n", name, got, expected); test_fail++; }
}

// Run a full multiply: set matrices, start, wait for done, return to idle
void run_multiply(int8_t A[N][N], int8_t B[N][N]) {
    set_matrix((uint32_t*)dut->a_data, A);
    set_matrix((uint32_t*)dut->b_data, B);
    dut->start = 1;
    tick();
    dut->start = 0;
    // Wait for completion
    int timeout = 30;
    while (dut->state_out != 0 && timeout-- > 0) tick();
}

void check_results(int8_t A[N][N], int8_t B[N][N], const char *test_name) {
    int32_t expected[N][N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            expected[i][j] = 0;
            for (int k = 0; k < N; k++)
                expected[i][j] += (int32_t)A[i][k] * (int32_t)B[k][j];
        }

    int local_fail = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            char buf[64];
            snprintf(buf, sizeof(buf), "C[%d][%d]", i, j);
            int32_t got = get_result(i, j);
            if (got != expected[i][j]) {
                printf("  FAIL: %s = %d, expected %d\n", buf, got, expected[i][j]);
                test_fail++; local_fail++;
            } else {
                test_pass++;
            }
        }

    if (local_fail > 0) {
        printf("  Got:\n");
        print_result_matrix();
        printf("  Expected:\n");
        for (int i = 0; i < N; i++) {
            printf("    [");
            for (int j = 0; j < N; j++) printf("%6d", expected[i][j]);
            printf(" ]\n");
        }
    } else {
        printf("  All 16 elements correct!\n");
    }
}

// ============================================================
void test_state_machine() {
    printf("=== Test 1: State machine transitions ===\n");
    reset();
    check("initial state IDLE", dut->state_out, 0);

    dut->start = 1; tick(); dut->start = 0;
    check("after start -> COMPUTE", dut->state_out, 1);

    int compute_cycles = 0;
    while (dut->state_out == 1) { compute_cycles++; tick(); }
    printf("  Compute lasted %d cycles (expected %d)\n", compute_cycles, 3*(N-1)+1);
    check("compute cycle count", compute_cycles, 3*(N-1)+1);

    check("reached DONE", dut->state_out, 2);
    tick();
    check("back to IDLE", dut->state_out, 0);
    printf("\n");
}

// ============================================================
void test_single_element() {
    printf("=== Test 2: Single non-zero element ===\n");
    printf("  A[0][0]=1, B[0][0]=7, everything else 0\n");
    printf("  Expected: C[0][0]=7, rest=0\n");
    reset();
    int8_t A[N][N] = {{1,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
    int8_t B[N][N] = {{7,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
    run_multiply(A, B);
    check_results(A, B, "single element");
    printf("\n");
}

// ============================================================
void test_row_col() {
    printf("=== Test 3: Row vector * Column vector ===\n");
    printf("  A=[1,2,3,4; 0...], B=[[1;1;1;1], 0...]\n");
    printf("  Expected: C[0][0]=10, rest=0\n");
    reset();
    int8_t A[N][N] = {{1,2,3,4},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
    int8_t B[N][N] = {{1,0,0,0},{1,0,0,0},{1,0,0,0},{1,0,0,0}};
    run_multiply(A, B);
    check_results(A, B, "row*col");
    printf("\n");
}

// ============================================================
void test_2x2_verbose() {
    printf("=== Test 4: 2x2 subproblem (verbose) ===\n");
    printf("  A=[[1,2],[3,4]], B=[[5,6],[7,8]]\n");
    printf("  Expected: C=[[19,22],[43,50]]\n");
    reset();
    int8_t A[N][N] = {{1,2,0,0},{3,4,0,0},{0,0,0,0},{0,0,0,0}};
    int8_t B[N][N] = {{5,6,0,0},{7,8,0,0},{0,0,0,0},{0,0,0,0}};
    set_matrix((uint32_t*)dut->a_data, A);
    set_matrix((uint32_t*)dut->b_data, B);

    dut->start = 1; tick(); dut->start = 0;

    printf("  --- COMPUTE ---\n");
    while (dut->state_out == 1) {
        printf("  cycle %2d: C[0][0]=%d C[0][1]=%d C[1][0]=%d C[1][1]=%d\n",
            cycle_num, get_result(0,0), get_result(0,1), get_result(1,0), get_result(1,1));
        tick();
    }
    if (dut->state_out == 2) tick(); // DONE -> IDLE

    check_results(A, B, "2x2");
    printf("\n");
}

// ============================================================
void test_diagonal() {
    printf("=== Test 5: All-ones * Diagonal ===\n");
    printf("  A=ones(4), B=diag(1,2,3,4)\n");
    printf("  Expected: each row = [1,2,3,4]\n");
    reset();
    int8_t A[N][N] = {{1,1,1,1},{1,1,1,1},{1,1,1,1},{1,1,1,1}};
    int8_t B[N][N] = {{1,0,0,0},{0,2,0,0},{0,0,3,0},{0,0,0,4}};
    run_multiply(A, B);
    check_results(A, B, "diagonal");
    printf("\n");
}

// ============================================================
void test_identity() {
    printf("=== Test 6: Identity * B = B ===\n");
    reset();
    int8_t A[N][N] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    int8_t B[N][N] = {{5,6,7,8},{1,2,3,4},{9,10,11,12},{13,14,15,16}};
    run_multiply(A, B);
    check_results(A, B, "identity");
    printf("\n");
}

// ============================================================
void test_general() {
    printf("=== Test 7: General 4x4 multiplication ===\n");
    reset();
    int8_t A[N][N] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
    int8_t B[N][N] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
    run_multiply(A, B);
    check_results(A, B, "general");
    printf("\n");
}

// ============================================================
void test_signed() {
    printf("=== Test 8: Signed multiplication ===\n");
    reset();
    int8_t A[N][N] = {{1,-2,3,-4},{-5,6,-7,8},{9,-10,11,-12},{-1,2,-3,4}};
    int8_t B[N][N] = {{2,-1,0,3},{-4,5,-6,7},{8,-9,10,-11},{-3,4,-5,6}};
    run_multiply(A, B);
    check_results(A, B, "signed");
    printf("\n");
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);
    dut = new Vsystolic_array;
    tfp = new VerilatedVcdC;
    dut->trace(tfp, 99);
    tfp->open("diag.vcd");

    test_state_machine();
    test_single_element();
    test_row_col();
    test_2x2_verbose();
    test_diagonal();
    test_identity();
    test_general();
    test_signed();

    tfp->close();
    delete tfp;
    delete dut;

    printf("========================================\n");
    printf("Diagnostic Tests: %d passed, %d failed\n", test_pass, test_fail);
    printf("========================================\n");
    return test_fail > 0 ? 1 : 0;
}
