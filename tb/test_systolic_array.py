"""Cocotb testbench for the 4x4 output-stationary systolic array.

Tests state machine transitions, single element, 2x2 subproblem, identity,
general 4x4, signed multiplication, row*col, and ones*diagonal.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles

N = 4
DATA_WIDTH = 8
ACC_WIDTH = 32


def matmul(A, B):
    """Software reference 4x4 matrix multiply."""
    C = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C


def pack_matrix(mat):
    """Pack a 4x4 matrix of int8 values into an integer for the DUT port.

    Bit layout: mat[i][j] occupies bits [(i*N+j)*8 +: 8].
    """
    val = 0
    for i in range(N):
        for j in range(N):
            byte = mat[i][j] & 0xFF
            val |= byte << ((i * N + j) * DATA_WIDTH)
    return val


def unpack_result(dut):
    """Read the 4x4 result matrix from the DUT as signed 32-bit integers."""
    raw = int(dut.result.value)
    C = [[0] * N for _ in range(N)]
    mask32 = (1 << ACC_WIDTH) - 1
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            v = (raw >> (idx * ACC_WIDTH)) & mask32
            if v >= (1 << (ACC_WIDTH - 1)):
                v -= 1 << ACC_WIDTH
            C[i][j] = v
    return C


def read_state(dut):
    return int(dut.state_out.value)


async def tick(dut):
    """Advance one clock cycle and settle."""
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)


async def reset(dut):
    """Reset the systolic array."""
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.a_data.value = 0
    dut.b_data.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await tick(dut)


async def run_multiply(dut, A, B):
    """Load matrices, start computation, wait for done, return to IDLE."""
    dut.a_data.value = pack_matrix(A)
    dut.b_data.value = pack_matrix(B)

    dut.start.value = 1
    await tick(dut)
    dut.start.value = 0

    # Wait for back to IDLE (state == 0)
    for _ in range(30):
        await tick(dut)
        if read_state(dut) == 0:
            break
    else:
        assert False, "Timed out waiting for multiply to complete"


def check_results(got, expected, test_name):
    """Compare result matrix against expected, with detailed failure messages."""
    for i in range(N):
        for j in range(N):
            assert got[i][j] == expected[i][j], (
                f"{test_name}: C[{i}][{j}] = {got[i][j]}, expected {expected[i][j]}"
            )


@cocotb.test()
async def test_state_machine(dut):
    """Test 1: State machine transitions and cycle count."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Should be in IDLE (0)
    assert read_state(dut) == 0, "Initial state should be IDLE"

    # Start
    dut.start.value = 1
    await tick(dut)
    dut.start.value = 0

    assert read_state(dut) == 1, "After start should be COMPUTE"

    # Count compute cycles
    compute_cycles = 0
    while read_state(dut) == 1:
        compute_cycles += 1
        await tick(dut)

    expected_cycles = 3 * (N - 1) + 1  # = 10
    assert compute_cycles == expected_cycles, (
        f"Compute lasted {compute_cycles} cycles, expected {expected_cycles}"
    )

    assert read_state(dut) == 2, "Should reach DONE"
    await tick(dut)
    assert read_state(dut) == 0, "Should return to IDLE"


@cocotb.test()
async def test_single_element(dut):
    """Test 2: Single non-zero element — A[0][0]=1, B[0][0]=7."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    A = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    B = [[7, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    await run_multiply(dut, A, B)
    got = unpack_result(dut)
    check_results(got, matmul(A, B), "single element")


@cocotb.test()
async def test_2x2_subproblem(dut):
    """Test 3: 2x2 subproblem — [[1,2],[3,4]] * [[5,6],[7,8]]."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    A = [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    B = [[5, 6, 0, 0], [7, 8, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

    await run_multiply(dut, A, B)
    got = unpack_result(dut)
    expected = matmul(A, B)
    check_results(got, expected, "2x2 subproblem")

    # Verify the 2x2 corner explicitly
    assert got[0][0] == 19
    assert got[0][1] == 22
    assert got[1][0] == 43
    assert got[1][1] == 50


@cocotb.test()
async def test_identity(dut):
    """Test 4: Identity * B = B."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    A = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    B = [[5, 6, 7, 8], [1, 2, 3, 4], [9, 10, 11, 12], [13, 14, 15, 16]]

    await run_multiply(dut, A, B)
    got = unpack_result(dut)
    check_results(got, B, "identity * B")


@cocotb.test()
async def test_general_4x4(dut):
    """Test 5: General 4x4 multiplication."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    B = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]

    await run_multiply(dut, A, B)
    got = unpack_result(dut)
    check_results(got, matmul(A, B), "general 4x4")


@cocotb.test()
async def test_signed(dut):
    """Test 6: Signed multiplication."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    A = [[1, -2, 3, -4], [-5, 6, -7, 8], [9, -10, 11, -12], [-1, 2, -3, 4]]
    B = [[2, -1, 0, 3], [-4, 5, -6, 7], [8, -9, 10, -11], [-3, 4, -5, 6]]

    await run_multiply(dut, A, B)
    got = unpack_result(dut)
    check_results(got, matmul(A, B), "signed")


@cocotb.test()
async def test_row_times_column(dut):
    """Test 7: Row vector * Column vector."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    A = [[1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    B = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]

    await run_multiply(dut, A, B)
    got = unpack_result(dut)
    expected = matmul(A, B)
    check_results(got, expected, "row * col")
    assert got[0][0] == 10


@cocotb.test()
async def test_ones_times_diagonal(dut):
    """Test 8: All-ones * Diagonal — each row should be [1,2,3,4]."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    A = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    B = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]

    await run_multiply(dut, A, B)
    got = unpack_result(dut)
    check_results(got, matmul(A, B), "ones * diagonal")
