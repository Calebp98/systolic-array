"""Cocotb testbench for the Processing Element (PE) — output-stationary version.

Tests: MAC (a_in * b_in), accumulation, clear, enable gating, signed values,
       passthrough timing, dot product simulation.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles


async def reset(dut):
    """Reset the PE and wait for it to come out of reset."""
    dut.rst_n.value = 0
    dut.clear_acc.value = 0
    dut.enable.value = 0
    dut.a_in.value = 0
    dut.b_in.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def tick(dut):
    """Advance one clock cycle and settle (read on falling edge)."""
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)


def signed8(val):
    """Convert a Python int to an 8-bit unsigned value for the DUT."""
    return val & 0xFF


def read_signed8(sig):
    """Read an 8-bit signal as a signed Python int."""
    v = int(sig.value) & 0xFF
    return v - 256 if v >= 128 else v


def read_acc(dut):
    """Read the 32-bit accumulator as a signed Python int."""
    v = int(dut.acc_out.value) & 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


@cocotb.test()
async def test_basic_mac(dut):
    """Test 1: Basic MAC (a_in * b_in) and accumulation."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # 3 * 7 = 21
    dut.a_in.value = 3
    dut.b_in.value = 7
    dut.enable.value = 1
    await tick(dut)

    assert read_acc(dut) == 21, f"acc after 3*7: got {read_acc(dut)}, expected 21"
    assert read_signed8(dut.a_out) == 3, f"a_out passthrough: got {read_signed8(dut.a_out)}"
    assert read_signed8(dut.b_out) == 7, f"b_out passthrough: got {read_signed8(dut.b_out)}"

    # 2 * 5 = 10, acc = 21 + 10 = 31
    dut.a_in.value = 2
    dut.b_in.value = 5
    await tick(dut)

    assert read_acc(dut) == 31, f"acc after 3*7 + 2*5: got {read_acc(dut)}, expected 31"


@cocotb.test()
async def test_clear_accumulator(dut):
    """Test 2: Clear accumulator."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # Accumulate something first
    dut.a_in.value = 3
    dut.b_in.value = 7
    dut.enable.value = 1
    await tick(dut)
    assert read_acc(dut) == 21

    # Clear
    dut.enable.value = 0
    dut.clear_acc.value = 1
    await tick(dut)

    assert read_acc(dut) == 0, f"acc after clear: got {read_acc(dut)}, expected 0"
    dut.clear_acc.value = 0


@cocotb.test()
async def test_enable_gating(dut):
    """Test 3: Enable gating — accumulator should not change when disabled."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    dut.a_in.value = 5
    dut.b_in.value = 10
    dut.enable.value = 0
    await tick(dut)

    assert read_acc(dut) == 0, f"acc should be 0 when disabled: got {read_acc(dut)}"


@cocotb.test()
async def test_signed_arithmetic(dut):
    """Test 4: Signed arithmetic."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    # (-4) * (-3) = 12
    dut.a_in.value = signed8(-4)
    dut.b_in.value = signed8(-3)
    dut.enable.value = 1
    await tick(dut)

    assert read_acc(dut) == 12, f"acc for (-4)*(-3): got {read_acc(dut)}, expected 12"

    # + 5 * (-3) = -15, total = 12 + (-15) = -3
    dut.a_in.value = signed8(5)
    dut.b_in.value = signed8(-3)
    await tick(dut)

    assert read_acc(dut) == -3, f"acc for (-4)*(-3) + 5*(-3): got {read_acc(dut)}, expected -3"


@cocotb.test()
async def test_passthrough_timing(dut):
    """Test 5: Data passthrough has 1-cycle delay."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    dut.a_in.value = 42
    dut.b_in.value = 99

    # Before clock edge, outputs should still be 0 (from reset)
    assert read_signed8(dut.a_out) == 0, "a_out should be 0 before tick"

    await tick(dut)

    assert read_signed8(dut.a_out) == 42, f"a_out after 1 tick: got {read_signed8(dut.a_out)}"
    assert read_signed8(dut.b_out) == 99, f"b_out after 1 tick: got {read_signed8(dut.b_out)}"

    # Change inputs and clock again — outputs should show previous inputs
    dut.a_in.value = 10
    dut.b_in.value = 20
    await tick(dut)

    assert read_signed8(dut.a_out) == 10, f"a_out shows prev input: got {read_signed8(dut.a_out)}"
    assert read_signed8(dut.b_out) == 20, f"b_out shows prev input: got {read_signed8(dut.b_out)}"


@cocotb.test()
async def test_dot_product(dut):
    """Test 6: Dot product simulation — C[0][0] = sum_k A[0][k]*B[k][0].

    A row: [1, 2, 3, 4], B col: [5, 6, 7, 8]
    Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    """
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset(dut)

    a_vals = [1, 2, 3, 4]
    b_vals = [5, 6, 7, 8]

    dut.enable.value = 1
    for k in range(4):
        dut.a_in.value = signed8(a_vals[k])
        dut.b_in.value = signed8(b_vals[k])
        await tick(dut)

    dut.enable.value = 0
    assert read_acc(dut) == 70, f"dot product: got {read_acc(dut)}, expected 70"
