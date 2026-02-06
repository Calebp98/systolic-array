VERILATOR = verilator
# Use system ar on macOS to avoid GNU binutils archive incompatibility
export PATH := /usr/bin:$(PATH)

RTL_DIR = rtl
TB_DIR = tb

RTL_PE = $(RTL_DIR)/pe.v
RTL_ALL = $(RTL_DIR)/pe.v $(RTL_DIR)/systolic_array.v

VFLAGS = --cc --exe --trace --build -Wall

.PHONY: all test test-pe test-diag test-full test-cocotb test-cocotb-pe test-cocotb-array sketch clean

all: test

test: test-pe test-diag test-full

# Unit test: single PE
test-pe: $(RTL_PE) $(TB_DIR)/tb_pe.cpp
	@echo "===== Building PE unit test ====="
	$(VERILATOR) $(VFLAGS) --top-module pe \
		$(RTL_PE) $(TB_DIR)/tb_pe.cpp \
		-o sim_pe -Mdir obj_pe
	@echo "===== Running PE unit test ====="
	./obj_pe/sim_pe

# Diagnostic test: systolic array with verbose tracing
test-diag: $(RTL_ALL) $(TB_DIR)/tb_diag.cpp
	@echo "===== Building diagnostic test ====="
	$(VERILATOR) $(VFLAGS) --top-module systolic_array \
		$(RTL_ALL) $(TB_DIR)/tb_diag.cpp \
		-o sim_diag -Mdir obj_diag
	@echo "===== Running diagnostic test ====="
	./obj_diag/sim_diag

# Full integration test (original)
test-full: $(RTL_ALL) $(TB_DIR)/tb_systolic.cpp
	@echo "===== Building full integration test ====="
	$(VERILATOR) $(VFLAGS) --top-module systolic_array \
		$(RTL_ALL) $(TB_DIR)/tb_systolic.cpp \
		-o sim_systolic -Mdir obj_full
	@echo "===== Running full integration test ====="
	./obj_full/sim_systolic

# Cocotb tests (requires: pip install cocotb; and iverilog on PATH)
test-cocotb: test-cocotb-pe test-cocotb-array

test-cocotb-pe:
	@echo "===== Running cocotb PE tests ====="
	$(MAKE) -C $(TB_DIR) -f Makefile.cocotb TEST_TARGET=pe

test-cocotb-array:
	@echo "===== Running cocotb systolic array tests ====="
	$(MAKE) -C $(TB_DIR) -f Makefile.cocotb

# Scratch pad — edit tb/sketch.py, then: make sketch
sketch:
	$(MAKE) -C $(TB_DIR) -f Makefile.cocotb TEST_TARGET=sketch

wave-diag: test-diag
	open -a gtkwave diag.vcd

clean:
	rm -rf obj_pe obj_diag obj_full *.vcd
	rm -rf $(TB_DIR)/sim_build* $(TB_DIR)/results.xml $(TB_DIR)/__pycache__
