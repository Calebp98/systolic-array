// 4x4 Output-Stationary Systolic Array for Matrix Multiplication
// Computes C = A * B where A, B are 4x4 matrices of 8-bit signed integers
//
// Architecture: Output-stationary — PE(i,j) accumulates C[i][j].
//   - A values flow left-to-right (horizontal), row i gets A[i][*]
//   - B values flow top-to-bottom (vertical), col j gets B[*][j]
//   - Both are skewed so A[i][k] and B[k][j] meet at PE(i,j) simultaneously
//
// Skewing:
//   - A[i][k] fed at left edge at cycle (i + k)
//   - B[k][j] fed at top edge at cycle (k + j)
//   - After propagation: A[i][k] arrives at PE(i,j) at cycle i+k+j
//                        B[k][j] arrives at PE(i,j) at cycle k+j+i
//   - These are equal! So the right pairs always meet.
//
// Compute duration: last product at PE(N-1,N-1) for k=N-1 arrives at
//   cycle 3*(N-1). So we need 3*(N-1)+1 = 3N-2 compute cycles.
//
// Operation phases:
//   1. IDLE:    Waiting for start signal
//   2. COMPUTE: Stream A and B through array (3N-2 cycles)
//   3. DONE:    Results available in PE accumulators

module systolic_array #(
    parameter N          = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Control interface
    input  wire                    start,
    output reg                     done,
    output reg  [3:0]              state_out,  // For debug

    // Weight matrix B input (row-major, all provided at start)
    // b_data[(k*N + j)*DATA_WIDTH +: DATA_WIDTH] = B[k][j]
    input  wire signed [N*N*DATA_WIDTH-1:0] b_data,

    // Activation matrix A input (row-major, all provided at start)
    // a_data[(i*N + k)*DATA_WIDTH +: DATA_WIDTH] = A[i][k]
    input  wire signed [N*N*DATA_WIDTH-1:0] a_data,

    // Result matrix C output (row-major)
    // result[(i*N + j)*ACC_WIDTH +: ACC_WIDTH] = C[i][j]
    output wire signed [N*N*ACC_WIDTH-1:0] result
);

    // State machine
    localparam IDLE    = 4'd0;
    localparam COMPUTE = 4'd1;
    localparam DONE    = 4'd2;

    reg [3:0] state;
    reg [4:0] cycle_cnt;  // needs to count up to 3*(N-1) = 9

    localparam COMPUTE_CYCLES = 3 * (N - 1);  // last cycle index

    // PE interconnect wires
    wire signed [DATA_WIDTH-1:0] a_wire [0:N-1][0:N]; // horizontal: col 0..N
    wire signed [DATA_WIDTH-1:0] b_wire [0:N][0:N-1]; // vertical: row 0..N
    wire signed [ACC_WIDTH-1:0]  acc    [0:N-1][0:N-1];

    // Control signals (combinational)
    wire clear_acc_sig = (state == IDLE) && start;
    wire enable_sig    = (state == COMPUTE);

    // Activation input with row skewing
    // Row i: A[i][k] fed at cycle (i + k), for k = 0..N-1
    genvar gi, gj;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : gen_a_left
            reg signed [DATA_WIDTH-1:0] a_feed;
            wire [4:0] k_idx = cycle_cnt - gi[4:0];
            /* verilator lint_off UNSIGNED */
            wire a_valid = (state == COMPUTE) && (cycle_cnt >= gi[4:0]) && (k_idx < N[4:0]);
            /* verilator lint_on UNSIGNED */

            always @(*) begin
                if (a_valid)
                    a_feed = a_data[(gi * N + k_idx) * DATA_WIDTH +: DATA_WIDTH];
                else
                    a_feed = {DATA_WIDTH{1'b0}};
            end

            assign a_wire[gi][0] = a_feed;
        end
    endgenerate

    // Weight input with column skewing
    // Col j: B[k][j] fed at cycle (k + j), for k = 0..N-1
    generate
        for (gj = 0; gj < N; gj = gj + 1) begin : gen_b_top
            reg signed [DATA_WIDTH-1:0] b_feed;
            wire [4:0] k_idx = cycle_cnt - gj[4:0];
            /* verilator lint_off UNSIGNED */
            wire b_valid = (state == COMPUTE) && (cycle_cnt >= gj[4:0]) && (k_idx < N[4:0]);
            /* verilator lint_on UNSIGNED */

            always @(*) begin
                if (b_valid)
                    b_feed = b_data[(k_idx * N + gj) * DATA_WIDTH +: DATA_WIDTH];
                else
                    b_feed = {DATA_WIDTH{1'b0}};
            end

            assign b_wire[0][gj] = b_feed;
        end
    endgenerate

    // Instantiate NxN PE grid
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : gen_row
            for (gj = 0; gj < N; gj = gj + 1) begin : gen_col
                pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH(ACC_WIDTH)
                ) u_pe (
                    .clk      (clk),
                    .rst_n    (rst_n),
                    .clear_acc(clear_acc_sig),
                    .enable   (enable_sig),
                    .a_in     (a_wire[gi][gj]),
                    .a_out    (a_wire[gi][gj+1]),
                    .b_in     (b_wire[gi][gj]),
                    .b_out    (b_wire[gi+1][gj]),
                    .acc_out  (acc[gi][gj])
                );
            end
        end
    endgenerate

    // Map accumulator outputs to flat result bus
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : gen_res_row
            for (gj = 0; gj < N; gj = gj + 1) begin : gen_res_col
                assign result[(gi*N + gj)*ACC_WIDTH +: ACC_WIDTH] = acc[gi][gj];
            end
        end
    endgenerate

    // State machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= IDLE;
            cycle_cnt <= 0;
            done      <= 0;
        end else begin
            done <= 0;

            case (state)
                IDLE: begin
                    if (start) begin
                        state     <= COMPUTE;
                        cycle_cnt <= 0;
                    end
                end

                COMPUTE: begin
                    if (cycle_cnt == COMPUTE_CYCLES) begin
                        state <= DONE;
                    end else begin
                        cycle_cnt <= cycle_cnt + 1;
                    end
                end

                DONE: begin
                    done  <= 1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

    always @(*) state_out = state;

endmodule
