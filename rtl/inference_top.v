// Integrated inference pipeline: systolic_array → bias_add → relu → requantize
// Processes one 4x4 tile through the full pipeline per invocation.
//
// FSM: IDLE → MATMUL (run systolic array) → POST (bias+relu+requant, 1 cycle) → DONE
//
// Interface:
//   - Load a_data, b_data, bias_data, shift_amount, then pulse start
//   - enable_bias, enable_relu control which post-processing steps run
//   - When done pulses, result_acc has raw int32 accumulator output,
//     and result_quant has the post-processed int8 output

module inference_top #(
    parameter N          = 4,
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
) (
    input  wire                          clk,
    input  wire                          rst_n,

    // Control
    input  wire                          start,
    output reg                           done,

    // Post-processing control
    input  wire                          enable_bias,
    input  wire                          enable_relu,
    input  wire                          enable_requant,
    input  wire [4:0]                    shift_amount,

    // Matrix inputs (same as systolic_array)
    input  wire signed [N*N*DATA_WIDTH-1:0] a_data,
    input  wire signed [N*N*DATA_WIDTH-1:0] b_data,

    // Bias input: one value per output column (4 values, each ACC_WIDTH bits)
    // Applied to each row of the result identically
    input  wire signed [N*ACC_WIDTH-1:0] bias_data,

    // Outputs
    output wire signed [N*N*ACC_WIDTH-1:0]  result_acc,    // Raw accumulator (int32)
    output wire signed [N*N*ACC_WIDTH-1:0]  result_post,   // Post-processed int32 (after bias+relu)
    output wire [N*N*DATA_WIDTH-1:0]        result_quant,  // Post-processed (int8)

    // Debug
    output wire [3:0]                    state_out
);

    // FSM states
    localparam IDLE   = 4'd0;
    localparam MATMUL = 4'd1;
    localparam POST   = 4'd2;
    localparam DONE   = 4'd3;

    reg [3:0] state;
    assign state_out = state;

    // Systolic array signals
    wire sa_done;
    wire [3:0] sa_state;
    reg  sa_start;
    wire signed [N*N*ACC_WIDTH-1:0] sa_result;

    systolic_array #(
        .N(N),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_sa (
        .clk(clk),
        .rst_n(rst_n),
        .start(sa_start),
        .done(sa_done),
        .state_out(sa_state),
        .a_data(a_data),
        .b_data(b_data),
        .result(sa_result)
    );

    assign result_acc = sa_result;

    // Post-processing chain (combinational)
    // Step 1: Bias addition — add bias[j] to each element in column j
    wire signed [N*N*ACC_WIDTH-1:0] after_bias;

    genvar gi, gj;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : gen_bias_row
            for (gj = 0; gj < N; gj = gj + 1) begin : gen_bias_col
                wire signed [ACC_WIDTH-1:0] sa_val = sa_result[(gi*N+gj)*ACC_WIDTH +: ACC_WIDTH];
                wire signed [ACC_WIDTH-1:0] b_val  = bias_data[gj*ACC_WIDTH +: ACC_WIDTH];
                assign after_bias[(gi*N+gj)*ACC_WIDTH +: ACC_WIDTH] =
                    enable_bias ? (sa_val + b_val) : sa_val;
            end
        end
    endgenerate

    // Step 2: ReLU
    wire [N*N*ACC_WIDTH-1:0] after_relu;

    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : gen_relu_row
            for (gj = 0; gj < N; gj = gj + 1) begin : gen_relu_col
                wire signed [ACC_WIDTH-1:0] val = after_bias[(gi*N+gj)*ACC_WIDTH +: ACC_WIDTH];
                assign after_relu[(gi*N+gj)*ACC_WIDTH +: ACC_WIDTH] =
                    (enable_relu && val[ACC_WIDTH-1]) ? {ACC_WIDTH{1'b0}} : val;
            end
        end
    endgenerate

    assign result_post = after_relu;

    // Step 3: Requantize (int32 → int8)
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : gen_req_row
            for (gj = 0; gj < N; gj = gj + 1) begin : gen_req_col
                wire signed [ACC_WIDTH-1:0] val = after_relu[(gi*N+gj)*ACC_WIDTH +: ACC_WIDTH];
                wire signed [ACC_WIDTH-1:0] shifted = val >>> shift_amount;
                wire overflow_pos = !shifted[ACC_WIDTH-1] && (shifted > 127);
                wire overflow_neg = shifted[ACC_WIDTH-1] && (shifted < -128);
                wire signed [DATA_WIDTH-1:0] clamped =
                    enable_requant ? (
                        overflow_pos ? 8'sd127 :
                        overflow_neg ? -8'sd128 :
                        shifted[DATA_WIDTH-1:0]
                    ) : val[DATA_WIDTH-1:0];
                assign result_quant[(gi*N+gj)*DATA_WIDTH +: DATA_WIDTH] = clamped;
            end
        end
    endgenerate

    // FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= IDLE;
            done     <= 0;
            sa_start <= 0;
        end else begin
            done     <= 0;
            sa_start <= 0;

            case (state)
                IDLE: begin
                    if (start) begin
                        sa_start <= 1;
                        state    <= MATMUL;
                    end
                end

                MATMUL: begin
                    // Wait for systolic array to finish (returns to IDLE)
                    if (sa_done) begin
                        state <= POST;
                    end
                end

                POST: begin
                    // Post-processing is combinational, result is ready
                    done  <= 1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
