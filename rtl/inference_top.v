// Integrated inference pipeline: systolic_array → bias_add → relu → requantize
// Processes one 4x4 tile through the full pipeline per invocation.
//
// FSM: IDLE → MATMUL (run systolic array; done when sa_done fires) → IDLE
// Post-processing is purely combinational, so results are valid immediately.
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

    // Step 1: Bias addition — one bias_add per row, broadcasting bias[j] across columns
    // bias_add expects matched-width element pairs, so we expand bias_data to N*N elements
    wire signed [N*N*ACC_WIDTH-1:0] bias_expanded;
    wire signed [N*N*ACC_WIDTH-1:0] bias_add_out;
    wire signed [N*N*ACC_WIDTH-1:0] after_bias;

    genvar gi, gj;
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : gen_bias_expand
            for (gj = 0; gj < N; gj = gj + 1) begin : gen_bias_col
                assign bias_expanded[(gi*N+gj)*ACC_WIDTH +: ACC_WIDTH] =
                    bias_data[gj*ACC_WIDTH +: ACC_WIDTH];
            end
        end
    endgenerate

    bias_add #(
        .DATA_WIDTH(ACC_WIDTH),
        .N_ELEM(N*N)
    ) u_bias_add (
        .data_in(sa_result),
        .bias(bias_expanded),
        .data_out(bias_add_out)
    );

    // Enable mux: select between biased and raw result
    assign after_bias = enable_bias ? bias_add_out : sa_result;

    // Step 2: ReLU
    wire [N*N*ACC_WIDTH-1:0] relu_out;
    wire [N*N*ACC_WIDTH-1:0] after_relu;

    relu #(
        .DATA_WIDTH(ACC_WIDTH),
        .N_ELEM(N*N)
    ) u_relu (
        .data_in(after_bias),
        .data_out(relu_out)
    );

    // Enable mux: select between ReLU'd and bypass
    assign after_relu = enable_relu ? relu_out : after_bias;

    assign result_post = after_relu;

    // Step 3: Requantize (int32 → int8)
    wire [N*N*DATA_WIDTH-1:0] requant_out;
    wire [N*N*DATA_WIDTH-1:0] after_requant;

    requantize #(
        .IN_WIDTH(ACC_WIDTH),
        .OUT_WIDTH(DATA_WIDTH),
        .N_ELEM(N*N)
    ) u_requantize (
        .data_in(after_relu),
        .shift_amount(shift_amount),
        .data_out(requant_out)
    );

    // Enable mux: select between requantized and raw truncation
    generate
        for (gi = 0; gi < N; gi = gi + 1) begin : gen_req_bypass_row
            for (gj = 0; gj < N; gj = gj + 1) begin : gen_req_bypass_col
                wire signed [ACC_WIDTH-1:0] val = after_relu[(gi*N+gj)*ACC_WIDTH +: ACC_WIDTH];
                assign after_requant[(gi*N+gj)*DATA_WIDTH +: DATA_WIDTH] =
                    enable_requant ? requant_out[(gi*N+gj)*DATA_WIDTH +: DATA_WIDTH]
                                   : val[DATA_WIDTH-1:0];
            end
        end
    endgenerate

    assign result_quant = after_requant;

    // FSM — no POST state; post-processing is combinational
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
                    if (sa_done) begin
                        done  <= 1;
                        state <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
