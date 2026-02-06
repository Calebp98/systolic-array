// Processing Element (PE) for output-stationary systolic array
// Each PE computes: acc += a_in * b_in
// Data flows: a_in -> a_out (horizontal, 1-cycle delay)
//             b_in -> b_out (vertical, 1-cycle delay)
// The 1-cycle delay on each path creates the systolic timing.
module pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
) (
    input  wire                    clk,
    input  wire                    rst_n,

    // Control
    input  wire                    clear_acc,    // Clear accumulator
    input  wire                    enable,       // Enable MAC operation

    // Data flow - horizontal (activations)
    input  wire signed [DATA_WIDTH-1:0] a_in,
    output reg  signed [DATA_WIDTH-1:0] a_out,

    // Data flow - vertical (weights)
    input  wire signed [DATA_WIDTH-1:0] b_in,
    output reg  signed [DATA_WIDTH-1:0] b_out,

    // Accumulated result
    output wire signed [ACC_WIDTH-1:0]  acc_out
);

    reg signed [ACC_WIDTH-1:0] acc;

    assign acc_out = acc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc   <= 0;
            a_out <= 0;
            b_out <= 0;
        end else begin
            // Pass data through to neighbors (1 cycle delay = systolic timing)
            a_out <= a_in;
            b_out <= b_in;

            if (clear_acc) begin
                acc <= 0;
            end else if (enable) begin
                acc <= acc + (a_in * b_in);
            end
        end
    end

endmodule
