// Combinational bias addition: out[i] = in[i] + bias[i]
// Processes N_ELEM elements of DATA_WIDTH bits each.
module bias_add #(
    parameter DATA_WIDTH = 32,
    parameter N_ELEM    = 4
) (
    input  wire signed [N_ELEM*DATA_WIDTH-1:0] data_in,
    input  wire signed [N_ELEM*DATA_WIDTH-1:0] bias,
    output wire signed [N_ELEM*DATA_WIDTH-1:0] data_out
);

    genvar i;
    generate
        for (i = 0; i < N_ELEM; i = i + 1) begin : gen_add
            wire signed [DATA_WIDTH-1:0] a = data_in[i*DATA_WIDTH +: DATA_WIDTH];
            wire signed [DATA_WIDTH-1:0] b = bias[i*DATA_WIDTH +: DATA_WIDTH];
            assign data_out[i*DATA_WIDTH +: DATA_WIDTH] = a + b;
        end
    endgenerate

endmodule
