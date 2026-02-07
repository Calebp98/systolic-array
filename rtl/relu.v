// Combinational ReLU: out[i] = in[i] < 0 ? 0 : in[i]
// Processes N_ELEM elements of DATA_WIDTH bits each.
module relu #(
    parameter DATA_WIDTH = 32,
    parameter N_ELEM    = 4
) (
    input  wire [N_ELEM*DATA_WIDTH-1:0] data_in,
    output wire [N_ELEM*DATA_WIDTH-1:0] data_out
);

    genvar i;
    generate
        for (i = 0; i < N_ELEM; i = i + 1) begin : gen_relu
            wire signed [DATA_WIDTH-1:0] val = data_in[i*DATA_WIDTH +: DATA_WIDTH];
            assign data_out[i*DATA_WIDTH +: DATA_WIDTH] = val[DATA_WIDTH-1] ? {DATA_WIDTH{1'b0}} : val;
        end
    endgenerate

endmodule
