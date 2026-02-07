// Requantize: shift right and clamp int32 accumulator values to int8.
// out[i] = clamp(in[i] >>> shift_amount, -128, 127)
// Uses arithmetic right shift (sign-extending).
module requantize #(
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = 8,
    parameter N_ELEM    = 4
) (
    input  wire [N_ELEM*IN_WIDTH-1:0]  data_in,
    input  wire [4:0]                  shift_amount,
    output wire [N_ELEM*OUT_WIDTH-1:0] data_out
);

    genvar i;
    generate
        for (i = 0; i < N_ELEM; i = i + 1) begin : gen_requant
            wire signed [IN_WIDTH-1:0] val = data_in[i*IN_WIDTH +: IN_WIDTH];
            wire signed [IN_WIDTH-1:0] shifted = val >>> shift_amount;

            // Clamp to [-128, 127]
            wire overflow_pos = !shifted[IN_WIDTH-1] && (shifted > 127);
            wire overflow_neg = shifted[IN_WIDTH-1] && (shifted < -128);

            wire signed [OUT_WIDTH-1:0] clamped =
                overflow_pos ? 8'sd127 :
                overflow_neg ? -8'sd128 :
                shifted[OUT_WIDTH-1:0];

            assign data_out[i*OUT_WIDTH +: OUT_WIDTH] = clamped;
        end
    endgenerate

endmodule
