// synthesis/vecmac_rtl.v
// VECMAC (Vector Multiply-Accumulate) RTL Design for TTA Architecture
//
// This module implements the core VECMAC functional unit optimized for
// AI workloads with support for various data types and sparsity awareness.

module vecmac_unit #(
    parameter VECTOR_WIDTH = 16,    // Number of parallel operations
    parameter DATA_WIDTH = 8,       // Bit width per element (8-bit quantized)
    parameter ACCUM_WIDTH = 32,     // Accumulator width to prevent overflow
    parameter SPARSITY_SUPPORT = 1  // Enable sparsity-aware optimizations
) (
    input wire clk,
    input wire rst_n,
    input wire enable,

    // Input vectors
    input wire [VECTOR_WIDTH*DATA_WIDTH-1:0] vector_a,
    input wire [VECTOR_WIDTH*DATA_WIDTH-1:0] vector_b,

    // Sparsity masks (1 = valid, 0 = skip)
    input wire [VECTOR_WIDTH-1:0] mask_a,
    input wire [VECTOR_WIDTH-1:0] mask_b,

    // Control signals
    input wire accumulate,          // 1 = accumulate, 0 = replace
    input wire [2:0] operation,     // 000=MUL, 001=MAC, 010=ADD, 011=SUB

    // Output
    output reg [ACCUM_WIDTH-1:0] result,
    output reg valid_out,
    output reg busy
);

    // Internal signals
    wire [VECTOR_WIDTH-1:0] element_valid;
    wire [VECTOR_WIDTH*ACCUM_WIDTH-1:0] partial_products;
    wire [ACCUM_WIDTH-1:0] sum_tree_result;
    reg [ACCUM_WIDTH-1:0] accumulator;

    // Pipeline registers for improved timing
    reg [VECTOR_WIDTH*DATA_WIDTH-1:0] vec_a_reg, vec_b_reg;
    reg [VECTOR_WIDTH-1:0] mask_a_reg, mask_b_reg;
    reg accumulate_reg;
    reg [2:0] operation_reg;
    reg stage1_valid, stage2_valid;

    // Generate element validity (sparsity-aware)
    generate
        if (SPARSITY_SUPPORT) begin : sparsity_gen
            assign element_valid = mask_a_reg & mask_b_reg;
        end else begin : no_sparsity_gen
            assign element_valid = {VECTOR_WIDTH{1'b1}};
        end
    endgenerate

    // Pipeline Stage 1: Input registration and multiplication
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vec_a_reg <= 0;
            vec_b_reg <= 0;
            mask_a_reg <= 0;
            mask_b_reg <= 0;
            accumulate_reg <= 0;
            operation_reg <= 0;
            stage1_valid <= 0;
        end else if (enable) begin
            vec_a_reg <= vector_a;
            vec_b_reg <= vector_b;
            mask_a_reg <= mask_a;
            mask_b_reg <= mask_b;
            accumulate_reg <= accumulate;
            operation_reg <= operation;
            stage1_valid <= 1;
        end else begin
            stage1_valid <= 0;
        end
    end

    // Generate parallel multipliers with sparsity gating
    generate
        genvar i;
        for (i = 0; i < VECTOR_WIDTH; i = i + 1) begin : mult_gen
            wire [DATA_WIDTH-1:0] a_elem = vec_a_reg[i*DATA_WIDTH +: DATA_WIDTH];
            wire [DATA_WIDTH-1:0] b_elem = vec_b_reg[i*DATA_WIDTH +: DATA_WIDTH];
            wire [2*DATA_WIDTH-1:0] product;

            // Multiply with zero-gating for sparsity
            assign product = element_valid[i] ? (a_elem * b_elem) : 0;

            // Sign-extend to accumulator width
            assign partial_products[i*ACCUM_WIDTH +: ACCUM_WIDTH] =
                {{(ACCUM_WIDTH-2*DATA_WIDTH){product[2*DATA_WIDTH-1]}}, product};
        end
    endgenerate

    // Optimized reduction tree for summation
    vecmac_reduction_tree #(
        .NUM_INPUTS(VECTOR_WIDTH),
        .DATA_WIDTH(ACCUM_WIDTH)
    ) reduction_tree_inst (
        .clk(clk),
        .rst_n(rst_n),
        .inputs(partial_products),
        .result(sum_tree_result)
    );

    // Pipeline Stage 2: Accumulation and output
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= 0;
            result <= 0;
            stage2_valid <= 0;
            valid_out <= 0;
        end else if (stage1_valid) begin
            case (operation_reg)
                3'b000: begin // MUL (no accumulation)
                    result <= sum_tree_result;
                    accumulator <= sum_tree_result;
                end
                3'b001: begin // MAC (multiply-accumulate)
                    if (accumulate_reg) begin
                        result <= accumulator + sum_tree_result;
                        accumulator <= accumulator + sum_tree_result;
                    end else begin
                        result <= sum_tree_result;
                        accumulator <= sum_tree_result;
                    end
                end
                3'b010: begin // ADD
                    result <= accumulator + sum_tree_result;
                    accumulator <= accumulator + sum_tree_result;
                end
                3'b011: begin // SUB
                    result <= accumulator - sum_tree_result;
                    accumulator <= accumulator - sum_tree_result;
                end
                default: begin
                    result <= sum_tree_result;
                    accumulator <= sum_tree_result;
                end
            endcase
            stage2_valid <= 1;
            valid_out <= 1;
        end else begin
            valid_out <= 0;
        end
    end

    // Busy signal generation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 0;
        end else begin
            busy <= enable || stage1_valid;
        end
    end

endmodule

// Optimized reduction tree for parallel summation
module vecmac_reduction_tree #(
    parameter NUM_INPUTS = 16,
    parameter DATA_WIDTH = 32
) (
    input wire clk,
    input wire rst_n,
    input wire [NUM_INPUTS*DATA_WIDTH-1:0] inputs,
    output reg [DATA_WIDTH-1:0] result
);

    // Calculate tree depth
    localparam TREE_DEPTH = $clog2(NUM_INPUTS);

    // Tree stages
    reg [DATA_WIDTH-1:0] tree_stage [0:TREE_DEPTH][0:NUM_INPUTS-1];

    integer i, j, stage;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 0;
            for (stage = 0; stage <= TREE_DEPTH; stage = stage + 1) begin
                for (i = 0; i < NUM_INPUTS; i = i + 1) begin
                    tree_stage[stage][i] <= 0;
                end
            end
        end else begin
            // Input stage
            for (i = 0; i < NUM_INPUTS; i = i + 1) begin
                tree_stage[0][i] <= inputs[i*DATA_WIDTH +: DATA_WIDTH];
            end

            // Reduction stages
            for (stage = 1; stage <= TREE_DEPTH; stage = stage + 1) begin
                for (i = 0; i < (NUM_INPUTS >> stage); i = i + 1) begin
                    tree_stage[stage][i] <= tree_stage[stage-1][2*i] + tree_stage[stage-1][2*i+1];
                end
            end

            // Output
            result <= tree_stage[TREE_DEPTH][0];
        end
    end

endmodule

// TTA Bus Interface for VECMAC
module tta_vecmac_wrapper #(
    parameter VECTOR_WIDTH = 16,
    parameter DATA_WIDTH = 8,
    parameter ACCUM_WIDTH = 32
) (
    input wire clk,
    input wire rst_n,

    // TTA Bus Interface
    input wire [31:0] tta_data_in,
    input wire tta_valid_in,
    input wire [2:0] tta_port_select,  // 000=vec_a, 001=vec_b, 010=mask_a, 011=mask_b, 100=ctrl

    output wire [31:0] tta_data_out,
    output wire tta_valid_out,
    output wire tta_busy
);

    // Internal VECMAC connections
    wire [VECTOR_WIDTH*DATA_WIDTH-1:0] vector_a, vector_b;
    wire [VECTOR_WIDTH-1:0] mask_a, mask_b;
    wire enable, accumulate;
    wire [2:0] operation;
    wire [ACCUM_WIDTH-1:0] vecmac_result;
    wire vecmac_valid, vecmac_busy;

    // Input port registers
    reg [VECTOR_WIDTH*DATA_WIDTH-1:0] vec_a_reg, vec_b_reg;
    reg [VECTOR_WIDTH-1:0] mask_a_reg, mask_b_reg;
    reg [7:0] control_reg; // [7]=enable, [6]=accumulate, [2:0]=operation

    // TTA Bus input handling
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vec_a_reg <= 0;
            vec_b_reg <= 0;
            mask_a_reg <= 0;
            mask_b_reg <= 0;
            control_reg <= 0;
        end else if (tta_valid_in) begin
            case (tta_port_select)
                3'b000: vec_a_reg <= tta_data_in[VECTOR_WIDTH*DATA_WIDTH-1:0];
                3'b001: vec_b_reg <= tta_data_in[VECTOR_WIDTH*DATA_WIDTH-1:0];
                3'b010: mask_a_reg <= tta_data_in[VECTOR_WIDTH-1:0];
                3'b011: mask_b_reg <= tta_data_in[VECTOR_WIDTH-1:0];
                3'b100: control_reg <= tta_data_in[7:0];
                default: ; // Do nothing
            endcase
        end
    end

    // Control signal extraction
    assign enable = control_reg[7];
    assign accumulate = control_reg[6];
    assign operation = control_reg[2:0];

    // VECMAC instantiation
    vecmac_unit #(
        .VECTOR_WIDTH(VECTOR_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .ACCUM_WIDTH(ACCUM_WIDTH),
        .SPARSITY_SUPPORT(1)
    ) vecmac_inst (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .vector_a(vec_a_reg),
        .vector_b(vec_b_reg),
        .mask_a(mask_a_reg),
        .mask_b(mask_b_reg),
        .accumulate(accumulate),
        .operation(operation),
        .result(vecmac_result),
        .valid_out(vecmac_valid),
        .busy(vecmac_busy)
    );

    // TTA Bus output
    assign tta_data_out = {{(32-ACCUM_WIDTH){vecmac_result[ACCUM_WIDTH-1]}}, vecmac_result};
    assign tta_valid_out = vecmac_valid;
    assign tta_busy = vecmac_busy;

endmodule