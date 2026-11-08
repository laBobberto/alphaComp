`default_nettype none
module lif_neuron #(
    parameter integer            WIDTH     = 10,        
    parameter signed [WIDTH-1:0] THRESHOLD = 10'sd256   
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    sim_start,   
    input  wire signed [WIDTH-1:0] bias_val,    
    input  wire                    acc_step,    
    input  wire signed [WIDTH-1:0] syn_weight,  
    input  wire                    ramp_step,   
    output wire                    spike_out,   
    output reg  signed [WIDTH-1:0] v_mem        
);
    assign spike_out = (~v_mem[WIDTH-1]) & v_mem[WIDTH-2];
    wire signed [WIDTH-1:0] op2 = acc_step ? syn_weight :
                                  ramp_step ? 10'sd16 : 10'sd0;
    wire signed [WIDTH-1:0] next_v = v_mem + op2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_mem <= 10'sd0;
        end else begin
            if (sim_start) begin
                v_mem <= bias_val;
            end else if (acc_step) begin
                v_mem <= next_v; 
            end else if (ramp_step) begin
                if (next_v[WIDTH-1])
                    v_mem <= 10'sd0; 
                else
                    v_mem <= next_v;
            end
        end
    end
endmodule
