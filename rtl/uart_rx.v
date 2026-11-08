`default_nettype none
module uart_rx #(
    parameter integer CLK_FREQ  = 24_000_000,
    parameter integer BAUD_RATE = 115200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx_pin,
    output reg  [7:0] data_out,
    output reg        valid_out
);
    localparam integer CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    reg rx_sync1, rx_sync2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_sync1 <= 1'b1;
            rx_sync2 <= 1'b1;
        end else begin
            rx_sync1 <= rx_pin;
            rx_sync2 <= rx_sync1;
        end
    end
    localparam [1:0]
        STATE_IDLE  = 2'd0,
        STATE_START = 2'd1,
        STATE_DATA  = 2'd2,
        STATE_STOP  = 2'd3;
    reg [1:0] state;
    reg [7:0] clk_cnt;
    reg [2:0] bit_idx;
    reg [7:0] rx_shift;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= STATE_IDLE;
            clk_cnt   <= 16'd0;
            bit_idx   <= 3'd0;
            rx_shift  <= 8'd0;
            data_out  <= 8'd0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= 1'b0;
            case (state)
                STATE_IDLE: begin
                    clk_cnt <= 16'd0;
                    bit_idx <= 3'd0;
                    if (!rx_sync2) begin
                        state <= STATE_START;
                    end
                end
                STATE_START: begin
                    if (clk_cnt == (CLKS_PER_BIT / 2)) begin
                        if (!rx_sync2) begin
                            clk_cnt <= 16'd0;
                            state   <= STATE_DATA;
                        end else begin
                            state <= STATE_IDLE;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end
                end
                STATE_DATA: begin
                    if (clk_cnt < CLKS_PER_BIT - 1) begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end else begin
                        clk_cnt <= 16'd0;
                        rx_shift[bit_idx] <= rx_sync2;
                        if (bit_idx < 3'd7) begin
                            bit_idx <= bit_idx + 1'b1;
                        end else begin
                            bit_idx <= 3'd0;
                            state   <= STATE_STOP;
                        end
                    end
                end
                STATE_STOP: begin
                    if (clk_cnt < CLKS_PER_BIT - 1) begin
                        clk_cnt <= clk_cnt + 1'b1;
                    end else begin
                        data_out  <= rx_shift;
                        valid_out <= 1'b1;
                        state     <= STATE_IDLE;
                    end
                end
                default: state <= STATE_IDLE;
            endcase
        end
    end
endmodule
