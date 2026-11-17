#!/usr/bin/env python3
import math
from typing import Dict, Any

class VerilogGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.arch = config.get('architecture', 'single_layer')
        self.is_hidden = self.arch == 'hidden_snn' or 'weights_l1' in config
        self.num_inputs = config['num_inputs']
        self.num_classes = config['num_classes']
        self.threshold = config.get('threshold', 200)
        self.leak_shift = config.get('leak_shift', 3)
        self.max_steps = config.get('max_steps', 32)
        self.input_idx_width = max(1, (self.num_inputs - 1).bit_length())
        self.class_width = max(1, (self.num_classes - 1).bit_length())

        if self.is_hidden:
            self.num_hidden = config.get('num_hidden', 12)
            self.weights_l1 = config['weights_l1']
            self.biases_l1 = config.get('biases_l1', [0] * self.num_hidden)
            self.weights_l2 = config['weights_l2']
            self.biases_l2 = config.get('biases_l2', [0] * self.num_classes)
            self.hidden_idx_width = max(1, (self.num_hidden - 1).bit_length())
            self.total_mem_depth = self.num_inputs + self.num_hidden
            self.mem_addr_width = max(7, (self.total_mem_depth - 1).bit_length())
        else:
            self.weights = config['weights']

        self.is_streaming = (self.num_inputs > 256)

    def generate_snn_core(self) -> str:
        if self.is_streaming:
            return "`default_nettype none\nmodule snn_core;\nendmodule\n"
        elif self.is_hidden:
            return self._generate_snn_core_hidden()
        else:
            return self._generate_snn_core_single()

    def _generate_snn_core_hidden(self) -> str:
        num_bram = (self.num_hidden + 3) // 4
        inits_by_block = [[] for _ in range(num_bram)]

        for i in range(self.num_inputs):
            for b in range(num_bram):
                val = 0
                for sub in range(4):
                    h_idx = b * 4 + sub
                    if h_idx < self.num_hidden:
                        w = int(self.weights_l1[h_idx][i]) & 0x1FF
                        val |= (w << (sub * 9))
                inits_by_block[b].append(f"        syn_mem_{b}[{i}] = 36'h{val:09X};")

        for j in range(self.num_hidden):
            row = self.num_inputs + j
            for b in range(num_bram):
                val = 0
                for sub in range(4):
                    c_idx = b * 4 + sub
                    if c_idx < self.num_classes:
                        w = int(self.weights_l2[c_idx][j]) & 0x1FF
                        val |= (w << (sub * 9))
                inits_by_block[b].append(f"        syn_mem_{b}[{row}] = 36'h{val:09X};")

        mem_depth = 256 if self.total_mem_depth <= 256 else 512
        mem_decls = []
        mem_reads = []
        for b in range(num_bram):
            mem_decls.append(f'    (* ram_style = "block" *) reg [35:0] syn_mem_{b} [0:{mem_depth - 1}];')
            mem_decls.append(f'    reg [35:0] syn_r{b};')
            mem_reads.append(f'        syn_r{b} <= syn_mem_{b}[mem_addr];')

        all_inits = []
        for b in range(num_bram):
            all_inits.extend(inits_by_block[b])

        mem_decls_str = "\n".join(mem_decls)
        mem_reads_str = "\n".join(mem_reads)
        mem_inits_str = "\n".join(all_inits)

        w_l1_wires = []
        for h in range(self.num_hidden):
            b = h // 4
            sub = h % 4
            off = sub * 9
            w_l1_wires.append(f"    assign w_l1[{h}] = {{{{1{{syn_r{b}[{off + 8}]}}}}, syn_r{b}[{off + 8}:{off}]}};")
        w_l1_str = "\n".join(w_l1_wires)

        w_l2_wires = []
        for c in range(self.num_classes):
            b = c // 4
            sub = c % 4
            off = sub * 9
            w_l2_wires.append(f"    assign w_l2[{c}] = {{{{1{{syn_r{b}[{off + 8}]}}}}, syn_r{b}[{off + 8}:{off}]}};")
        w_l2_str = "\n".join(w_l2_wires)

        b1_loads = []
        for k in range(self.num_hidden):
            bv = int(self.biases_l1[k])
            b_str = f"10'sd{bv}" if bv >= 0 else f"-10'sd{abs(bv)}"
            b1_loads.append(f"                    v_mem[{k}] <= {b_str};")
        b1_loads_str = "\n".join(b1_loads)

        b2_loads = []
        for c in range(self.num_classes):
            bv = int(self.biases_l2[c])
            b_str = f"10'sd{bv}" if bv >= 0 else f"-10'sd{abs(bv)}"
            b2_loads.append(f"                    v_mem[{c}] <= {b_str};")
        b2_loads_str = "\n".join(b2_loads)

        pe_lines = [f"        if (out_spikes[0]) pe_class = {self.class_width}'d0;"]
        for k in range(1, self.num_classes):
            pe_lines.append(f"        else if (out_spikes[{k}]) pe_class = {self.class_width}'d{k};")
        pe_lines.append(f"        else pe_class = {self.class_width}'d0;")
        pe_str = "\n".join(pe_lines)

        step_width = max(6, (self.num_inputs + 2).bit_length())

        code = f"""`default_nettype none

module snn_core #(
    parameter integer NUM_INPUTS    = {self.num_inputs},
    parameter integer NUM_HIDDEN    = {self.num_hidden},
    parameter integer NUM_CLASSES   = {self.num_classes},
    parameter integer MAX_STEPS     = {self.max_steps},
    parameter signed [9:0] THRESHOLD = 10'sd{self.threshold},
    parameter integer LEAK_SHIFT    = {self.leak_shift}
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     sim_start,
    input  wire [NUM_INPUTS-1:0]    pixels,
    output reg                      done,
    output reg  [{self.class_width - 1}:0]               best_class,
    output reg  [5:0]               ttfs_step
);

{mem_decls_str}
    reg [{self.mem_addr_width - 1}:0] mem_addr;

    initial begin
{mem_inits_str}
    end

    always @(posedge clk) begin
{mem_reads_str}
    end

    wire signed [9:0] w_l1 [0:{self.num_hidden - 1}];
{w_l1_str}

    wire signed [9:0] w_l2 [0:{self.num_classes - 1}];
{w_l2_str}

    reg signed [9:0] v_mem [0:{self.num_hidden - 1}];
    reg [{self.num_hidden - 1}:0] h_spikes;

    reg pixel_pipe;
    reg l1_valid;
    reg spike_pipe;
    reg l2_valid;

    localparam [2:0]
        ST_IDLE      = 3'd0,
        ST_L1_LOAD   = 3'd1,
        ST_L1_ACC    = 3'd2,
        ST_L1_SPIKE  = 3'd3,
        ST_L2_ACC    = 3'd4,
        ST_TTFS_RAMP = 3'd5;

    reg [2:0] state;
    reg [{step_width - 1}:0] step_cnt;

    wire [{self.num_classes - 1}:0] out_spikes;
    genvar g;
    generate
        for (g = 0; g < {self.num_classes}; g = g + 1) begin : gen_out_spikes
            assign out_spikes[g] = (~v_mem[g][9]) & v_mem[g][8];
        end
    endgenerate

    reg [{self.class_width - 1}:0] pe_class;
    always @(*) begin
{pe_str}
    end

    integer k;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= ST_IDLE;
            mem_addr   <= {self.mem_addr_width}'d0;
            step_cnt   <= {step_width}'d0;
            done       <= 1'b0;
            best_class <= {self.class_width}'d0;
            ttfs_step  <= 6'd0;
            h_spikes   <= {self.num_hidden}'d0;
            pixel_pipe <= 1'b0;
            l1_valid   <= 1'b0;
            spike_pipe <= 1'b0;
            l2_valid   <= 1'b0;
            for (k = 0; k < {self.num_hidden}; k = k + 1) v_mem[k] <= 10'sd0;
        end else begin
            done <= 1'b0;

            if (state == ST_L1_ACC) begin
                pixel_pipe <= pixels[mem_addr];
                l1_valid   <= 1'b1;
            end else begin
                pixel_pipe <= 1'b0;
                l1_valid   <= 1'b0;
            end

            if (state == ST_L2_ACC) begin
                spike_pipe <= h_spikes[mem_addr - {self.mem_addr_width}'d{self.num_inputs}];
                l2_valid   <= 1'b1;
            end else begin
                spike_pipe <= 1'b0;
                l2_valid   <= 1'b0;
            end

            if (l1_valid && pixel_pipe) begin
                for (k = 0; k < {self.num_hidden}; k = k + 1) begin
                    v_mem[k] <= v_mem[k] + w_l1[k];
                end
            end

            if (l2_valid && spike_pipe) begin
                for (k = 0; k < {self.num_classes}; k = k + 1) begin
                    v_mem[k] <= v_mem[k] + w_l2[k];
                end
            end

            case (state)
                ST_IDLE: begin
                    if (sim_start) begin
                        state    <= ST_L1_LOAD;
                        mem_addr <= {self.mem_addr_width}'d0;
                        step_cnt <= {step_width}'d0;
                    end
                end

                ST_L1_LOAD: begin
{b1_loads_str}
                    state    <= ST_L1_ACC;
                    mem_addr <= {self.mem_addr_width}'d0;
                    step_cnt <= {step_width}'d0;
                end

                ST_L1_ACC: begin
                    if (mem_addr < {self.mem_addr_width}'d{self.num_inputs - 1}) begin
                        mem_addr <= mem_addr + 1'b1;
                    end else begin
                        mem_addr <= {self.mem_addr_width}'d{self.num_inputs};
                    end

                    if (step_cnt == {step_width}'d{self.num_inputs}) begin
                        state    <= ST_L1_SPIKE;
                        step_cnt <= {step_width}'d0;
                    end else begin
                        step_cnt <= step_cnt + 1'b1;
                    end
                end

                ST_L1_SPIKE: begin
                    for (k = 0; k < {self.num_hidden}; k = k + 1) begin
                        h_spikes[k] <= ~v_mem[k][9];
                    end
{b2_loads_str}
                    state    <= ST_L2_ACC;
                    mem_addr <= {self.mem_addr_width}'d{self.num_inputs};
                    step_cnt <= {step_width}'d0;
                end

                ST_L2_ACC: begin
                    if (mem_addr < {self.mem_addr_width}'d{self.total_mem_depth - 1}) begin
                        mem_addr <= mem_addr + 1'b1;
                    end

                    if (step_cnt == {step_width}'d{self.num_hidden}) begin
                        state    <= ST_TTFS_RAMP;
                        step_cnt <= {step_width}'d0;
                    end else begin
                        step_cnt <= step_cnt + 1'b1;
                    end
                end

                ST_TTFS_RAMP: begin
                    for (k = 0; k < {self.num_classes}; k = k + 1) begin
                        v_mem[k] <= v_mem[k] + 10'sd16;
                    end

                    if (|out_spikes) begin
                        done       <= 1'b1;
                        best_class <= pe_class;
                        ttfs_step  <= step_cnt[5:0];
                        state      <= ST_IDLE;
                    end else if (step_cnt >= {step_width}'d{self.max_steps - 1}) begin
                        done       <= 1'b1;
                        best_class <= pe_class;
                        ttfs_step  <= step_cnt[5:0];
                        state      <= ST_IDLE;
                    end else begin
                        step_cnt <= step_cnt + 1'b1;
                    end
                end
            endcase
        end
    end

endmodule
"""
        return code

    def _generate_snn_core_single(self) -> str:
        blocks = []
        c_idx = 0
        while c_idx < self.num_classes:
            chunk = min(4, self.num_classes - c_idx)
            blocks.append((c_idx, chunk))
            c_idx += chunk

        mem_decls = []
        syn_reg_decls = []
        mem_reads = []
        mem_inits = []
        syn_wires = []

        for b_idx, (start_c, count) in enumerate(blocks):
            width = count * 9
            mem_decls.append(f'    (* ram_style = "block" *) reg [{width - 1}:0] syn_mem_blk_{b_idx} [0:NUM_INPUTS-1];')
            syn_reg_decls.append(f'    reg [{width - 1}:0] syn_data_blk_{b_idx};')
            mem_reads.append(f'        syn_data_blk_{b_idx} <= syn_mem_blk_{b_idx}[acc_addr];')

            for i in range(self.num_inputs):
                val_num = 0
                for sub_c in range(count):
                    w = int(self.weights[start_c + sub_c][i])
                    w_9b = w & 0x1FF
                    val_num |= (w_9b << (sub_c * 9))
                hex_digits = (width + 3) // 4
                mem_inits.append(f"        syn_mem_blk_{b_idx}[{i}] = {width}'h{val_num:0{hex_digits}X};")

            for sub_c in range(count):
                c = start_c + sub_c
                offset = sub_c * 9
                syn_wires.append(
                    f"    wire signed [9:0] syn_w_{c} = {{ {{1{{syn_data_blk_{b_idx}[{offset + 8}]}}}}, syn_data_blk_{b_idx}[{offset + 8} : {offset}] }};"
                )

        mem_decls_str = "\n".join(mem_decls)
        syn_reg_decls_str = "\n".join(syn_reg_decls)
        mem_reads_str = "\n".join(mem_reads)
        mem_inits_str = "\n".join(mem_inits)
        syn_wires_str = "\n".join(syn_wires)

        neuron_insts = []
        for c in range(self.num_classes):
            b_val = int(self.biases[c])
            b_str = f"10'sd{b_val}" if b_val >= 0 else f"-10'sd{abs(b_val)}"
            neuron_insts.append(
                f"""        lif_neuron #(
            .WIDTH(10),
            .THRESHOLD(10'sd256)
        ) u_lif_{c} (
            .clk(clk),
            .rst_n(rst_n),
            .sim_start(sim_start),
            .bias_val({b_str}),
            .acc_step(acc_step),
            .syn_weight(syn_w_{c}),
            .ramp_step(sim_active),
            .spike_out(spikes[{c}]),
            .v_mem(v_mems[{c}])
        );"""
            )
        neuron_insts_str = "\n".join(neuron_insts)

        pe_lines = [f"        if (spikes[0]) winning_class = {self.class_width}'d0;"]
        for k in range(1, self.num_classes):
            pe_lines.append(f"        else if (spikes[{k}]) winning_class = {self.class_width}'d{k};")
        pe_lines.append(f"        else winning_class = {self.class_width}'d0;")
        pe_str = "\n".join(pe_lines)

        code = f"""`default_nettype none

module snn_core #(
    parameter integer NUM_INPUTS   = {self.num_inputs},
    parameter integer NUM_CLASSES  = {self.num_classes},
    parameter signed [9:0] THRESHOLD = 10'sd{self.threshold},
    parameter integer LEAK_SHIFT   = {self.leak_shift},
    parameter integer MAX_STEPS    = {self.max_steps}
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    sim_start,
    input  wire [NUM_INPUTS-1:0]   pixels,
    output reg                     done,
    output reg  [{self.class_width - 1}:0]       best_class,
    output reg  [5:0]              ttfs_step
);

{mem_decls_str}
{syn_reg_decls_str}
    reg [{self.input_idx_width - 1}:0] acc_addr;

    initial begin
{mem_inits_str}
    end

    always @(posedge clk) begin
{mem_reads_str}
    end

{syn_wires_str}

    reg              acc_active;
    reg              acc_valid;
    reg              pixel_pipe;
    reg              sim_active;

    wire acc_step = acc_valid && pixel_pipe;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_addr   <= {self.input_idx_width}'d0;
            acc_active <= 1'b0;
            acc_valid  <= 1'b0;
            pixel_pipe <= 1'b0;
            sim_active <= 1'b0;
        end else begin
            if (sim_start) begin
                acc_addr   <= {self.input_idx_width}'d0;
                acc_active <= 1'b1;
                acc_valid  <= 1'b0;
                pixel_pipe <= 1'b0;
                sim_active <= 1'b0;
            end else begin
                if (acc_active) begin
                    acc_valid  <= 1'b1;
                    pixel_pipe <= pixels[acc_addr];
                    if (acc_addr == {self.input_idx_width}'d{self.num_inputs - 1}) begin
                        acc_active <= 1'b0;
                    end else begin
                        acc_addr <= acc_addr + 1'b1;
                    end
                end else begin
                    acc_valid <= 1'b0;
                    if (acc_valid) begin
                        sim_active <= 1'b1;
                    end
                end

                if (done) begin
                    sim_active <= 1'b0;
                end
            end
        end
    end

    wire [NUM_CLASSES-1:0] spikes;
    wire signed [9:0]      v_mems [0:NUM_CLASSES-1];

{neuron_insts_str}

    reg [5:0] step_cnt;
    wire any_spike = |spikes;

    reg [{self.class_width - 1}:0] winning_class;
    always @(*) begin
{pe_str}
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            step_cnt   <= 6'd0;
            done       <= 1'b0;
            best_class <= {self.class_width}'d0;
            ttfs_step  <= 6'd0;
        end else begin
            done <= 1'b0;

            if (sim_active && !done) begin
                if (any_spike) begin
                    done       <= 1'b1;
                    ttfs_step  <= step_cnt;
                    best_class <= winning_class;
                    step_cnt   <= 6'd0;
                end else if (step_cnt >= MAX_STEPS - 1) begin
                    done       <= 1'b1;
                    ttfs_step  <= step_cnt;
                    best_class <= winning_class;
                    step_cnt   <= 6'd0;
                end else begin
                    step_cnt <= step_cnt + 1'b1;
                end
            end else begin
                step_cnt <= 6'd0;
            end
        end
    end

endmodule
"""
        return code

    def generate_top(self) -> str:
        if self.is_streaming:
            return self._generate_top_streaming()
        pixel_cnt_width = max(1, self.num_inputs.bit_length())
        extra_param = f",\n        .NUM_HIDDEN({self.num_hidden})" if self.is_hidden else ""
        code = f"""`default_nettype none

module top #(
    parameter CLK_FREQ = 27_000_000
)(
    input  wire       sys_clk,
    input  wire       sys_rst_n,
    input  wire       key_b,
    input  wire       uart_rx,
    output wire       uart_tx,
    output wire [2:0] led
);

    reg [7:0] por_cnt = 8'd0;
    wire rst_n = (por_cnt == 8'd255);
    always @(posedge sys_clk) begin
        if (por_cnt < 8'd255)
            por_cnt <= por_cnt + 1'b1;
    end

    wire [7:0] rx_byte;
    wire       rx_valid;

    uart_rx #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(115200)
    ) u_uart_rx (\n        .clk(sys_clk),
        .rst_n(rst_n),
        .rx_pin(uart_rx),
        .data_out(rx_byte),
        .valid_out(rx_valid)
    );

    reg [{self.num_inputs - 1}:0] pixel_shift;
    reg [{pixel_cnt_width}:0]   pixel_cnt;
    reg                         snn_start;

    always @(posedge sys_clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_shift <= {self.num_inputs}'d0;
            pixel_cnt   <= {pixel_cnt_width + 1}'d0;
            snn_start   <= 1'b0;
        end else begin
            snn_start <= 1'b0;

            if (rx_valid) begin
                if (rx_byte == "0" || rx_byte == "1") begin
                    pixel_shift <= {{ (rx_byte == "1"), pixel_shift[{self.num_inputs - 1}:1] }};
                    if (pixel_cnt == {pixel_cnt_width + 1}'d{self.num_inputs - 1}) begin
                        snn_start <= 1'b1;
                        pixel_cnt <= {pixel_cnt_width + 1}'d0;
                    end else begin
                        pixel_cnt <= pixel_cnt + 1'b1;
                    end
                end else if (rx_byte == 8'h0A || rx_byte == 8'h0D) begin
                    pixel_cnt <= {pixel_cnt_width + 1}'d0;
                end
            end
        end
    end

    wire                          snn_done;
    wire [{self.class_width - 1}:0] snn_class;
    wire [5:0]                    snn_ttfs;

    snn_core #(
        .NUM_INPUTS({self.num_inputs}),
        .NUM_CLASSES({self.num_classes}){extra_param},
        .THRESHOLD(10'sd{self.threshold}),
        .LEAK_SHIFT({self.leak_shift}),
        .MAX_STEPS({self.max_steps})
    ) u_snn (
        .clk(sys_clk),
        .rst_n(rst_n),
        .sim_start(snn_start),
        .pixels(pixel_shift),
        .done(snn_done),
        .best_class(snn_class),
        .ttfs_step(snn_ttfs)
    );

    reg [4:0]  tx_idx;
    reg [{self.class_width - 1}:0] saved_class;
    reg [5:0]  saved_ttfs;
    reg [7:0]  tx_char;
    reg        tx_start;
    reg        tx_sending;
    wire       tx_busy;

    reg [3:0] ttfs_tens;
    reg [3:0] ttfs_units;
    always @(*) begin
        if (saved_ttfs >= 6'd30) begin
            ttfs_tens  = 4'd3;
            ttfs_units = saved_ttfs - 6'd30;
        end else if (saved_ttfs >= 6'd20) begin
            ttfs_tens  = 4'd2;
            ttfs_units = saved_ttfs - 6'd20;
        end else if (saved_ttfs >= 6'd10) begin
            ttfs_tens  = 4'd1;
            ttfs_units = saved_ttfs - 6'd10;
        end else begin
            ttfs_tens  = 4'd0;
            ttfs_units = saved_ttfs[3:0];
        end
    end

    always @(*) begin
        case (tx_idx)
            5'd0:  tx_char = "C";
            5'd1:  tx_char = "l";
            5'd2:  tx_char = "a";
            5'd3:  tx_char = "s";
            5'd4:  tx_char = "s";
            5'd5:  tx_char = ":";
            5'd6:  tx_char = " ";
            5'd7:  tx_char = 8'd48 + {{ 4'd0, saved_class }};
            5'd8:  tx_char = " ";
            5'd9:  tx_char = "(";
            5'd10: tx_char = "T";
            5'd11: tx_char = "T";
            5'd12: tx_char = "F";
            5'd13: tx_char = "S";
            5'd14: tx_char = ":";
            5'd15: tx_char = " ";
            5'd16: tx_char = 8'd48 + {{ 4'd0, ttfs_tens }};
            5'd17: tx_char = 8'd48 + {{ 4'd0, ttfs_units }};
            5'd18: tx_char = ")";
            5'd19: tx_char = 8'h0D;
            5'd20: tx_char = 8'h0A;
            default: tx_char = 8'd0;
        endcase
    end

    reg [7:0] tx_data_reg;

    always @(posedge sys_clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_idx      <= 5'd0;
            tx_start    <= 1'b0;
            tx_sending  <= 1'b0;
            saved_class <= {self.class_width}'d0;
            saved_ttfs  <= 6'd0;
            tx_data_reg <= 8'd0;
        end else begin
            tx_start <= 1'b0;

            if (snn_done) begin
                saved_class <= snn_class;
                saved_ttfs  <= snn_ttfs;
                tx_idx      <= 5'd0;
                tx_sending  <= 1'b1;
            end else if (tx_sending) begin
                if (!tx_busy && !tx_start) begin
                    if (tx_idx < 5'd21) begin
                        tx_data_reg <= tx_char;
                        tx_start    <= 1'b1;
                        tx_idx      <= tx_idx + 1'b1;
                    end else begin
                        tx_sending <= 1'b0;
                    end
                end
            end
        end
    end

    uart_tx #(
        .CLK_FREQ(CLK_FREQ),
        .BAUD_RATE(115200)
    ) u_uart_tx (
        .clk(sys_clk),
        .rst_n(rst_n),
        .tx_data(tx_data_reg),
        .tx_start(tx_start),
        .tx_pin(uart_tx),
        .tx_busy(tx_busy)
    );

    assign led[0] = ~saved_class[0];
    assign led[1] = (self_classes_gt_1) ? ~saved_class[1] : 1'b1;
    assign led[2] = (self_classes_gt_2) ? ~saved_class[2] : 1'b1;

endmodule
""".replace("(self_classes_gt_1)", "1'b1" if self.num_classes > 1 else "1'b0").replace("(self_classes_gt_2)", "1'b1" if self.num_classes > 2 else "1'b0")
        return code

    def _generate_top_streaming(self) -> str:
        inits = [[] for _ in range(4)]
        for i in range(self.num_inputs):
            val0 = (int(self.weights_l1[0][i]) & 0x3F) | ((int(self.weights_l1[1][i]) & 0x3F) << 6) | ((int(self.weights_l1[2][i]) & 0x3F) << 12)
            inits[0].append(f"        syn_mem_0[{i}] = 18'h{val0:05X};")
            val1 = (int(self.weights_l1[3][i]) & 0x3F) | ((int(self.weights_l1[4][i]) & 0x3F) << 6) | ((int(self.weights_l1[5][i]) & 0x3F) << 12)
            inits[1].append(f"        syn_mem_1[{i}] = 18'h{val1:05X};")
            val2 = (int(self.weights_l1[6][i]) & 0x3F) | ((int(self.weights_l1[7][i]) & 0x3F) << 6) | ((int(self.weights_l1[8][i]) & 0x3F) << 12)
            inits[2].append(f"        syn_mem_2[{i}] = 18'h{val2:05X};")
            val3 = (int(self.weights_l1[9][i]) & 0x3F) | ((int(self.weights_l1[10][i]) & 0x3F) << 6) | ((int(self.weights_l1[11][i]) & 0x3F) << 12)
            inits[3].append(f"        syn_mem_3[{i}] = 18'h{val3:05X};")

        for j in range(self.num_hidden):
            row = self.num_inputs + j
            v0 = (int(self.weights_l2[0][j]) & 0x3F) | ((int(self.weights_l2[1][j]) & 0x3F) << 6) | ((int(self.weights_l2[2][j]) & 0x3F) << 12)
            inits[0].append(f"        syn_mem_0[{row}] = 18'h{v0:05X};")
            v1 = (int(self.weights_l2[3][j]) & 0x3F) | ((int(self.weights_l2[4][j]) & 0x3F) << 6) | ((int(self.weights_l2[5][j]) & 0x3F) << 12)
            inits[1].append(f"        syn_mem_1[{row}] = 18'h{v1:05X};")
            v2 = (int(self.weights_l2[6][j]) & 0x3F) | ((int(self.weights_l2[7][j]) & 0x3F) << 6) | ((int(self.weights_l2[8][j]) & 0x3F) << 12)
            inits[2].append(f"        syn_mem_2[{row}] = 18'h{v2:05X};")
            v3 = int(self.weights_l2[9][j]) & 0x3F
            inits[3].append(f"        syn_mem_3[{row}] = 18'h{v3:05X};")

        inits_str = "\n".join(inits[0] + inits[1] + inits[2] + inits[3])

        b1_loads = []
        for k in range(self.num_hidden):
            val = int(self.biases_l1[k])
            b_str = f"-10'sd{abs(val)}" if val < 0 else f"10'sd{val}"
            b1_loads.append(f"                v_mem[{k}] <= {b_str};")
        b1_str = "\n".join(b1_loads)

        b2_loads = []
        for c in range(self.num_classes):
            val = int(self.biases_l2[c])
            b_str = f"-10'sd{abs(val)}" if val < 0 else f"10'sd{val}"
            b2_loads.append(f"                    v_mem[{c}] <= {b_str};")
        b2_str = "\n".join(b2_loads)

        code = f"""`default_nettype none

module top #(
    parameter CLK_FREQ = 27_000_000
)(
    input  wire       sys_clk,
    input  wire       sys_rst_n,
    input  wire       key_b,
    input  wire       uart_rx,
    output wire       uart_tx,
    output wire [2:0] led
);

    reg [7:0] por = 8'd0;
    wire rst_n = (por == 8'd255);
    always @(posedge sys_clk) if (por < 8'd255) por <= por + 1'b1;

    wire [7:0] rx_byte;
    wire       rx_valid;
    uart_rx #(.CLK_FREQ(CLK_FREQ), .BAUD_RATE(115200)) u_rx (
        .clk(sys_clk), .rst_n(rst_n), .rx_pin(uart_rx), .data_out(rx_byte), .valid_out(rx_valid)
    );

    (* ram_style = "block" *) reg [17:0] syn_mem_0 [0:1023];
    (* ram_style = "block" *) reg [17:0] syn_mem_1 [0:1023];
    (* ram_style = "block" *) reg [17:0] syn_mem_2 [0:1023];
    (* ram_style = "block" *) reg [17:0] syn_mem_3 [0:1023];

    reg [9:0]  mem_addr;
    reg [17:0] syn_r0, syn_r1, syn_r2, syn_r3;

    initial begin
{inits_str}
    end

    always @(posedge sys_clk) begin
        syn_r0 <= syn_mem_0[mem_addr];
        syn_r1 <= syn_mem_1[mem_addr];
        syn_r2 <= syn_mem_2[mem_addr];
        syn_r3 <= syn_mem_3[mem_addr];
    end

    wire signed [9:0] w_h [0:11];
    assign w_h[0]  = {{{{4{{syn_r0[5]}}}},  syn_r0[5:0]}};
    assign w_h[1]  = {{{{4{{syn_r0[11]}}}}, syn_r0[11:6]}};
    assign w_h[2]  = {{{{4{{syn_r0[17]}}}}, syn_r0[17:12]}};
    assign w_h[3]  = {{{{4{{syn_r1[5]}}}},  syn_r1[5:0]}};
    assign w_h[4]  = {{{{4{{syn_r1[11]}}}}, syn_r1[11:6]}};
    assign w_h[5]  = {{{{4{{syn_r1[17]}}}}, syn_r1[17:12]}};
    assign w_h[6]  = {{{{4{{syn_r2[5]}}}},  syn_r2[5:0]}};
    assign w_h[7]  = {{{{4{{syn_r2[11]}}}}, syn_r2[11:6]}};
    assign w_h[8]  = {{{{4{{syn_r2[17]}}}}, syn_r2[17:12]}};
    assign w_h[9]  = {{{{4{{syn_r3[5]}}}},  syn_r3[5:0]}};
    assign w_h[10] = {{{{4{{syn_r3[11]}}}}, syn_r3[11:6]}};
    assign w_h[11] = {{{{4{{syn_r3[17]}}}}, syn_r3[17:12]}};

    wire signed [9:0] w_o [0:9];
    assign w_o[0]  = {{{{4{{syn_r0[5]}}}},  syn_r0[5:0]}};
    assign w_o[1]  = {{{{4{{syn_r0[11]}}}}, syn_r0[11:6]}};
    assign w_o[2]  = {{{{4{{syn_r0[17]}}}}, syn_r0[17:12]}};
    assign w_o[3]  = {{{{4{{syn_r1[5]}}}},  syn_r1[5:0]}};
    assign w_o[4]  = {{{{4{{syn_r1[11]}}}}, syn_r1[11:6]}};
    assign w_o[5]  = {{{{4{{syn_r1[17]}}}}, syn_r1[17:12]}};
    assign w_o[6]  = {{{{4{{syn_r2[5]}}}},  syn_r2[5:0]}};
    assign w_o[7]  = {{{{4{{syn_r2[11]}}}}, syn_r2[11:6]}};
    assign w_o[8]  = {{{{4{{syn_r2[17]}}}}, syn_r2[17:12]}};
    assign w_o[9]  = {{{{4{{syn_r3[5]}}}},  syn_r3[5:0]}};

    reg signed [9:0] v_mem [0:11];
    reg [11:0]       h_spikes;

    reg [9:0] pixel_cnt;
    reg [2:0] state;
    reg [5:0] ramp_cnt;
    reg [3:0] best_class;
    reg [5:0] saved_ttfs;
    reg [3:0] l2_idx;

    wire [9:0] out_spikes;
    genvar g;
    generate
        for (g = 0; g < 10; g = g + 1) begin : gen_out_spikes
            assign out_spikes[g] = (~v_mem[g][9]) & v_mem[g][8];
        end
    endgenerate

    reg [3:0] pe_class;
    always @(*) begin
        if (out_spikes[0]) pe_class = 4'd0;
        else if (out_spikes[1]) pe_class = 4'd1;
        else if (out_spikes[2]) pe_class = 4'd2;
        else if (out_spikes[3]) pe_class = 4'd3;
        else if (out_spikes[4]) pe_class = 4'd4;
        else if (out_spikes[5]) pe_class = 4'd5;
        else if (out_spikes[6]) pe_class = 4'd6;
        else if (out_spikes[7]) pe_class = 4'd7;
        else if (out_spikes[8]) pe_class = 4'd8;
        else if (out_spikes[9]) pe_class = 4'd9;
        else pe_class = 4'd0;
    end

    localparam ST_STREAM_RX = 3'd0;
    localparam ST_WAIT_BRAM = 3'd1;
    localparam ST_ACC_PIXEL = 3'd2;
    localparam ST_SPIKE_L1  = 3'd3;
    localparam ST_WAIT_L2   = 3'd4;
    localparam ST_ACC_L2    = 3'd5;
    localparam ST_RAMP      = 3'd6;
    localparam ST_TX        = 3'd7;

    reg [4:0]  tx_idx;
    reg [7:0]  tx_char;
    reg        tx_start;
    reg        tx_sending;
    wire       tx_busy;

    reg [3:0] ttfs_tens;
    reg [3:0] ttfs_units;
    always @(*) begin
        if (saved_ttfs >= 6'd30) begin
            ttfs_tens  = 4'd3;
            ttfs_units = saved_ttfs - 6'd30;
        end else if (saved_ttfs >= 6'd20) begin
            ttfs_tens  = 4'd2;
            ttfs_units = saved_ttfs - 6'd20;
        end else if (saved_ttfs >= 6'd10) begin
            ttfs_tens  = 4'd1;
            ttfs_units = saved_ttfs - 6'd10;
        end else begin
            ttfs_tens  = 4'd0;
            ttfs_units = saved_ttfs[3:0];
        end
    end

    always @(*) begin
        case (tx_idx)
            5'd0:  tx_char = "C";
            5'd1:  tx_char = "l";
            5'd2:  tx_char = "a";
            5'd3:  tx_char = "s";
            5'd4:  tx_char = "s";
            5'd5:  tx_char = ":";
            5'd6:  tx_char = " ";
            5'd7:  tx_char = 8'd48 + {{ 4'd0, best_class }};
            5'd8:  tx_char = " ";
            5'd9:  tx_char = "(";
            5'd10: tx_char = "T";
            5'd11: tx_char = "T";
            5'd12: tx_char = "F";
            5'd13: tx_char = "S";
            5'd14: tx_char = ":";
            5'd15: tx_char = " ";
            5'd16: tx_char = 8'd48 + {{ 4'd0, ttfs_tens }};
            5'd17: tx_char = 8'd48 + {{ 4'd0, ttfs_units }};
            5'd18: tx_char = ")";
            5'd19: tx_char = 8'h0D;
            5'd20: tx_char = 8'h0A;
            default: tx_char = " ";
        endcase
    end

    integer k;

    always @(posedge sys_clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= ST_STREAM_RX;
            mem_addr   <= 10'd0;
            pixel_cnt  <= 10'd0;
            l2_idx     <= 4'd0;
            ramp_cnt   <= 6'd0;
            best_class <= 4'd0;
            saved_ttfs <= 6'd0;
            tx_start   <= 1'b0;
            tx_sending <= 1'b0;
            tx_idx     <= 5'd0;
            h_spikes   <= 12'd0;
{b1_str}
        end else begin
            tx_start <= 1'b0;

            if (tx_sending) begin
                if (!tx_busy && !tx_start) begin
                    if (tx_idx == 5'd20) begin
                        tx_sending <= 1'b0;
                        state      <= ST_STREAM_RX;
                        pixel_cnt  <= 10'd0;
{b1_str}
                    end else begin
                        tx_idx   <= tx_idx + 1'b1;
                        tx_start <= 1'b1;
                    end
                end
            end else begin
                case (state)
                    ST_STREAM_RX: begin
                        if (rx_valid) begin
                            if (rx_byte == "0" || rx_byte == "1") begin
                                mem_addr <= pixel_cnt;
                                if (rx_byte == "1") begin
                                    state <= ST_WAIT_BRAM;
                                end else begin
                                    if (pixel_cnt == 10'd783) begin
                                        state <= ST_SPIKE_L1;
                                        pixel_cnt <= 10'd0;
                                    end else begin
                                        pixel_cnt <= pixel_cnt + 1'b1;
                                    end
                                end
                            end else if (rx_byte == 8'h0A || rx_byte == 8'h0D) begin
                                if (pixel_cnt > 10'd0) begin
                                    state <= ST_SPIKE_L1;
                                    pixel_cnt <= 10'd0;
                                end
                            end
                        end
                    end

                    ST_WAIT_BRAM: begin
                        state <= ST_ACC_PIXEL;
                    end

                    ST_ACC_PIXEL: begin
                        for (k = 0; k < 12; k = k + 1) begin
                            v_mem[k] <= v_mem[k] + w_h[k];
                        end
                        if (pixel_cnt == 10'd783) begin
                            state <= ST_SPIKE_L1;
                            pixel_cnt <= 10'd0;
                        end else begin
                            pixel_cnt <= pixel_cnt + 1'b1;
                            state <= ST_STREAM_RX;
                        end
                    end

                    ST_SPIKE_L1: begin
                        for (k = 0; k < 12; k = k + 1) begin
                            h_spikes[k] <= ~v_mem[k][9];
                        end
{b2_str}
                        mem_addr <= 10'd784;
                        l2_idx   <= 4'd0;
                        state    <= ST_WAIT_L2;
                    end

                    ST_WAIT_L2: begin
                        state <= ST_ACC_L2;
                    end

                    ST_ACC_L2: begin
                        if (h_spikes[l2_idx]) begin
                            for (k = 0; k < 10; k = k + 1) begin
                                v_mem[k] <= v_mem[k] + w_o[k];
                            end
                        end
                        if (l2_idx == 4'd11) begin
                            state    <= ST_RAMP;
                            ramp_cnt <= 6'd0;
                        end else begin
                            l2_idx   <= l2_idx + 1'b1;
                            mem_addr <= mem_addr + 1'b1;
                            state    <= ST_WAIT_L2;
                        end
                    end

                    ST_RAMP: begin
                        for (k = 0; k < 10; k = k + 1) begin
                            v_mem[k] <= v_mem[k] + 10'sd16;
                        end
                        if (|out_spikes) begin
                            best_class <= pe_class;
                            saved_ttfs <= ramp_cnt;
                            state      <= ST_TX;
                        end else if (ramp_cnt >= 6'd31) begin
                            best_class <= pe_class;
                            saved_ttfs <= 6'd31;
                            state      <= ST_TX;
                        end else begin
                            ramp_cnt <= ramp_cnt + 1'b1;
                        end
                    end

                    ST_TX: begin
                        tx_sending <= 1'b1;
                        tx_idx     <= 5'd0;
                        tx_start   <= 1'b1;
                    end
                endcase
            end
        end
    end

    uart_tx #(.CLK_FREQ(CLK_FREQ), .BAUD_RATE(115200)) u_tx (
        .clk(sys_clk), .rst_n(rst_n), .tx_data(tx_char), .tx_start(tx_start), .tx_pin(uart_tx), .tx_busy(tx_busy)
    );

    assign led = ~best_class[2:0];

endmodule
"""
        return code
