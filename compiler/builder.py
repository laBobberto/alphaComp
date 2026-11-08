#!/usr/bin/env python3
import os
import subprocess
import shutil
from pathlib import Path
class FPGABuilder:
    def __init__(self, build_dir: Path, target_device: str='GW1NZ-LV1QN48C6/I5'):
        self.build_dir = Path(build_dir)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.target_device = target_device
        self.env = os.environ.copy()
        cad_suite_paths = ['/home/bobbert/oss-cad-suite/bin', '/home/bobbert/oss-cad-suite/py3bin'] if os.path.isdir('/home/bobbert/oss-cad-suite/bin') else []
        self.env['PATH'] = ':'.join(cad_suite_paths) + ':' + self.env.get('PATH', '')
    def run_cmd(self, cmd_args, step_name):
        print(f"\n===> [{step_name}] Running: {' '.join(cmd_args)}")
        res = subprocess.run(cmd_args, cwd=str(self.build_dir), env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode != 0:
            print(f'Error during {step_name}!\nOutput:\n{res.stdout}')
            raise RuntimeError(f"Step '{step_name}' failed with code {res.returncode}")
        print(f'===> [{step_name}] Success!')
        return res.stdout
    def build(self, verilog_files: list, cst_file: str):
        top_json = self.build_dir / 'top.json'
        pnr_json = self.build_dir / 'top_pnr.json'
        bitstream_fs = self.build_dir / 'top.fs'
        v_paths = ' '.join([f'"{f}"' for f in verilog_files])
        yosys_script = f'read_verilog {v_paths}; synth_gowin -top top -json {top_json.name}'
        self.run_cmd(['yosys', '-p', yosys_script], '1/3 Yosys Synthesis')
        nextpnr_cmd = ['nextpnr-himbaechel', '--json', str(top_json.name), '--write', str(pnr_json.name), '--device', self.target_device, '--vopt', f'cst={cst_file}', '--timing-allow-fail']
        self.run_cmd(nextpnr_cmd, '2/3 NextPNR Place & Route')
        pack_cmd = ['gowin_pack', '-d', 'GW1NZ-1', '-o', str(bitstream_fs.name), str(pnr_json.name)]
        self.run_cmd(pack_cmd, '3/3 Gowin Pack')
        print(f'\n🎉 Bitstream built successfully at: {bitstream_fs.resolve()}')
        return bitstream_fs
    def flash(self, bitstream_path: Path):
        print(f'\n===> Flashing to Tang Nano 1K...')
        flash_cmd = ['openFPGALoader', '-b', 'tangnano1k', str(bitstream_path)]
        try:
            res = subprocess.run(flash_cmd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(res.stdout)
            if res.returncode == 0:
                print('🎉 FPGA programmed successfully!')
                return True
            else:
                print('⚠️  FPGA programming failed (board might not be plugged in).')
                return False
        except Exception as e:
            print(f'⚠️  Could not run openFPGALoader: {e}')
            return False
