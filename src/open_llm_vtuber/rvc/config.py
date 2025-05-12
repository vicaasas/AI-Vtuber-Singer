import torch
import argparse
import sys
from multiprocessing import cpu_count

class Config:
    def __init__(self):
        self.device = "cuda:0"  # 預設設備為第一個 GPU
        self.is_half = True     # 預設啟用半精度（FP16）
        self.gpu_name = None    # GPU 名稱，後續檢測
        self.gpu_mem = None     # 顯存大小，後續檢測
        (
            self.python_cmd,
            self.listen_port,
            self.is_cli,
        ) = self.arg_parse()    # 解析命令行參數
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()  # 配置硬體參數

    @staticmethod
    def arg_parse() -> tuple:
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865, help="Listen port for web UI")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument("--is_cli", action="store_true", help="Use CLI instead of web UI")
        cmd_opts = parser.parse_args()

        # 確保端口號有效
        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.is_cli,
        )

    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    def device_config(self) -> tuple:
        # 假設 use_fp32_config 是一個外部函數，若不存在則需移除或實現
        def use_fp32_config():
            pass  # 根據實際需求實現或留空

        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            # 特定 GPU 型號禁用半精度
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("Found GPU", self.gpu_name, ", force to fp32")
                self.is_half = False
            else:
                print("Found GPU", self.gpu_name)
                use_fp32_config()
            # 計算顯存（單位：GB）
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024 / 1024 / 1024
                + 0.4
            )
        elif self.has_mps():
            print("No supported Nvidia GPU found, use MPS instead")
            self.device = "mps"
            self.is_half = False
            use_fp32_config()
        else:
            print("No supported Nvidia GPU found, use CPU instead")
            self.device = "cpu"
            self.is_half = False
            use_fp32_config()

        # 根據是否使用半精度和顯存大小設置參數
        if self.is_half:
            # 6G 顯存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G 顯存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        # 顯存 <= 4GB 時進一步降低參數
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max