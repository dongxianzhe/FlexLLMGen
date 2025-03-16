from dataclasses import dataclass
from typing import Any

global_cpu_device = None
global_disk_device = None

def set_global_cpu_device(value):
    global global_cpu_device
    global_cpu_device = value

def set_global_disk_device(value):
    global global_disk_device
    global_disk_device = value

def get_global_cpu_device(value):
    return global_cpu_device

def get_global_disk_device(value):
    return global_disk_device

@dataclass(frozen=True)
class ExecutionEnv:
    gpu: Any = None
    cpu: Any = None
    disk: Any = None
    mixed: Any = None

    # @classmethod
    # def create(cls, offload_dir):
    #     # fix recursive import
    #     from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
    #     gpu = TorchDevice("cuda:0")
    #     cpu = TorchDevice("cpu")
    #     disk = TorchDisk(offload_dir)
    #     return cls(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    def close_copy_threads(self):
        self.disk.close_copy_threads()