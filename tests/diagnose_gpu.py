                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    import sys
from pathlib import Path
import os

# Setup python path
ROOT_DIR = Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import cupy as cp
    print(f"CuPy Version: {cp.__version__}")

    dev_count = cp.cuda.runtime.getDeviceCount()
    print(f"Detected {dev_count} GPUs via CuPy/CUDA Runtime.")


    for i in range(dev_count):
        props = cp.cuda.runtime.getDeviceProperties(i)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]

        # Get memory info by switching
        try:
            with cp.cuda.Device(i):
                mem_info = cp.cuda.runtime.memGetInfo()
                free_gb = mem_info[0] / (1024**3)
                total_gb = mem_info[1] / (1024**3)
                pci_bus_id = props["pciBusID"]
                print(f"GPU {i}: {name} (Total: {total_gb:.2f} GB, Free: {free_gb:.2f} GB) PCI Bus: {pci_bus_id}")
        except Exception as e:
            print(f"GPU {i}: Error getting memory info: {e}")

    # Check Environment Variables
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"BACKTEST_GPU_ID: {os.environ.get('BACKTEST_GPU_ID')}")

except ImportError:
    print("CuPy not installed.")
except Exception as e:
    print(f"Error during GPU check: {e}")
