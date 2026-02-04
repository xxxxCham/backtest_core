import sys
from pathlib import Path
import os
import traceback

# Setup python path like ui/app.py
ROOT_DIR = Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

print(f"Diagnostics: Root Dir: {ROOT_DIR}")
print(f"Diagnostics: Python Path: {sys.path[0]}")

print("\n--- Testing Imports from ui.context ---")
try:
    from ui import context
    print(f"BACKEND_AVAILABLE: {context.BACKEND_AVAILABLE}")
    print(f"IMPORT_ERROR: '{context.IMPORT_ERROR}'")

    if not context.BACKEND_AVAILABLE:
        print("Backend unavailable. Attempting individual imports to isolate error:")
        try:
            from backtest.engine import BacktestEngine
            print("  - backtest.engine imported OK")
        except Exception as e:
            print(f"  - backtest.engine FAILED: {e}")

        try:
            from backtest.storage import get_storage
            print("  - backtest.storage imported OK")
        except Exception as e:
            print(f"  - backtest.storage FAILED: {e}")

        try:
            from data.loader import load_ohlcv
            print("  - data.loader imported OK")
        except Exception as e:
            print(f"  - data.loader FAILED: {e}")

        try:
            from strategies.base import get_strategy
            print("  - strategies.base imported OK")
        except Exception as e:
            print(f"  - strategies.base FAILED: {e}")

except Exception as e:
    print(f"Critical error importing ui.context: {e}")
    traceback.print_exc()

print("\n--- Testing GPU Detection ---")
try:
    from performance.gpu import GPUDeviceManager, get_gpu_info

    manager = GPUDeviceManager()
    info = get_gpu_info()
    print("Imported GPUDeviceManager.")
    print(
        "GPU Available: "
        f"{info.get('gpu_available')} | CuPy: {info.get('cupy_available')} | "
        f"Numba: {info.get('numba_cuda_available')}"
    )

    if info.get("cupy_device") is not None:
        print(
            "Active Device: "
            f"{info.get('cupy_device')} - {info.get('cupy_device_name')} "
            f"({info.get('cupy_memory_total_gb', 0):.1f} GB)"
        )
        if "cupy_memory_free_gb" in info:
            print(f"Free Mem: {info['cupy_memory_free_gb']:.2f} GB")
        if "available_gpu_count" in info:
            print(f"Available GPU Count: {info['available_gpu_count']}")

    if info.get("cupy_available"):
        try:
            import cupy as cp
            print(f"CuPy imported: {cp.__version__}")
            count = cp.cuda.runtime.getDeviceCount()
            print(f"Device Count: {count}")
            for i in range(count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
                print(f"  Device {i}: {name}")
        except Exception as e:
            print(f"CuPy error: {e}")
    else:
        print("CuPy not installed or disabled.")

except Exception as e:
    print(f"GPU module import error: {e}")