"""Hardwre reporting"""

from __future__ import annotations

import platform
import subprocess as sp
import logging
from typing import Optional

import torch
from torch import device

try:
    import resource
except ImportError:  # Windows does not provide the "resource" module
    resource = None

SYSTEM = platform.system()
logger = logging.getLogger(__name__)


def get_cuda_driver_version() -> Optional[str]:
    """Get the CUDA driver version via modinfo if possible.

    This is for Linux only.

    :returns: driver version or None
    """

    # Alternative
    # result = sp.run(["/usr/bin/nvidia-smi"], shell=False, capture_output=True)
    # if "Driver Version:" in str_line:
    #    version = str_line.split()[5]

    try:
        result = sp.run(["/sbin/modinfo", "nvidia"], shell=False, capture_output=True)
    except Exception:
        return

    for line in result.stdout.splitlines():
        str_line = line.decode()

        if str_line.startswith("version:"):
            cuda_driver_version = str_line.split()[1]

            return cuda_driver_version


def get_mac_hardware_name():
    try:
        result = sp.run(["sysctl", "-n", "machdep.cpu.brand_string"], shell=False, capture_output=True)
        return result.stdout.decode().strip()
    except Exception:
        return None


def get_accelerator() -> tuple[str, str]:
    """
    Return accelerator name and runtime version

    :returns: accelerator name and runtime version
    """

    # CUDA and ROCm both use torch.cuda
    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None):
            return "ROCm", torch.version.hip

        return "CUDA", torch.version.cuda

    # Intel XPU
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
        return "XPU", torch.version.xpu

    # Apple Metal
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return "MPS", None

    return None, None


def report_hardware(actual_device: device):
    if SYSTEM == "Darwin":
        chip_name = get_mac_hardware_name()

        if actual_device.type == "mps" and torch.backends.mps.is_available():
            msg = f"Using GPU device {actual_device}" + f" on {chip_name}" if chip_name else ""
        else:
            msg = f"Using CPU" + f" {chip_name}" if chip_name else ""

        logger.info(msg)
    elif (
        hasattr(torch, actual_device.type) and actual_device.type != "cpu"
    ):  # "cuda" (incl. ROCm) and "xpu"
        gpu = getattr(torch, actual_device.type)

        current_device = gpu.current_device()
        device_name = gpu.get_device_name(current_device)
        logger.info(f"Using GPU device:{current_device} {device_name}")

        free_memory, total_memory = gpu.mem_get_info()
        logger.info(
            f"GPU memory: {free_memory // 1024 ** 2} MiB free, "
            f"{total_memory // 1024 ** 2} MiB total"
        )
    else:
        logger.info(f"Using CPU {platform.processor()}")


def report_resource_usage(actual_device: device):
    if SYSTEM != "Windows" and resource is not None:
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mem = 0

        if SYSTEM == "Linux":
            peak_mem = maxrss / 1024
        elif SYSTEM == "Darwin":  # MacOSX
            peak_mem = maxrss / 1024**2

        if peak_mem:
            logger.info(f"Peak main memory usage: {peak_mem:.3f} MiB")

    if (
        hasattr(torch, actual_device.type)
        and actual_device.type != "mps"
        and actual_device.type != "cpu"
    ):
        gpu = getattr(torch, actual_device.type)
        peak_mem_gpu = gpu.max_memory_reserved() / 1024**2
        logger.info(f"Peak reserved GPU memory usage: {peak_mem_gpu:.3f} MiB")
