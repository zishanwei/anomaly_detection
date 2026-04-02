"""
PyTorch device selection: NVIDIA CUDA, Intel XPU (Arc / Data Center GPU with IPEX), or CPU.

Intel Arc: install Intel Extension for PyTorch matching your torch build, e.g.
  pip install intel-extension-for-pytorch
See https://intel.github.io/intel-extension-for-pytorch/
"""

from __future__ import annotations

import torch

from config import TORCH_DEVICE


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def get_torch_device() -> torch.device:
    """
    Resolve device from config TORCH_DEVICE:
      auto — cuda if available, else xpu (Intel), else cpu
      xpu  — force Intel XPU when available
      cuda — force CUDA when available
      cpu  — always CPU
    """
    pref = (TORCH_DEVICE or "auto").lower().strip()

    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if pref == "xpu":
        if _xpu_available():
            return torch.device("xpu")
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _xpu_available():
        return torch.device("xpu")
    return torch.device("cpu")


def get_hf_device_map() -> str:
    """String for Hugging Face Accelerate device_map (Chronos, etc.)."""
    d = get_torch_device()
    return d.type


def load_intel_extension_for_pytorch():
    """
    Import Intel Extension for PyTorch (IPEX). Required for many Intel GPU / XPU setups
    before loading models with device_map='xpu'.
    Returns the module, or None if not installed.
    """
    try:
        import intel_extension_for_pytorch as ipex

        return ipex
    except ImportError:
        return None


def describe_device() -> str:
    d = get_torch_device()
    if d.type == "xpu":
        try:
            name = torch.xpu.get_device_name(d.index or 0)
        except Exception:
            name = "Intel XPU"
        return f"{d} ({name})"
    if d.type == "cuda":
        try:
            name = torch.cuda.get_device_name(d)
        except Exception:
            name = "CUDA"
        return f"{d} ({name})"
    return str(d)
