"""
src/models/device.py
Hardware accelerator detector for Apple Silicon M1 (MPS) and CUDA fallbacks.
"""

import torch

def get_compute_device() -> torch.device:
    """
    Selects the optimal hardware device:
    1. Apple Silicon GPU ('mps')
    2. NVIDIA GPU ('cuda')
    3. CPU fallback
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

if __name__ == "__main__":
    device = get_compute_device()
    print(f"Optimal compute device detected: {device}")
