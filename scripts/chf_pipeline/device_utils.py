"""
Single source of truth for GPU/CPU device selection across the pipeline.
On a machine with a CUDA GPU (e.g. a PARAM Shakti GPU node), everything
using DEVICE below automatically runs on GPU; on CPU-only machines it falls
back transparently, so the same code runs unmodified in both places.
"""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    print(f"[device_utils] Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
else:
    print("[device_utils] No CUDA GPU found -- running on CPU", flush=True)
