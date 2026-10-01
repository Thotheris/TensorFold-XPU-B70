"""The B70 toolchain, memory limits, and checkpoint names are pinned."""

from __future__ import annotations

PCI_ID = "8086:e223"
KERNEL_MIN = (6, 17, 0)
COMPUTE_RUNTIME = "26.31.39395.13"
IGC = "2.40.13"
LEVEL_ZERO_MIN = (1, 32, 0)
TORCH_PIN = "2.14.1+xpu"
TORCH_SERIES_PREFIX = "2.14"
TRITON_SERIES = "3.8"
DLE = "2026.1"
DLE_PREFIX = "/opt/intel/dle-2026.1"
XPU_INDEX = "https://download.pytorch.org/whl/xpu"
PEAK_GBPS = 608.0
PEAK_BF16_TFLOPS = 183.0
HOST_RAM_WARN_GIB = 64
SWAP_OFFER_GIB = 32
MEM_AVAILABLE_FLOOR_KIB = 3 * 1024 * 1024
SHADOW_SIZES_GIB = (4, 8, 16)
MAX_ALLOC_BYTES = 4 * 1024**3
CHUNK_BYTES = 2 * 1024**3
MODELS = (
    "devan-carlin/Qwen3.8-27B-int4-AutoRound",
    "RedHatAI/Qwen3.8-27B-INT4",
    "letechlead/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-INT4-AutoRound",
    "SergiioB/Nemotron-3.5-Lightning-30B-A3B-GPTQ-INT4-G64-sym",
    "z-lab/Qwen3.8-27B-DFlash2",
)
