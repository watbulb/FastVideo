# Windows Installation

FastVideo runs on Windows with NVIDIA GPUs, but several upstream components
behave differently from Linux:

- **Triton** is not officially shipped for Windows; FastVideo installs the
  [`triton-windows`](https://github.com/woct0rdho/triton-windows) community
  port automatically for Windows installs.
- **NCCL** is not bundled with PyTorch on Windows. FastVideo automatically
  falls back to `gloo` for the single-process case; multi-GPU collectives
  are not supported by upstream PyTorch on Windows.
- **CUDA IPC** is not implemented in the Windows driver, so cross-process
  CUDA tensor sharing is unavailable. FastVideo's worker copies the result
  tensor to CPU before returning it.

If you need full multi-GPU or unconstrained Triton support, prefer WSL2.

## Prerequisites

- Python 3.10–3.12, 64-bit
- Visual Studio 2022 with the "Desktop development with C++" workload
  (older toolchains do not reliably compile Triton's runtime launchers)
- CUDA Toolkit 12.x
- Git for Windows

## Install

```powershell
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo

uv venv --python 3.12 --seed
.\.venv\Scripts\activate

uv pip install -e .

cd fastvideo-kernel
.\build.ps1
```

`build.ps1` enters a VS developer shell automatically, detects your GPU's
compute capability via `torch.cuda.get_device_capability()`, and forwards
the appropriate `CMAKE_CUDA_ARCHITECTURES` to scikit-build-core. Override
with `TORCH_CUDA_ARCH_LIST` or `CMAKE_ARGS` when needed.

## Quick verification

```powershell
python examples\inference\basic\basic_windows.py
```

This script picks the single-GPU, no-Triton, no-NCCL configuration that is
known to work on a clean Windows install. Treat it as the baseline before
trying anything more exotic.

## Attention backends on Windows

| Backend         | Status   | Notes                                                                                                |
|-----------------|----------|------------------------------------------------------------------------------------------------------|
| Torch SDPA      | Works    | Default fallback when nothing else loads                                                             |
| Flash-Attention | Untested | Upstream wheels are not provided                                                                     |
| SageAttention 2 | Works    | Prebuilt wheels: [sdbds/SageAttention-for-windows](https://github.com/sdbds/SageAttention-for-windows/releases) |
| SageAttention 3 | Unstable | sm_120 / NVFP4 ecosystem is too new in 2026                                                          |
| SLA / SageSLA   | Falls back | Triton import gates are missing on Windows                                                         |
