# GPU Benchmark

Reproducible PyTorch/Accelerate training and GPU hardware measurements. Supports ResNet50, ViT-B/L, ConvNeXt-base/large, and DistilBERT on IMDb. Legacy LLM inference remains available through the single-benchmark command.

## Install

```bash
uv sync --frozen --extra fp8 --python 3.11
uv run --frozen --extra fp8 gpu-benchmark --help
uvx --python 3.11 --from '.[fp8]' gpu-benchmark-run --help
```

The optional `fp8` extra installs the matching torchao backend. Runtime commands also work from installed wheels outside the repository. Original root Python scripts remain wrappers for compatibility.

## Training matrix

```bash
uv run --frozen --extra fp8 gpu-benchmark-run \
  --gpu-counts 1 2 3 4 --precisions fp32 tf32 fp16 bf16 fp8 bf16-compiled \
  --vision_data /path/to/imagenette2 --vision_class_num 10 \
  --steps 100 --warmup-steps 10 --validation-steps 25 --repetitions 3 \
  --output-dir /path/to/results
```

Vision data requires separate `train/` and `val/` class directories. `--tasks vision` or `--tasks language` selects a workload; `--models` selects vision models. Precision comparisons use the same per-GPU batches (64 vision, 16 language); they are configurable. `no` remains an alias for FP32. `--synthetic` avoids datasets and pretrained weights. `--resume` requires the same source and configuration and retries incomplete cases.

FP32 disables TF32; TF32 permits it for matrix and convolution operations. FP8 initially supports ViT and DistilBERT and records the converted layer count. Compare FP8 with `bf16-compiled` to control for compilation. Other models explicitly skip FP8. Every case records measured-step throughput, latency, memory, process logs, and NVML telemetry. Warmup is excluded. The suite measures performance, not convergence.

## Cactus profiles

```bash
# Smoke first: synthetic data, all GPU counts and supported modes.
uv run --frozen --extra fp8 python -m gpu_benchmark.profile smoke --output /path/to/smoke
# Cache IMDb, tokenization and model weights; also downloads Imagenette-320.
uv run --frozen --extra fp8 python -m gpu_benchmark.prepare_data
# Full real-data run; use the original Imagenette dataset when available.
uv run --frozen --extra fp8 python -m gpu_benchmark.profile full \
  --data /path/to/imagenette2 --output /path/to/full
```

Profiles include GEMM (FP64/FP32/TF32/FP16/BF16/FP8), GPU and host transfer bandwidth, GPU-pair transfers, and NCCL all-reduce. The full profile calibrates a uniform step count between 20 and 100, performs a separate batch-capacity sweep, and enforces an eight-hour wall-clock budget. Preserve partial results when the budget expires.

**Cactus transport workaround:** profiles set `NCCL_P2P_DISABLE=1` because native NCCL P2P currently stalls with the GPUs in translated `DMA-FQ` IOMMU domains. NCCL results therefore characterize shared-memory transport. An administrator must resolve and validate the host configuration before collecting native P2P results. Ordinary matrix runs inherit the caller's NCCL settings. Manifests record those settings.

Run directories contain per-case JSON, logs, telemetry, dependency metadata and GPU topology. They are artifacts; existing historical results are unchanged. Hardware rates distinguish FLOP/s, memory read+write traffic, and transfer payload bytes/s. `gpu-benchmark-plot --csv_file /path/to/monitor.csv --output_dir /path/to/plots` plots telemetry.

```bash
uv run --frozen --extra fp8 pytest -q
```
