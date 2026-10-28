"""Dense compute, transfer and NCCL probes with synchronized CUDA-event timing."""

import argparse
import json
import os
from pathlib import Path
import statistics
import torch
import torch.distributed as dist


def measure(operation, warmup=10, steps=100):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    timings = []
    for _ in range(steps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        a.record()
        operation()
        b.record()
        b.synchronize()
        timings.append(a.elapsed_time(b) / 1000)
    return dict(
        seconds=statistics.median(timings),
        p95_seconds=sorted(timings)[min(len(timings) - 1, int(0.95 * len(timings)))],
    )


def run(args):
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    rows = []

    def record(kind, operation, units, **metadata):
        timing = measure(operation, args.warmup, args.steps)
        row = dict(kind=kind, **metadata, **timing)
        row["rate"] = units / timing["seconds"]
        rows.append(row)

    if args.communication:
        dist.init_process_group("nccl")
        try:
            for size in args.transfer_mib:
                tensor = torch.ones(size * 1024 * 1024 // 4, device=device)

                def collective():
                    tensor.fill_(1)
                    dist.all_reduce(tensor)

                record(
                    "all_reduce",
                    collective,
                    tensor.numel() * 4,
                    mib=size,
                    gpus=dist.get_world_size(),
                    unit="bytes/s",
                    includes_buffer_reset=True,
                )
                if not torch.all(tensor == dist.get_world_size()):
                    raise RuntimeError("Collective correctness failure")
        finally:
            dist.destroy_process_group()
    else:
        for mode in ["fp64", "fp32", "tf32", "fp16", "bf16", "fp8"]:
            torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
            dtype = {
                "fp64": torch.float64,
                "fp32": torch.float32,
                "tf32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp8": torch.float8_e4m3fn,
            }[mode]
            for n in args.sizes:
                try:
                    a = torch.randn((n, n), device=device, dtype=torch.float32) * 0.1
                    b = torch.randn_like(a) * 0.1
                    ref = a @ b
                    if mode == "fp8":
                        a = a.to(dtype)
                        b = b.T.contiguous().to(dtype).T
                        scale = torch.ones((), device=device)
                        operation = lambda: torch._scaled_mm(
                            a, b, scale_a=scale, scale_b=scale, out_dtype=torch.float32
                        )
                    else:
                        a = a.to(dtype)
                        b = b.to(dtype)
                        operation = lambda: a @ b
                    output = operation()
                    error = float((output.float() - ref).norm() / ref.norm())
                    if not torch.isfinite(output).all() or error > 0.15:
                        raise RuntimeError(f"Numerical error {error}")
                    record(
                        "gemm",
                        operation,
                        2 * n**3,
                        precision=mode,
                        size=n,
                        unit="FLOP/s",
                        relative_error=error,
                    )
                except (RuntimeError, NotImplementedError) as exc:
                    if mode != "fp8":
                        raise
                    rows.append(
                        dict(
                            kind="gemm",
                            precision=mode,
                            size=n,
                            status="unsupported",
                            reason=str(exc),
                        )
                    )
        for size in args.transfer_mib:
            count = size * 1024 * 1024 // 4
            a = torch.ones(count, device=device)
            b = torch.empty_like(a)
            host = torch.empty(count, pin_memory=True)
            record(
                "device_copy",
                lambda: b.copy_(a),
                count * 8,
                mib=size,
                unit="bytes/s",
                traffic="read+write",
            )
            record(
                "device_to_host",
                lambda: host.copy_(a, non_blocking=True),
                count * 4,
                mib=size,
                unit="bytes/s",
            )
            record(
                "host_to_device",
                lambda: a.copy_(host, non_blocking=True),
                count * 4,
                mib=size,
                unit="bytes/s",
            )
            for other in range(torch.cuda.device_count()):
                if other == rank:
                    continue
                target = torch.empty_like(a, device=f"cuda:{other}")

                def peer_copy():
                    target.copy_(a)
                    torch.cuda.synchronize(other)

                record(
                    "gpu_pair_copy",
                    peer_copy,
                    count * 4,
                    mib=size,
                    source=rank,
                    target=other,
                    unit="bytes/s",
                )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not args.communication or rank == 0:
        path.write_text(json.dumps(rows, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--communication", action="store_true")
    p.add_argument("--sizes", nargs="+", type=int, default=[1024, 4096, 8192])
    p.add_argument("--transfer-mib", nargs="+", type=int, default=[1, 16, 256])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--output", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
