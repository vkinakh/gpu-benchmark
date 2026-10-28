"""Bounded training and rank-aware measurement shared by workloads."""

import os
import json
import time
from pathlib import Path
from itertools import islice
import statistics
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed


def initialize():
    set_seed(int(os.environ.get("GPUBENCH_SEED", "42")))
    precision = os.environ.get("GPUBENCH_PRECISION", "fp32")
    tf32 = precision == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    if precision in ("fp8", "bf16-compiled"):
        from accelerate.utils import TorchDynamoPlugin

        kwargs = []
        if precision == "fp8":
            from accelerate.utils import AORecipeKwargs
            from torchao.float8 import Float8LinearConfig

            kwargs = [AORecipeKwargs(config=Float8LinearConfig(pad_inner_dim=True))]
        return Accelerator(
            kwargs_handlers=kwargs,
            dynamo_plugin=TorchDynamoPlugin(
                backend="inductor", use_regional_compilation=True
            ),
        )
    return Accelerator()


def validation_batches(loader):
    return islice(loader, int(os.environ.get("GPUBENCH_VALIDATION", str(len(loader)))))


def training_batches(loader, accelerator):
    warmup = int(os.environ.get("GPUBENCH_WARMUP", "0"))
    steps = int(os.environ.get("GPUBENCH_STEPS", str(len(loader))))
    elapsed, samples, latencies = 0.0, 0, []
    if not len(loader):
        raise ValueError("Empty training loader; reduce batch size")

    def repeat_batches():
        while True:
            yield from loader

    measured_start = None
    for i, batch in enumerate(islice(repeat_batches(), warmup + steps)):
        if i == warmup and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        if i == warmup:
            measured_start = start
        yield batch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - start
        if i >= warmup:
            elapsed += duration
            latencies.append(duration)
            samples += (
                len(batch["input_ids"]) if isinstance(batch, dict) else len(batch[0])
            )
    elapsed = time.perf_counter() - measured_start
    times = accelerator.gather(torch.tensor([elapsed], device=accelerator.device))
    counts = accelerator.gather(torch.tensor([samples], device=accelerator.device))
    if not latencies:
        raise RuntimeError(
            "No measured training steps; reduce warmup or provide more data"
        )
    row = dict(
        samples=int(counts.sum()),
        seconds=float(times.max()),
        steps=len(latencies),
        samples_per_second=float(counts.sum()) / float(times.max()),
        peak_memory_bytes=(
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
    )
    row["latency_p50_seconds"] = statistics.median(latencies)
    row["latency_p95_seconds"] = sorted(latencies)[
        min(len(latencies) - 1, int(0.95 * len(latencies)))
    ]
    row["fp8_layers"] = sum(
        "Float8" in type(m).__name__
        for model in accelerator._models
        for m in model.modules()
    )
    if os.environ.get("GPUBENCH_PRECISION") == "fp8" and not row["fp8_layers"]:
        raise RuntimeError("FP8 requested but no layers were converted")
    if accelerator.is_main_process and os.environ.get("GPUBENCH_METRICS"):
        Path(os.environ["GPUBENCH_METRICS"]).write_text(json.dumps(row, indent=2))


def synthetic():
    return os.environ.get("GPUBENCH_SYNTHETIC") == "1"


class TokenData(torch.utils.data.Dataset):
    def __len__(self):
        return 4096

    def __getitem__(self, index):
        gen = torch.Generator().manual_seed(index)
        return dict(
            input_ids=torch.randint(0, 30522, (512,), generator=gen),
            attention_mask=torch.ones(512, dtype=torch.long),
            labels=torch.tensor(index % 2),
        )
