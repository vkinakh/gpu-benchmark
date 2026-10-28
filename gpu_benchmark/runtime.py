"""Bounded training and rank-aware measurement shared by workloads."""
import os
import json
import time
from pathlib import Path
from itertools import islice
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed


def initialize():
    set_seed(int(os.environ.get('GPUBENCH_SEED', '42')))
    return Accelerator()


def validation_batches(loader):
    return islice(loader, int(os.environ.get('GPUBENCH_VALIDATION', str(len(loader)))))


def training_batches(loader, accelerator):
    warmup = int(os.environ.get('GPUBENCH_WARMUP', '0'))
    steps = int(os.environ.get('GPUBENCH_STEPS', str(len(loader))))
    elapsed, samples, latencies = 0., 0, []
    for i, batch in enumerate(islice(loader, warmup + steps)):
        if i == warmup and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        accelerator.wait_for_everyone()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start = time.perf_counter()
        yield batch
        if torch.cuda.is_available(): torch.cuda.synchronize()
        duration = time.perf_counter() - start
        if i >= warmup:
            elapsed += duration
            latencies.append(duration)
            samples += len(batch['input_ids']) if isinstance(batch, dict) else len(batch[0])
    times = accelerator.gather(torch.tensor([elapsed], device=accelerator.device))
    counts = accelerator.gather(torch.tensor([samples], device=accelerator.device))
    if not latencies: raise RuntimeError('No measured training steps; reduce warmup or provide more data')
    row = dict(samples=int(counts.sum()), seconds=float(times.max()), steps=len(latencies),
               samples_per_second=float(counts.sum())/float(times.max()),
               peak_memory_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
    if accelerator.is_main_process and os.environ.get('GPUBENCH_METRICS'):
        Path(os.environ['GPUBENCH_METRICS']).write_text(json.dumps(row, indent=2))


def synthetic():
    return os.environ.get('GPUBENCH_SYNTHETIC') == '1'


class TokenData(torch.utils.data.Dataset):
    def __len__(self): return 4096
    def __getitem__(self, index):
        gen = torch.Generator().manual_seed(index)
        return dict(input_ids=torch.randint(0, 30522, (512,), generator=gen),
                    attention_mask=torch.ones(512, dtype=torch.long), labels=torch.tensor(index % 2))
