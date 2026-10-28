"""Cactus smoke/full orchestration. All results remain outside source control."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile", choices=["smoke", "full"])
    p.add_argument("--output", required=True)
    p.add_argument("--data", default="data/imagenette2-320")
    a = p.parse_args()
    os.environ["NCCL_P2P_DISABLE"] = (
        "1"  # Cactus: native NCCL peer path stalls; use validated shared-memory transport.
    )
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    start = time.monotonic()
    deadline = start + (28800 if a.profile == "full" else 7200)

    def call(arguments, logfile):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Profile budget exhausted")
        from .run import execute

        result = execute(
            [python] + arguments, os.environ.copy(), remaining, out / logfile
        )
        if result:
            raise RuntimeError(f"{logfile}: exit {result}; inspect log")

    smoke = a.profile == "smoke"
    # Hardware first establishes single-device and interconnect correctness.
    for gpu in range(4):
        # Each process selects its device via LOCAL_RANK without creating a process group.
        env = os.environ.copy()
        env["LOCAL_RANK"] = str(gpu)
        from .run import execute

        code = execute(
            [
                python,
                "-m",
                "gpu_benchmark.hardware",
                "--sizes",
                *(["256"] if smoke else ["1024", "4096", "8192"]),
                "--steps",
                "3" if smoke else "100",
                "--warmup",
                "1" if smoke else "10",
                "--output",
                str(out / f"hardware-gpu{gpu}.json"),
            ],
            env,
            min(900, deadline - time.monotonic()),
            out / f"hardware-gpu{gpu}.log",
        )
        if code:
            raise RuntimeError(f"Hardware GPU {gpu} failed")
    for count in [2, 3, 4]:
        call(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={count}",
                "--module",
                "gpu_benchmark.hardware",
                "--communication",
                "--steps",
                "3" if smoke else "100",
                "--output",
                str(out / f"nccl-{count}.json"),
            ],
            f"nccl-{count}.log",
        )
    if not smoke:
        os.environ["GPUBENCH_TOKENIZED_DATA"] = str(
            Path("data/imdb-tokenized").resolve()
        )
        # Pilot actual models to choose one uniform measurement length.
        call(
            [
                "-m",
                "gpu_benchmark.run",
                "--gpu-counts",
                "1",
                "--precisions",
                "fp32",
                "--output-dir",
                str(out / "calibration"),
                "--vision_data",
                a.data,
                "--warmup-steps",
                "2",
                "--steps",
                "10",
                "--validation-steps",
                "1",
                "--repetitions",
                "1",
            ],
            "calibration.log",
        )
        pilot = [
            json.loads(x.read_text())
            for x in (out / "calibration").glob("*/result.json")
        ]
        maximum = max(r["seconds"] / r["steps"] for r in pilot)
        steps = max(20, min(100, int((25200 - 360 * 15 - 3600) / (360 * maximum * 2))))
        (out / "calibration-choice.json").write_text(
            json.dumps({"steps": steps, "pilot_max_step_seconds": maximum})
        )
        # Separate per-mode capacity sweep; successful comparisons retain fixed batches.
        for task, model, base in [
            ("vision", "vit_large_patch16_224", 64),
            ("language", "distilbert-base-uncased", 16),
        ]:
            for precision in ["bf16", "fp8"]:
                batch = base
                while batch <= base * 32:
                    folder = out / f"sweep-{task}-{precision}-{batch}"
                    arguments = [
                        "-m",
                        "gpu_benchmark.run",
                        "--gpu-counts",
                        "1",
                        "--tasks",
                        task,
                        "--precisions",
                        precision,
                        "--steps",
                        "5",
                        "--warmup-steps",
                        "2",
                        "--validation-steps",
                        "1",
                        "--repetitions",
                        "1",
                        "--vision_data",
                        a.data,
                        "--output-dir",
                        str(folder),
                        "--vision_batch_size",
                        str(batch if task == "vision" else 64),
                        "--language_batch_size",
                        str(batch if task == "language" else 16),
                    ]
                    if task == "vision":
                        arguments += ["--models", model]
                    try:
                        call(arguments, f"sweep-{task}-{precision}-{batch}.log")
                    except RuntimeError:
                        records = [
                            json.loads(x.read_text())
                            for x in folder.glob("*/result.json")
                        ]
                        if records and all(r["status"] == "oom" for r in records):
                            break
                        raise
                    result = json.loads(next(folder.glob("*/result.json")).read_text())
                    import torch

                    if (
                        result["peak_memory_bytes"]
                        >= 0.8 * torch.cuda.get_device_properties(0).total_memory
                    ):
                        break
                    batch *= 2
    common = [
        "-m",
        "gpu_benchmark.run",
        "--gpu-counts",
        "1",
        "2",
        "3",
        "4",
        "--output-dir",
        str(out / "training"),
        "--precisions",
        "fp32",
        "tf32",
        "fp16",
        "bf16",
        "fp8",
        "bf16-compiled",
        "--budget-seconds",
        str(max(1, int(deadline - time.monotonic()))),
    ]
    if smoke:
        common += [
            "--models",
            "vit_base_patch16_224",
            "--steps",
            "2",
            "--warmup-steps",
            "1",
            "--validation-steps",
            "1",
            "--repetitions",
            "1",
            "--vision_batch_size",
            "2",
            "--language_batch_size",
            "2",
            "--synthetic",
            "--n_workers",
            "0",
            "--case-timeout",
            "900",
        ]
    else:
        common += [
            "--vision_data",
            a.data,
            "--steps",
            str(steps),
            "--case-timeout",
            "900",
        ]
    call(common, "training.log")
    summary = [
        json.loads(path.read_text())
        for path in (out / "training").glob("*/result.json")
    ]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Profile complete: {out}", flush=True)


if __name__ == "__main__":
    main()
