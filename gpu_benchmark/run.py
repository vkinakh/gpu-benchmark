"""Execute a resumable, bounded matrix of isolated benchmark processes."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading
from importlib.metadata import distributions
from .monitoring import setup_monitoring_logger, monitor_resources_logging

VISION_MODELS = [
    "resnet50",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
    "convnext_base",
    "convnext_large",
]


def execute(command, env, timeout, logfile):
    with open(logfile, "w") as output:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def matrix(args):
    models = [("vision", m) for m in args.models] if "vision" in args.tasks else []
    if "language" in args.tasks:
        models.append(("language", "distilbert-base-uncased"))
    # Round-robin repetitions ensure all combinations get a first measurement.
    return itertools.product(
        range(args.repetitions), args.gpu_counts, args.precisions, models
    )


def run(args):
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration.pop("resume", None)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        commit = os.environ.get("GPUBENCH_COMMIT", "installed-wheel")
    transport = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
    identity = hashlib.sha256(
        json.dumps([commit, configuration, transport], sort_keys=True).encode()
    ).hexdigest()
    manifest = out / "manifest.json"
    if manifest.exists():
        if not args.resume:
            raise ValueError("Output exists; pass --resume or choose a new directory")
        if json.loads(manifest.read_text())["identity"] != identity:
            raise ValueError("Resume configuration or source commit differs")
    else:
        manifest.write_text(
            json.dumps(
                dict(
                    identity=identity,
                    commit=commit,
                    configuration=configuration,
                    transport_environment={
                        k: v for k, v in os.environ.items() if k.startswith("NCCL_")
                    },
                    packages={d.metadata["Name"]: d.version for d in distributions()},
                ),
                indent=2,
            )
        )
        with (out / "gpu-topology.txt").open("w") as f:
            subprocess.run(["nvidia-smi", "topo", "-m"], stdout=f, stderr=f)
        with (out / "gpu-info.txt").open("w") as f:
            subprocess.run(["nvidia-smi", "-q"], stdout=f, stderr=f)
    deadline = time.monotonic() + args.budget_seconds
    failures = 0
    for repeat, count, precision, (task, model) in matrix(args):
        name = f"{task}_{model}_{count}gpu_{precision}_r{repeat}"
        folder = out / name
        folder.mkdir(exist_ok=True)
        result_path = folder / "result.json"
        if (
            args.resume
            and result_path.exists()
            and json.loads(result_path.read_text())["status"]
            in ["success", "unsupported"]
        ):
            continue
        if time.monotonic() >= deadline:
            raise SystemExit("Time budget exhausted; partial results retained")
        row = dict(
            task=task, model=model, gpus=count, precision=precision, repetition=repeat
        )
        if precision == "fp8" and task == "vision" and not model.startswith("vit_"):
            row.update(
                status="unsupported",
                reason="FP8 currently enabled for transformer workloads",
            )
            result_path.write_text(json.dumps(row, indent=2))
            continue
        if precision == "fp8":
            import importlib.util

            if importlib.util.find_spec("torchao") is None:
                raise RuntimeError("Install the fp8 extra to request FP8")
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=",".join(map(str, range(count))),
            GPUBENCH_WARMUP=str(args.warmup_steps),
            GPUBENCH_STEPS=str(args.steps),
            GPUBENCH_VALIDATION=str(args.validation_steps),
            GPUBENCH_SEED=str(args.seed),
            GPUBENCH_SYNTHETIC=str(int(args.synthetic)),
            GPUBENCH_METRICS=str(folder / "metrics.json"),
            GPUBENCH_PRECISION=precision,
            OMP_NUM_THREADS="4",
            TOKENIZERS_PARALLELISM="false",
        )
        config = dict(
            compute_environment="LOCAL_MACHINE",
            distributed_type="MULTI_GPU" if count > 1 else "NO",
            mixed_precision={"fp32": "no", "tf32": "no", "bf16-compiled": "bf16"}.get(
                precision, precision
            ),
            num_processes=count,
            num_machines=1,
            machine_rank=0,
            gpu_ids="all",
            use_cpu=False,
            rdzv_backend="static",
            same_network=True,
        )
        import yaml

        config_path = folder / "accelerate.yaml"
        config_path.write_text(yaml.safe_dump(config))
        command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(config_path),
            "--module",
            "gpu_benchmark.benchmark",
            "--task",
            task,
            "--epochs",
            "1",
            "--batch_size",
            str(
                args.vision_batch_size if task == "vision" else args.language_batch_size
            ),
            "--lr",
            str(args.vision_lr if task == "vision" else args.language_lr),
            "--workers",
            str(args.n_workers),
            "--log",
            str(folder / "training.log"),
        ]
        if task == "vision":
            command += [
                "--model",
                model,
                "--data",
                args.vision_data,
                "--classes",
                str(args.vision_class_num),
            ]
        stop = threading.Event()
        logger = setup_monitoring_logger(str(folder / "monitor.csv"))
        thread = threading.Thread(
            target=monitor_resources_logging,
            args=(stop, logger, args.update_time),
            daemon=True,
        )
        start = time.monotonic()
        thread.start()
        try:
            code = execute(
                command,
                env,
                min(args.case_timeout, deadline - start),
                folder / "process.log",
            )
            row.update(status="success" if code == 0 else "failed", exit_code=code)
            if code != 0:
                failures += 1
                if "out of memory" in (folder / "process.log").read_text().lower():
                    row["status"] = "oom"
            elif (folder / "metrics.json").exists():
                row.update(json.loads((folder / "metrics.json").read_text()))
            else:
                row.update(status="failed", reason="No measured training metrics")
                failures += 1
        except subprocess.TimeoutExpired:
            row.update(status="timeout")
            failures += 1
        finally:
            stop.set()
            thread.join(timeout=5)
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        row["wall_seconds"] = time.monotonic() - start
        result_path.write_text(json.dumps(row, indent=2))
        print(json.dumps(row), flush=True)
    if failures:
        raise SystemExit(f"{failures} cases failed; inspect results")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=["vision", "language"],
        default=["vision", "language"],
    )
    p.add_argument("--models", nargs="+", choices=VISION_MODELS, default=VISION_MODELS)
    p.add_argument("--gpu-counts", nargs="+", type=int, default=[1])
    p.add_argument(
        "--precisions",
        nargs="+",
        choices=["no", "fp32", "tf32", "fp16", "bf16", "fp8", "bf16-compiled"],
        default=["fp32", "fp16", "bf16"],
    )
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--validation-steps", type=int, default=25)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--vision_data", default="")
    p.add_argument("--vision_class_num", type=int, default=10)
    p.add_argument("--vision_lr", type=float, default=3e-4)
    p.add_argument("--language_lr", type=float, default=2e-5)
    p.add_argument("--vision_batch_size", type=int, default=64)
    p.add_argument("--language_batch_size", type=int, default=16)
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--update_time", type=float, default=1.0)
    p.add_argument("--budget-seconds", type=int, default=28800)
    p.add_argument("--case-timeout", type=int, default=900)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    if (
        min(a.gpu_counts) < 1
        or min(
            a.steps, a.validation_steps, a.repetitions, a.budget_seconds, a.case_timeout
        )
        < 1
        or a.warmup_steps < 0
    ):
        p.error("Invalid counts or durations")
    a.precisions = ["fp32" if x == "no" else x for x in a.precisions]
    if not a.synthetic and "vision" in a.tasks and not a.vision_data:
        p.error("--vision_data is required for real vision data")
    run(a)


if __name__ == "__main__":
    main()
