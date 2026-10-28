import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
import pytest
from gpu_benchmark.run import execute, matrix


def test_matrix_round_robin():
    args = SimpleNamespace(
        models=["resnet50"],
        tasks=["vision", "language"],
        repetitions=2,
        gpu_counts=[1, 2, 3, 4],
        precisions=["fp32", "bf16"],
    )
    rows = list(matrix(args))
    assert len(rows) == 32
    assert all(r[0] == 0 for r in rows[:16])


def test_exit_failure_is_propagated(tmp_path):
    assert (
        execute(
            [sys.executable, "-c", "raise SystemExit(7)"],
            os.environ.copy(),
            10,
            tmp_path / "failure.log",
        )
        == 7
    )


def test_timeout_terminates_process_group(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        execute(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            os.environ.copy(),
            0.2,
            tmp_path / "timeout.log",
        )


def test_configuration_is_installed():
    from importlib.resources import files

    assert files("gpu_benchmark").joinpath("default_config.yaml").is_file()


def test_warmup_excluded_and_short_loader_repeated(tmp_path, monkeypatch):
    import json
    import torch
    from accelerate import Accelerator
    from gpu_benchmark.runtime import training_batches

    monkeypatch.setenv("GPUBENCH_WARMUP", "2")
    monkeypatch.setenv("GPUBENCH_STEPS", "3")
    monkeypatch.setenv("GPUBENCH_METRICS", str(tmp_path / "metrics.json"))
    loader = [(torch.ones(2, 3), torch.zeros(2))]
    seen = list(training_batches(loader, Accelerator(cpu=True)))
    result = json.loads((tmp_path / "metrics.json").read_text())
    assert len(seen) == 5
    assert result["steps"] == 3
    assert result["samples"] == 6
    assert result["samples_per_second"] > 0


def test_report_excludes_failures_and_uses_median(tmp_path):
    import json
    from gpu_benchmark.report import summarize

    for i, (status, rate) in enumerate(
        [("success", 10), ("success", 30), ("failed", 1000)]
    ):
        directory = tmp_path / str(i)
        directory.mkdir()
        (directory / "result.json").write_text(
            json.dumps(
                dict(
                    task="vision",
                    model="resnet50",
                    precision="fp32",
                    gpus=1,
                    status=status,
                    samples_per_second=rate,
                    peak_memory_bytes=100,
                )
            )
        )
    result = summarize(tmp_path)
    assert len(result) == 1
    assert result[0]["samples_per_second"] == 20
    assert result[0]["repetitions"] == 2
    assert result[0]["scaling_efficiency"] == 1
