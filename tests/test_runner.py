import os
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
import pytest
from gpu_benchmark.run import execute, matrix


def test_matrix_round_robin():
    args=SimpleNamespace(models=['resnet50'],tasks=['vision','language'],repetitions=2,gpu_counts=[1,2,3,4],precisions=['fp32','bf16'])
    rows=list(matrix(args));assert len(rows)==32;assert all(r[0]==0 for r in rows[:16])


def test_exit_failure_is_propagated(tmp_path):
    assert execute([sys.executable,'-c','raise SystemExit(7)'],os.environ.copy(),10,tmp_path/'failure.log')==7


def test_timeout_terminates_process_group(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        execute([sys.executable,'-c','import time; time.sleep(60)'],os.environ.copy(),.2,tmp_path/'timeout.log')


def test_configuration_is_installed():
    from importlib.resources import files
    assert files('gpu_benchmark').joinpath('default_config.yaml').is_file()


def test_training_regressions():
    # Ensure a second epoch actually performs training rather than evaluation.
    import ast, inspect
    from gpu_benchmark.train_language_models import train_language_model
    tree=ast.parse(inspect.getsource(train_language_model))
    epochs=[n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch']
    assert any('model.train()' in ast.unparse(n) for n in epochs)
