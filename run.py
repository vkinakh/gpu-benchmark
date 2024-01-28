from typing import Dict
from pathlib import Path
import subprocess
import yaml
import copy
import argparse
import datetime
import threading

from train_vision_models import MODEL_NAMES as VISION_MODEL_NAMES
from llm_inference import LLM_MODEL_NAMES
from monitoring import setup_monitoring_logger, monitor_resources_logging


def setup_distributed_config(config: Dict, n_gpus: int) -> Dict:
    config["num_processes"] = n_gpus

    if n_gpus > 1:
        config["distributed_type"] = "MULTI_GPU"
    else:
        config["distributed_type"] = "NO"

    gpu_ids = ",".join([str(i) for i in range(n_gpus)])
    config["gpu_ids"] = gpu_ids

    return config


def run(args):
    precisions = args.precisions
    n_epochs = args.n_epochs
    vision_batch_size = args.vision_batch_size
    path_vision_data = args.vision_data
    vision_lr = args.vision_lr
    vision_class_num = args.vision_class_num
    language_batch_size = args.language_batch_size
    language_lr = args.language_lr
    n_workers = args.n_workers
    max_n_gpus = args.n_gpus
    update_time = args.update_time

    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_out = Path(f"benchmark_results/{datetime_str}")
    path_out.mkdir(exist_ok=True, parents=True)

    # save args
    with open(path_out / "args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    with open("default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # vision tasks
    for n_gpus in range(1, max_n_gpus + 1):
        config_copy = copy.deepcopy(config)
        config_copy = setup_distributed_config(config_copy, n_gpus)

        for model in VISION_MODEL_NAMES:
            for precision in precisions:
                config_copy["mixed_precision"] = precision
                # save config
                with open("temp_config.yaml", "w") as f:
                    yaml.dump(config_copy, f)

                path_log = (
                    path_out
                    / f"vision_{model}_n_gpus_{n_gpus}_precision_{precision}.log"
                )
                path_monitor = (
                    path_out
                    / f"vision_{model}_n_gpus_{n_gpus}_precision_{precision}_monitor.csv"
                )

                curr_batch_size = (
                    vision_batch_size if precision == "no" else vision_batch_size * 2
                )

                # Initialize the logger
                logger = setup_monitoring_logger(str(path_monitor))
                stop_event = threading.Event()

                # start monitoring
                monitor_thread = threading.Thread(
                    target=monitor_resources_logging,
                    args=(stop_event, logger, update_time),
                )
                monitor_thread.start()

                # run
                subprocess.run(
                    [
                        "accelerate",
                        "launch",
                        "--config_file",
                        "temp_config.yaml",
                        "benchmark.py",
                        "--task",
                        "vision",
                        "--model",
                        model,
                        "--epochs",
                        str(n_epochs),
                        "--batch_size",
                        str(curr_batch_size),
                        "--data",
                        path_vision_data,
                        "--log",
                        str(path_log),
                        "--workers",
                        str(n_workers),
                        "--lr",
                        str(vision_lr),
                        "--classes",
                        str(vision_class_num),
                    ]
                )

                # stop monitoring
                stop_event.set()
                monitor_thread.join()

    # language tasks
    for n_gpus in range(1, max_n_gpus + 1):
        config_copy = copy.deepcopy(config)
        config_copy = setup_distributed_config(config_copy, n_gpus)

        for precision in precisions:
            config_copy["mixed_precision"] = precision
            # save config
            with open("temp_config.yaml", "w") as f:
                yaml.dump(config_copy, f)

            path_log = path_out / f"language_n_gpus_{n_gpus}_precision_{precision}.log"
            path_monitor = (
                path_out / f"language_n_gpus_{n_gpus}_precision_{precision}_monitor.csv"
            )

            curr_batch_size = (
                language_batch_size if precision == "no" else language_batch_size * 2
            )

            # Initialize the logger
            logger = setup_monitoring_logger(str(path_monitor))
            stop_event = threading.Event()

            # start monitoring
            monitor_thread = threading.Thread(
                target=monitor_resources_logging, args=(stop_event, logger, update_time)
            )
            monitor_thread.start()

            # run
            subprocess.run(
                [
                    "accelerate",
                    "launch",
                    "--config_file",
                    "temp_config.yaml",
                    "benchmark.py",
                    "--task",
                    "language",
                    "--epochs",
                    str(n_epochs),
                    "--batch_size",
                    str(curr_batch_size),
                    "--log",
                    str(path_log),
                    "--workers",
                    str(n_workers),
                    "--lr",
                    str(language_lr),
                ]
            )

            # stop monitoring
            stop_event.set()
            monitor_thread.join()

    # llm tasks
    for n_gpus in range(1, max_n_gpus + 1):
        for model in LLM_MODEL_NAMES:
            config_copy = copy.deepcopy(config)
            config_copy = setup_distributed_config(config_copy, n_gpus)

            # save config
            with open("temp_config.yaml", "w") as f:
                yaml.dump(config_copy, f)

            path_log = path_out / f"llm_{model.replace('/', '-')}_n_gpus_{n_gpus}.log"
            path_monitor = (
                path_out / f"llm_{model.replace('/', '-')}_n_gpus_{n_gpus}_monitor.csv"
            )

            # Initialize the logger
            logger = setup_monitoring_logger(str(path_monitor))
            stop_event = threading.Event()

            # start monitoring
            monitor_thread = threading.Thread(
                target=monitor_resources_logging, args=(stop_event, logger, update_time)
            )
            monitor_thread.start()

            # run
            subprocess.run(
                [
                    "accelerate",
                    "launch",
                    "--config_file",
                    "temp_config.yaml",
                    "benchmark.py",
                    "--task",
                    "llm",
                    "--model",
                    model,
                    "--log",
                    str(path_log),
                ]
            )

            # stop monitoring
            stop_event.set()
            monitor_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--precisions", nargs="+", default=["no", "fp16", "bf16"])
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--n_workers", type=int, default=16)
    parser.add_argument("--vision_batch_size", type=int, default=32)
    parser.add_argument("--vision_lr", type=float, default=3e-4)
    parser.add_argument("--vision_class_num", type=int, required=True)
    parser.add_argument(
        "--vision_data", type=str, required=True, help="Path to vision data"
    )
    parser.add_argument("--language_batch_size", type=int, default=16)
    parser.add_argument("--language_lr", type=float, default=2e-5)
    parser.add_argument("--update_time", type=float, default=0.1)
    cli_args = parser.parse_args()
    run(cli_args)
