import csv
import time
import psutil
import GPUtil
import multiprocessing
import argparse

from train_vision_models import train_vision_model
from train_language_models import train_language_model
from llm_inference import llm_inference


def monitor_resources(stop_event, csv_file: str = "system_usage_log.csv") -> None:
    # Automatically get the number of GPUs
    num_gpus = len(GPUtil.getGPUs())

    # Open the CSV file for writing
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(
            [
                "Timestamp",
                "RAM_Usage (%)",
                "CPU_Usage (%)",
                *["GPU{}_Usage (%)".format(i) for i in range(num_gpus)],
                *["GPU{}_Memory_Usage (%)".format(i) for i in range(num_gpus)],
                *["GPU{}_Temperature (C)".format(i) for i in range(num_gpus)],
            ]
        )

        # Record metrics until stop_event is set
        while not stop_event.is_set():
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Get RAM and CPU usage
            ram_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent(interval=1)

            # Get GPU usage details for each GPU
            gpu_usage = [gpu.load * 100 for gpu in GPUtil.getGPUs()]
            gpu_memory_usage = [gpu.memoryUtil * 100 for gpu in GPUtil.getGPUs()]
            gpu_temperature = [gpu.temperature for gpu in GPUtil.getGPUs()]

            # Write the metrics to the CSV
            writer.writerow(
                [
                    timestamp,
                    ram_usage,
                    cpu_usage,
                    *gpu_usage,
                    *gpu_memory_usage,
                    *gpu_temperature,
                ]
            )

            # Flush the contents to ensure they're written to the file
            file.flush()

            # Wait for 1 second before recording the next set of metrics
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, required=True, choices=["vision", "language", "llm"]
    )
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--monitor_log", type=str, default="system_usage_log.csv")
    parser.add_argument("--log", type=str, default="log.log")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--classes", type=int, default=0)
    cli_args = parser.parse_args()

    task = cli_args.task

    stop_event = multiprocessing.Event()
    monitor_process = multiprocessing.Process(
        target=monitor_resources, args=(stop_event, cli_args.monitor_log)
    )
    monitor_process.start()

    try:
        if task == "vision":
            if cli_args.model == "":
                raise ValueError("Model name not provided.")

            if cli_args.data == "":
                raise ValueError("Path to data not provided.")

            if cli_args.classes == 0:
                raise ValueError("Number of classes not provided.")

            train_vision_model(
                model_name=cli_args.model,
                num_epochs=cli_args.epochs,
                batch_size=cli_args.batch_size,
                path_to_data=cli_args.data,
                log_filename=cli_args.log,
                n_workers=cli_args.workers,
                lr=cli_args.lr,
                n_classes=cli_args.classes,
            )
        elif task == "language":
            train_language_model(
                num_epochs=cli_args.epochs,
                batch_size=cli_args.batch_size,
                log_filename=cli_args.log,
                n_workers=cli_args.workers,
                lr=cli_args.lr,
            )
        elif task == "llm":
            if cli_args.model == "":
                raise ValueError("Model name not provided.")

            llm_inference(
                model_name=cli_args.model,
                log_filename=cli_args.log,
            )
    finally:
        stop_event.set()
        monitor_process.join()
