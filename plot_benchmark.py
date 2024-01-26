import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def plot_benchmark(csv_file: str, out_dir: str) -> None:
    data = pd.read_csv(csv_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # Convert the 'Timestamp' column to datetime format
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])

    # Set the timestamp as the index
    data.set_index("Timestamp", inplace=True)

    # Plotting each category
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 18), sharex=True)

    # Formatting the date on the x-axis
    date_form = DateFormatter("%H:%M:%S")
    axes[0].xaxis.set_major_formatter(date_form)

    # RAM Usage
    axes[0].plot(data.index, data["RAM_Usage (%)"], label="RAM Usage (%)", color="blue")
    axes[0].set_ylabel("RAM Usage (%)")
    axes[0].legend(loc="upper right")

    # CPU Usage
    axes[1].plot(data.index, data["CPU_Usage (%)"], label="CPU Usage (%)", color="red")
    axes[1].set_ylabel("CPU Usage (%)")
    axes[1].legend(loc="upper right")

    # Adjust layout for the first figure
    plt.tight_layout()
    plt.savefig(out_dir / "system_usage.jpg", dpi=600)

    # Determine the number of GPUs
    gpu_columns = [col for col in data.columns if "GPU" in col]
    num_gpus = len(
        set(col.split("_")[0] for col in gpu_columns)
    )  # number of unique GPUs

    # Create separate figures for GPU metrics
    for i in range(num_gpus):
        gpu_usage_col = f"GPU{i}_Usage (%)"
        gpu_memory_col = f"GPU{i}_Memory_Usage (%)"
        gpu_temp_col = f"GPU{i}_Temperature (C)"

        fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # GPU Usage
        ax[0].plot(data.index, data[gpu_usage_col], label=gpu_usage_col, color="green")
        ax[0].set_ylabel(gpu_usage_col)
        ax[0].legend(loc="upper right")
        ax[0].xaxis.set_major_formatter(date_form)

        # GPU Memory Usage
        ax[1].plot(
            data.index, data[gpu_memory_col], label=gpu_memory_col, color="purple"
        )
        ax[1].set_ylabel(gpu_memory_col)
        ax[1].legend(loc="upper right")
        ax[1].xaxis.set_major_formatter(date_form)

        # GPU Temperature
        ax[2].plot(data.index, data[gpu_temp_col], label=gpu_temp_col, color="orange")
        ax[2].set_ylabel(gpu_temp_col)
        ax[2].legend(loc="upper right")
        ax[2].xaxis.set_major_formatter(date_form)

        plt.tight_layout()
        plt.savefig(out_dir / f"gpu{i}_usage.jpg", dpi=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    cli_args = parser.parse_args()

    plot_benchmark(cli_args.csv_file, cli_args.out_prefix)
