# GPU Benchmark

Efficiently evaluate your GPU's deep learning performance with GPU Benchmark, leveraging `pytorch` and `accelerate` for a broad range of models.

# Introduction

Maximize your GPU's potential in deep learning with this repository, providing comprehensive benchmarking scripts and performance insights.

# Features

- Utilize `accelerate` for deep learning training.
- Precision options: `fp32`, `fp16`, `bf16`.
- Scale across GPUs (1 to max available).
- Log key metrics:
    - CPU, RAM, GPU usage
    - GPU memory, temperature
- Plot system metrics.
- Log training/validation metrics.
- Support various models:
    - Vision: ResNet50, ViT-B-16-224, ViT-L-16-224, ConvNext-base, ConvNext-large
    - Language: [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)
    - LLM Inference: [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [sgugger/sharded-gpt-j-6B](https://huggingface.co/sgugger/sharded-gpt-j-6B)

# Files

- `train_vision_models.py`: Train/evaluate vision models.
- `train_language_models.py`: Train/evaluate language models.
- `llm_inference.py`: Perform LLM inference.
- `benchmark.py`: Launch benchmarking.
- `run.py`: Execute multiple benchmarks with varied settings.

# Usage

## Environment Setup

Ensure NVIDIA drivers, CUDA, and `conda` are installed.

### Create Conda Environment
```bash
conda env create -f environment.yml
```

For vision tasks, download ImageNet-like dataset, [imagenette](https://github.com/fastai/imagenette) - recommended

## Single Benchmark

Prepare accelerate config [default_config.yaml](default_config.yaml).

Execute
```bash
accelerate launch --config_file=<config_file> benchmark.py \
                  --model=<model name> \                      # only needed for vision task, see MODEL_NAMES
                  --epochs=<n epochs> \                       # default: 5
                  --batch_size=<batch_size> \                 # default: 32
                  --data=<path/to/data> \                     # only needed for vision task, path to dataset, should have train and val subfolders with class subfolders
                  --monitor_log=<path/to/monitor/csv/file> \  # default: system_usage_log.csv
                  --log=<path/to/log/file> \                  # default: log.log
                  --workers=<n workers> \                     # default: 16
                  --lr=<learning rate> \                      # 3e-4
                  --classes=<n classes>                       # only needed for vision task, number of classes in dataset
```

## Multiple Benchmarks

Example to run all vision models, language model and LLM inference with multiple GPUs and precisions.

```bash
python run.py --n_gpus=<number of GPUs> \
              --precisions=<list of precisions> \   # choices: no, fp16, bf16
              --n_epochs=<n epochs> \               # default: 5
              --n_workers=<n workers> \             # default: 16
              --vision_batch_size=<batch size> \    # default: 32
              --vision_lr=<lr> \                    # default: 3e-4
              --vision_class_num=<n classes> \      # n classes for vision tasks
              --vision_data=<path/to/dataset> \     # path to vision dataset
              --language_batch_size=<batch size> \  # default: 16
              --language_lr=<lr> \                  # default: 2e-5
```

Results of the benchmark, can be found in `benchmark_results/%Y-%m-%d_%H-%M-%S` folder with a timestamp

## Visualizing System Information

The script will create plots of system information:
- CPU usage, RAM usage
- GPU temperature, GPU usage, GPU memory usage per each GPU

```bash
python plot_benchmark.py --csv_file=<path/to/csv/file> --output_dir=<path/to/output/directory>
```
