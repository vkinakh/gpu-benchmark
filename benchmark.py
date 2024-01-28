import argparse

from train_vision_models import train_vision_model
from train_language_models import train_language_model
from llm_inference import llm_inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, required=True, choices=["vision", "language", "llm"]
    )
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--log", type=str, default="log.log")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--classes", type=int, default=0)
    cli_args = parser.parse_args()

    task = cli_args.task

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
