"""Download/cache real benchmark inputs before timed runs."""

from pathlib import Path
import hashlib
import json
import subprocess
import tarfile
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import timm
from .run import VISION_MODELS


def main():
    root = Path("data")
    root.mkdir(exist_ok=True)
    archive = root / "imagenette2-320.tgz"
    if not (root / "imagenette2-320" / "val").is_dir():
        subprocess.run(
            [
                "curl",
                "-fL",
                "--retry",
                "2",
                "--max-time",
                "600",
                "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
                "-o",
                str(archive),
            ],
            check=True,
        )
        with tarfile.open(archive) as f:
            f.extractall(root, filter="data")
    target = root / "imdb-tokenized"
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    if not target.exists():
        dataset = load_dataset("imdb")
        dataset = dataset.map(
            lambda examples: tokenizer(
                examples["text"], padding="max_length", truncation=True
            ),
            batched=True,
        )
        dataset = dataset.remove_columns(["text"]).rename_column("label", "labels")
        dataset.save_to_disk(str(target))
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased"
    )
    del model
    for name in VISION_MODELS:
        model = timm.create_model(name, pretrained=True, num_classes=10)
        del model
    metadata = {
        "vision_archive_sha256": (
            hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
            if archive.exists()
            else None
        ),
        "tokenized_data": str(target.resolve()),
        "vision_data": str((root / "imagenette2-320").resolve()),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("All real datasets and weights cached", flush=True)


if __name__ == "__main__":
    main()
