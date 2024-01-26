import time

from tqdm.auto import trange

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from accelerate import Accelerator
from accelerate.utils import compute_module_sizes

from utils import setup_logger


def train_language_model(
    num_epochs: int,
    batch_size: int,
    log_filename: str,
    n_workers: int = 16,
    lr: float = 2e-5,
) -> None:
    model_name = "distilbert-base-uncased"

    logger = setup_logger(log_filename)

    accelerator = Accelerator()

    start_time_data_loading = time.time()
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    imdb = load_dataset("imdb")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = imdb.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    data_loading_time = time.time() - start_time_data_loading
    logger.info(f"Data loading time: {data_loading_time:.3f} seconds")

    start_model_loading = time.time()
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    model_loading_time = time.time() - start_model_loading
    logger.info(f"Model loading time: {model_loading_time:.3f} seconds")

    module_size = compute_module_sizes(model)[""]
    logger.info(f"Model size: {module_size / 1e6:.3f}M parameters")

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    dl_train = DataLoader(
        tokenized_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
    )
    dl_test = DataLoader(
        tokenized_datasets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
    )

    # Prepare everything with Accelerate
    model, optimizer, dl_train, dl_test = accelerator.prepare(
        model,
        optimizer,
        dl_train,
        dl_test,
    )

    # Training loop
    model.train()
    for epoch in trange(num_epochs, desc="Epoch"):
        start_time_training_epoch = time.time()
        for batch in dl_train:
            optimizer.zero_grad()

            with accelerator.autocast():
                outputs = model(**batch)
                loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()

        training_epoch_time = time.time() - start_time_training_epoch
        logger.info(f"Epoch {epoch}. Training time: {training_epoch_time:.3f} seconds")

        start_time_eval = time.time()
        acc = Accuracy(task="multiclass", num_classes=2).to(accelerator.device)
        total_eval_loss = 0

        model.eval()
        for batch in dl_test:
            with torch.inference_mode():
                outputs = model(**batch)

            logits = outputs.logits
            losses = accelerator.gather(outputs.loss)
            total_eval_loss += torch.mean(losses)
            acc.update(logits, batch["labels"])

        total_eval_loss = torch.mean(accelerator.gather(total_eval_loss))
        total_eval_loss /= len(dl_test)
        logger.info(f"Epoch: {epoch}. Loss: {total_eval_loss}, acc: {acc.compute()}")
        acc.reset()
        eval_time = time.time() - start_time_eval
        logger.info(f"Epoch {epoch}. Evaluation time: {eval_time:.3f} seconds")
