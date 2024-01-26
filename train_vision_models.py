import time

from tqdm.auto import trange

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchmetrics import Accuracy
import timm
from accelerate import Accelerator
from accelerate.utils import compute_module_sizes

from utils import setup_logger


IMG_SIZE = 224
IMG_RESIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_NAMES = [
    "resnet50",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
    "convnext_base",
    "convnext_large",
]


def train_vision_model(
    model_name: str,
    num_epochs: int,
    batch_size: int,
    path_to_data: str,
    log_filename: str,
    n_workers: int = 16,
    lr: float = 3e-4,
    n_classes: int = 10,
) -> None:
    if model_name not in MODEL_NAMES:
        raise ValueError("Invalid model name.")

    logger = setup_logger(log_filename)

    start_time_model_logging = time.time()
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=n_classes,
    )
    model_loading_time = time.time() - start_time_model_logging
    logger.info(f"Model loading time: {model_loading_time:.3f} seconds")

    module_size = compute_module_sizes(model)[""]
    logger.info(f"Model size: {module_size / 1e6:.3f}M parameters")

    accelerator = Accelerator()

    trans_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    trans_val = transforms.Compose(
        [
            transforms.Resize(IMG_RESIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    ds_train = datasets.ImageFolder(path_to_data, transform=trans_train)
    ds_val = datasets.ImageFolder(path_to_data, transform=trans_val)

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        drop_last=True,
    )

    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model, opt, dl_train, dl_val = accelerator.prepare(model, opt, dl_train, dl_val)

    for epoch in trange(num_epochs, desc="Epoch"):
        start_time_training_epoch = time.time()
        model.train()
        for x, y in dl_train:
            opt.zero_grad()
            with accelerator.autocast():
                # Forward pass in autocast mode
                logits = model(x)
                loss = criterion(logits, y)
            # Backward pass in full precision
            accelerator.backward(loss)
            opt.step()

        training_epoch_time = time.time() - start_time_training_epoch
        logger.info(f"Epoch {epoch}. Training time: {training_epoch_time:.3f} seconds")

        start_time_eval_epoch = time.time()
        model.eval()
        total_loss = 0
        acc = Accuracy(task="multiclass", num_classes=n_classes).to(accelerator.device)
        for x, y in dl_val:
            with torch.inference_mode():
                logits = model(x)
                loss = criterion(logits, y)
            losses = accelerator.gather(loss)
            total_loss += torch.mean(losses)
            acc.update(logits, y)
        total_loss = torch.mean(accelerator.gather(total_loss))
        total_loss /= len(dl_val)
        logger.info(f"Epoch {epoch}. Loss: {total_loss}, acc: {acc.compute()}")
        acc.reset()
        eval_epoch_time = time.time() - start_time_eval_epoch
        logger.info(f"Epoch {epoch}. Evaluation time: {eval_epoch_time:.3f} seconds")
