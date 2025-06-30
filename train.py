import torch
import os

import albumentations as A
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import CometLogger

from unets.data import CityscapesDataModule
from unets.models import UNetLitModel

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True
os.environ["TORCHDYNAMO_VERBOSE"] = "0"


def train_model():
    dm = CityscapesDataModule(
        data_dir="./data",
        batch_size=4,
        transforms={
            "train": [
                A.SmallestMaxSize(max_size=224 * 2, p=1.0),
                A.RandomCrop(height=224, width=224, p=1.0),
                A.SquareSymmetry(p=1.0),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.ToTensorV2(),
            ],
            "eval": [
                A.SmallestMaxSize(max_size=224 * 2, p=1.0),
                A.CenterCrop(height=224, width=224, p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.ToTensorV2(),
            ],
        },
    )

    model = UNetLitModel()

    logger = CometLogger(project="unet-cityscapes")

    checkpoint_callback = ModelCheckpoint(
        monitor="val/iou",
        mode="max",
        dirpath="checkpoints/",
        filename="unets-{epoch:02d}",
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        accumulate_grad_batches=32,
        max_epochs=100,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[
            RichProgressBar(),
            RichModelSummary(),
            LearningRateMonitor(logging_interval="step"),
            checkpoint_callback,
        ],
    )

    trainer.fit(model=model, datamodule=dm)

    logger.experiment.log_model("BestModel", checkpoint_callback.best_model_path)
    logger.experiment.end()


if __name__ == "__main__":
    train_model()
