from typing import Sequence

import lightning as L
import torch
import torch.nn as nn
import torchmetrics as tm

from unets.models import UNetModel
from unets.models.modules import DICELoss


class UNetLitModel(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 19,
        latent_channels: Sequence[int] = [64, 128, 256, 512, 1024],
        activation: type[nn.Module] = nn.SiLU,
        learning_rate: float = 3.311311214825908e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.model = torch.compile(
            UNetModel(in_channels, out_channels, latent_channels, activation),
            mode="max-autotune",
        )

        self.ce_criterion = nn.CrossEntropyLoss()
        self.dice_criterion = DICELoss()

        self.train_iou = tm.JaccardIndex(task="multiclass", num_classes=out_channels)
        self.val_iou = tm.JaccardIndex(task="multiclass", num_classes=out_channels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, one_hot_masks = batch
        one_hot_masks = one_hot_masks.permute((0, 3, 1, 2))
        logits = self.forward(images)

        ce_loss = self.ce_criterion(logits, one_hot_masks.float())
        dice_loss = self.dice_criterion(logits, one_hot_masks)

        loss = ce_loss + dice_loss

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        integer_masks = torch.argmax(one_hot_masks, dim=1)
        self.train_iou(logits, integer_masks)

        self.log("train/iou", self.train_iou, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, one_hot_masks = batch
        logits = self.forward(images)

        ce_loss = self.ce_criterion(logits, one_hot_masks.float())
        dice_loss = self.dice_criterion(logits, one_hot_masks)

        loss = ce_loss + dice_loss

        integer_masks = torch.argmax(one_hot_masks, dim=1)
        self.val_iou(logits, integer_masks)

        self.log("val/iou", self.val_iou, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=int(self.trainer.estimated_stepping_batches),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
