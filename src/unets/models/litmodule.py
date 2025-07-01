from typing import Sequence

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torchmetrics as tm

from unets.data import overlay_mask
from unets.models import UNetModel
from unets.models.modules import DICELoss


def unnormalize_image(
    tensor: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
):
    tensor = tensor.clone()

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    tensor = torch.clamp(tensor, 0.0, 1.0)

    return tensor


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
            UNetModel(in_channels, out_channels, latent_channels, activation)
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

        loss = dice_loss + ce_loss

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        integer_masks = torch.argmax(one_hot_masks, dim=1)
        self.train_iou(logits, integer_masks)

        self.log("train/iou", self.train_iou, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, one_hot_masks = batch
        one_hot_masks = one_hot_masks.permute((0, 3, 1, 2))

        logits = self.forward(images)

        ce_loss = self.ce_criterion(logits, one_hot_masks.float())
        dice_loss = self.dice_criterion(logits, one_hot_masks)

        loss = dice_loss + ce_loss

        integer_masks = torch.argmax(one_hot_masks, dim=1)
        self.val_iou(logits, integer_masks)

        self.log("val/iou", self.val_iou, on_epoch=True)

        return {"loss": loss, "preds": logits}

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx != 0:
            return

        images, one_hot_masks = batch
        logits = outputs["preds"]

        image_to_log = images[0]
        true_mask_to_log = one_hot_masks[0]
        logits_to_log = logits[0]

        true_mask_int = torch.argmax(true_mask_to_log, dim=-1).cpu().numpy()
        pred_mask_int = torch.argmax(logits_to_log, dim=0).cpu().numpy()

        viewable_image = unnormalize_image(image_to_log).permute(1, 2, 0).cpu().numpy()
        viewable_image = (viewable_image * 255).astype(np.uint8)
        viewable_true_mask = overlay_mask(viewable_image, true_mask_int, 0.5)
        viewable_pred_mask = overlay_mask(viewable_image, pred_mask_int, 0.5)

        composite_image = np.hstack(
            [viewable_image, viewable_true_mask, viewable_pred_mask]
        )

        self.logger.experiment.log_image(
            composite_image,
            name=f"Validation Preview",
            step=self.current_epoch,  # Log against the current epoch
        )

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
