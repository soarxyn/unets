import torch
import torch.nn as nn
import torch.nn.functional as F


class DICELoss(nn.Module):
    def __init__(self, softmax: bool = True):
        super().__init__()

        self.softmax = softmax

    def forward(
        self,
        logits,
        targets,
        eps=1e-6,
    ):
        probs = F.softmax(logits, dim=1) if self.softmax else logits
        targets = targets.float()

        intersection = torch.sum(probs * targets, (2, 3))
        union = torch.sum(probs + targets, (2, 3))

        dice_score = (2.0 * intersection + eps) / (union + eps)
        dice_loss = 1.0 - dice_score

        return dice_loss.mean()
