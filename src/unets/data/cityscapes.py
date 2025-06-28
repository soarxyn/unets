from typing import Optional, TypedDict

import cv2
import lightning as L
import numpy as np
from albumentations.core.composition import TransformsSeqType
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes

cv2.setNumThreads(0)


# Based off https://github.com/albumentations-team/autoalbument/blob/c0b18955b0036c753866bedc02c8c2c1fff73ded/examples/cityscapes/dataset.py#L10
class CityscapesDataset(Cityscapes):
    classmap: dict[int, tuple] = {
        class_.train_id: (class_.id, class_.color)
        for class_ in Cityscapes.classes
        if class_.train_id not in (-1, 255)
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, mode="fine", target_type="semantic")

    def encode_segmask(self, mask: np.ndarray) -> np.ndarray:
        H, W = mask.shape[:2]
        segmask = np.zeros((H, W, len(self.classmap)), dtype=np.float32)

        for idx, (label, _) in self.classmap.items():
            segmask[:, :, idx] = (mask == label).astype(float)
        return segmask

    def __getitem__(self, index: int):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.targets[index][0], cv2.IMREAD_UNCHANGED)
        mask = self.encode_segmask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    H, W = mask.shape[:2]

    label_mask = np.argmax(mask, axis=-1)
    mask = np.zeros((H, W, 3), dtype=np.uint8)

    for idx, (_, color) in CityscapesDataset.classmap.items():
        mask[label_mask == idx] = np.array(color)

    image = image.reshape(H, W, 3)
    return cv2.addWeighted(image, 1, mask, alpha, 0)


class Transforms(TypedDict):
    train: TransformsSeqType
    eval: TransformsSeqType


class CityscapesDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        transforms: Transforms = {"train": [], "eval": []},
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_transforms = transforms["train"]
        self.eval_transforms = transforms["eval"]

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = CityscapesDataset(
                self.data_dir, split="train", transforms=self.train_transforms
            )

            self.val_dataset = CityscapesDataset(
                self.data_dir, split="val", transforms=self.eval_transforms
            )

        if stage == "test" or stage is None:
            self.test_dataset = CityscapesDataset(
                self.data_dir, split="test", transforms=self.eval_transforms
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )
