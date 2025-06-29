from lightning import Trainer
from lightning.pytorch.tuner.tuning import Tuner
from loguru import logger
from unets.data import CityscapesDataModule

import albumentations as A
from unets.models import UNetLitModel


def find_learning_rate():
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

    model = UNetLitModel(learning_rate=1e-7)

    trainer = Trainer(accelerator="gpu", devices=1, max_epochs=1)

    logger.info("Starting LR Finder.")
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=dm)
    print(lr_finder.results)
    fig = lr_finder.plot(suggest=True)
    fig.show()


if __name__ == "__main__":
    find_learning_rate()
