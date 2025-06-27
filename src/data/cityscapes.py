import cv2
import lightning as L
from torchvision.datasets import Cityscapes


class CityscapesDataset(Cityscapes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, mode="fine", target_type="semantic")

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.targets[index][0], cv2.IMREAD_UNCHANGED)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        print(image)
        return image, mask
