import torch
import numpy as np
import os
import albumentations as A


from PIL import Image
from typing import Optional
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_names, transform: Optional[A.Compose] = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx] + ".jpg")
        mask_path = os.path.join(self.mask_dir, self.image_names[idx] + ".png")

        original_image = np.array(Image.open(image_path).convert("RGB")).astype(
            np.float32
        )
        original_mask = np.array(Image.open(mask_path).convert("RGB").convert("L")).astype(
            np.float32
        )
        
        if self.transform:
            transformed = self.transform(image=original_image, mask=original_mask)
            image, mask = torch.tensor(transformed["image"]), torch.Tensor(
                transformed["mask"]
            )
            image = image.permute(2, 0, 1)
            mask = mask.unsqueeze(0)
            original_image = torch.tensor(original_image).permute(2, 0, 1)
            original_mask = torch.tensor(original_mask).unsqueeze(0)
            return image, mask, original_image, original_mask
        
        original_image = torch.tensor(original_image).permute(2, 0, 1)
        original_mask = torch.tensor(original_mask).unsqueeze(0)

        return original_image, original_mask, image_path, original_image, original_mask
        




        