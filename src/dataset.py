import torch
import numpy as np
import os
import albumentations as A

from PIL import Image
from typing import Optional
from torch.utils.data import Dataset, DataLoader


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_names, transform: Optional[A.Compose] = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name + ".jpg")
        label_path = os.path.join(self.mask_dir, image_name + ".png")

        image = np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0
        label = np.array(Image.open(label_path).convert("L")).astype(np.float32) / 255.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"].unsqueeze(0)
            return image, label, image_name

        return image, label, image_name
    
    
def get_loaders(
    image_names: list[str],
    images_dir: str,
    labels_dir: str,
    batch_size: int = 2,
    train_transform: Optional[A.Compose] = None,
    val_test_transform: Optional[A.Compose] = None,
    seed: int = 42,
) -> DataLoader | DataLoader | DataLoader:
    """
    Create DataLoader objects for the training, validation, and test sets.

    Args:
        image_names (list): List of image names to include in the dataset.
        images_dir (str): Path to the directory containing the original images.
        labels_dir (str): Path to the directory containing the masks.
        batch_size (int): Batch size for the DataLoader objects.
        transform (Optional[A.Compose]): Transform to apply to the images and masks.
        seed (int): Seed for the random number generator.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Split the dataset into training, validation, and test sets
    generator = torch.Generator().manual_seed(seed)
    dataset = SegmentationDataset(images_dir, labels_dir, image_names)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.8, 0.1, 0.1], 
        generator=generator
    )
    
    # Apply the transforms to the datasets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform
    
    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader
        




        