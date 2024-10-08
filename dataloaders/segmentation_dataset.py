import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, contiguous_ids_file, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # List the image names and turn them into a set of ids
        names_images = os.listdir(self.img_dir)
        self.image_ids = set()
        for name in names_images:
            # The images should be named like 1.png
            digit = name.split('.')[0]
            # Check the name of the file is a digit (if not don't use)
            if digit.isdigit():
                self.image_ids.add(int(digit))

        # Needed because the ids of our images are not conitguous while pytorch dataloader gives contiguous ids from dataset length
        self.contiguous_ids = np.load(contiguous_ids_file, allow_pickle=True)

        # Define transformations
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            #transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=(0, 0.05), contrast=(0, 0.05), saturation=(0, 0.05), hue=(0, 0.05)),]), p=0.3),
            transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.11)),]), p=0.2),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Imagenet weights
        ])



    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, dataloader_id):
        # [()] because it is a dic loaded in a np array...
        _id = self.contiguous_ids[()][dataloader_id]
        img_name = os.path.join(self.img_dir, str(_id) + ".png")
        image = Image.open(img_name).convert('RGB')  # Open the image and convert it to RGB
        seg_mask = np.load(os.path.join(self.mask_dir, str(_id) + ".npy"))

        seg_mask = torch.tensor(seg_mask)#, dtype=torch.long)
        image = self.transform(image)

        return image, seg_mask, _id


# Function to get data loaders for train, validation, and test splits
def get_data_loaders(img_dir, mask_dir, contiguous_ids_file, split_seed, batch_size=32, val_split=0.1, test_split=0.1):
    # Create dataset
    dataset = SegmentationDataset(img_dir, mask_dir, contiguous_ids_file)

    # Determine sizes for training, validation, and test sets
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * (dataset_size - test_size))
    train_size = dataset_size - test_size - val_size

    # Split the dataset
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size],
                                                generator=torch.Generator().manual_seed(split_seed))

    # Create DataLoader for each set
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
