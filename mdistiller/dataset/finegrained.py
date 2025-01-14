import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch

class ImageDataset(Dataset):
    def __init__(self, json_path, image_folder, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        file_name = list(self.data.keys())[idx]
        image_path = f"{file_name}.png"
        full_image_path = os.path.join(self.image_folder, image_path)

        label_value = int(file_name.split("_")[-2])
        label = torch.tensor(label_value, dtype=torch.long)

        aspect_logits = self.data[file_name]
        aspect_logits = torch.tensor(aspect_logits, dtype=torch.float32)

        if not os.path.exists(full_image_path):
            raise FileNotFoundError(f"Image file {full_image_path} not found.")

        image = Image.open(full_image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 데이터 반환
        return image, label, aspect_logits


class TestImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_paths = [
            os.path.join(image_folder, fname)
            for fname in os.listdir(image_folder)
            if fname.endswith(('png', 'jpg', 'jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        full_image_path = self.image_paths[idx]
        image = Image.open(full_image_path).convert('RGB')

        label_value = int(full_image_path.split("_")[-2])
        label = torch.tensor(label_value, dtype=torch.long)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, label
