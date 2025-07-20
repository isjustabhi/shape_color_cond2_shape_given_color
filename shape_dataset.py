import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

COLOR_CLASSES = ['red', 'green', 'blue']

class ShapeGivenColorDataset(Dataset):
    def __init__(self, folder_path):
        self.folder = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

        self.to_gray = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        self.to_rgb = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        path = os.path.join(self.folder, filename)

        img = Image.open(path).convert("RGB")
        gray_img = self.to_gray(img)

        # Extract color label from filename
        color = filename.split("_")[1]  # shape_color_index.png
        color_onehot = self.encode_color(color)

        return color_onehot, gray_img

    def encode_color(self, color):
        vec = [0, 0, 0]
        if color in COLOR_CLASSES:
            vec[COLOR_CLASSES.index(color)] = 1
        return torch.tensor(vec, dtype=torch.float)
