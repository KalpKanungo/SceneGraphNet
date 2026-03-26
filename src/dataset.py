import json
import os
import cv2
import torch
from torch.utils.data import Dataset

class RelationshipDataset(Dataset):
    def __init__(self, image_dir, label_path):
        self.image_dir = image_dir

        with open(label_path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_path = os.path.join(self.image_dir, item["image"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = (image - 0.5) / 0.5

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        label = torch.tensor(item["label"], dtype=torch.long)

        return image, label