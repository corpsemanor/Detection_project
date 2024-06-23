import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = self.create_label_map()

    def create_label_map(self):
        labels = self.annotations['Finding Label'].unique()

        return {label: idx for idx, label in enumerate(labels)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        
        boxes = []
        labels = []
        
        image_id = self.annotations.iloc[idx, 0]
        img_annotations = self.annotations[self.annotations['Image Index'] == image_id]
        
        for _, row in img_annotations.iterrows():
            x_min = row['Bbox [x']
            y_min = row['y']
            x_max = x_min + row['w']
            y_max = y_min + row['h]']
            
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.label_map[row['Finding Label']])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transform:
            image = self.transform(image)

        return image, target
