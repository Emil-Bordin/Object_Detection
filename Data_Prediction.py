import albumentations as A
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class ArthropodDatasetPredict(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Ihre vorhandene Initialisierungslogik...
        self.root_dir = Path(root_dir)
        annotations = pd.read_csv(csv_file)
        self.image_list = annotations["image_name"].values
        self.boxes = [self.decodeString(item) for item in annotations["BoxesString"]]
        self.class_labels0 = [self.decodeString(item) for item in annotations["Classes"]]
        self.transform = transform

    def __len__(self):
        # Ihre vorhandene Logik zur Bestimmung der Länge...
        return len(self.image_list)

    def __getitem__(self, idx):
        imgp = str(self.root_dir / (self.image_list[idx] + ".jpg"))
        bboxes = torch.tensor(self.boxes[idx])
        class_labels = self.class_labels0[idx]
        img = cv2.imread(imgp)

        # Konvertierung in RGB, falls es ein Graustufenbild ist
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image_transformed = transformed["image"]
            bboxes = transformed["bboxes"]
            class_labels = transformed["class_labels"]

        if len(bboxes) > 0:
            bboxes = torch.stack([torch.tensor(item) for item in bboxes])
            class_labels = torch.stack([torch.tensor(item, dtype=torch.int64) for item in class_labels])
        else:
            bboxes = torch.zeros((0, 4))
            class_labels = torch.zeros((0, 4))

        return image_transformed, image, imgp, bboxes, class_labels

    def decodeString(self, BoxesString):
        # Ihre vorhandene Logik zur Dekodierung von Strings...
        if BoxesString == "no_box":
            return np.zeros((0, 4))
        else:
            try:
                boxes = np.array([np.array([int(i) for i in box.split(" ")])
                                  for box in BoxesString.split(";")])
                return boxes
            except:
                print(BoxesString)
                print("Submisssion is not well formatted")
                return np.zeros((0, 4))

# Vergessen Sie nicht, Ihre Transformationen zu definieren
transform = A.Compose([
    A.Resize(height=512, width=512, p=1),
    A.RandomBrightnessContrast(p=0.5)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_area=20))
