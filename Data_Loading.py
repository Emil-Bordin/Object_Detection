import torch
import torchvision as torchvision
from torch.utils.data import random_split, DataLoader

from Dataset_Class import ArthropodDataset, transform  # updated the import of the class for the dataset and the transform function

def collate_fn(batch):

    images = list()
    targets = list()
    labels = list()

    for i,t,l, in batch:
        images.append(i)
        targets.append(t)
        labels.append(l)

    images = torch.stack([torchvision.transforms.functional.to_tensor(item) for item in images], dim=0)

    return images, targets, labels


class DataLoading:
    def __init__(self):
        self.batch_size = 8
        self.num_workers = 0


    def load_data(self):

        csv_file = r"/home/jovyan/data/Annotations1-8.csv"
        img_path = r"/home/jovyan/data/ArthroImages"

        dataset = ArthropodDataset(csv_file, img_path, transform=transform)

        train_dataset_length = round(0.1 * len(dataset))  # updated the dataset: previous -> train_set
        testlength = round(0.9 * len(dataset))  # updated the dataset: previous -> train_set
        train_dataset, test_set = random_split(dataset, [train_dataset_length, testlength])

        trainlength = round(0.9 * len(train_dataset))  # updated the dataset: previous -> train_set
        vallength = round(0.1 * len(train_dataset))  # updated the dataset: previous -> train_set
        train_set, val_set = random_split(train_dataset, [trainlength, vallength])

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)  # added the collate_fn
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)  # added the collate_fn

        return train_loader, val_loader, test_loader
