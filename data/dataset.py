import os
import glob
import random

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class GetTrainingPairs(Dataset):
    def __init__(self, raw_path, reference_path, raw_edge, reference_edge, transforms_=None, edge_transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.edge_transforms = transforms.Compose(edge_transforms_)
        self.filesA = sorted(glob.glob(raw_path + "/*.*"))
        self.filesB = sorted(glob.glob(reference_path + "/*.*"))
        self.filesC = sorted(glob.glob(raw_edge + "/*.*"))
        self.filesD = sorted(glob.glob(reference_edge + "/*.*"))
        self.len = min(len(self.filesA), len(self.filesB))

    def __getitem__(self, index):
        img_A = Image.open(self.filesA[index % self.len]).convert("RGB")
        img_B = Image.open(self.filesB[index % self.len]).convert("RGB")
        img_C = Image.open(self.filesC[index % self.len]).convert("L")
        img_D = Image.open(self.filesD[index % self.len]).convert("L")
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
            img_C = Image.fromarray(np.array(img_C)[:, ::-1], "L")
            img_D = Image.fromarray(np.array(img_D)[:, ::-1], "L")
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        img_C = self.edge_transforms(img_C)
        img_D = self.edge_transforms(img_D)
        return {"A": img_A, "B": img_B, 'C': img_C, 'D': img_D}

    def __len__(self):
        return self.len


class GetTestImage(Dataset):
    def __init__(self, raw_path, raw_edge, transforms_=None, edge_transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.edge_transforms=transforms.Compose(edge_transforms_)
        self.raw_images = sorted(glob.glob(raw_path + "/*.*"))
        self.raw_edges = sorted(glob.glob(raw_edge + "/*.*"))
        self.len = len(self.raw_images)

    def __getitem__(self, index):
        img_test = Image.open(self.raw_images[index % self.len])
        edge_test = Image.open(self.raw_edges[index % self.len])
        img_test = self.transform(img_test)
        edge_test = self.edge_transforms(edge_test)
        name = self.raw_images[index % self.len].split("\\")[-1]
        return {"test": img_test,'edge':edge_test, "name": name}

    def __len__(self):
        return self.len
