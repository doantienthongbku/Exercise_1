import torch
import os
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np

class HeroImage(Dataset):
    def __init__(self, images_path='datasets/images', label_path='datasets/labels.csv', transform=None):
        self.images_path = images_path
        self.label_path = label_path
        self.transform = transform

        self.images = os.listdir(self.images_path)
        self.images.sort()
        self.labels = pd.read_csv(self.label_path)
        self.labels = self.labels.to_numpy()

        print(self.labels)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.images_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # make circle filter
        circle_filter = np.zeros_like(image)
        cv2.circle(circle_filter, (30, 30), 30, (255, 255, 255), -1)
        image = cv2.bitwise_and(image, circle_filter)
        
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx, 1]
        return image, label