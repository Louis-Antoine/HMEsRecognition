import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TestingDataset(Dataset):
    def __init__(self):
        self.imgs_path = "testing_data/"
        file_list = glob.glob(self.imgs_path + "*")
        print(file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "\*.jpg"):
                self.data.append([img_path, class_name])
        print(self.data)
        self.class_map = {"0" : 0, "1": 1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "-":10, "+":11, "=":12, "times":13, "div": 14}
        self.img_dim = (45, 45)
        self.transformations = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1), transforms.RandomInvert(1) ,transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor(class_id)
        img_tensor = self.transformations(img_tensor)
        return img_tensor, class_id