# -*- coding: utf-8 -*-
# @Time    : 2024/11/4 19:39
# @Author  : lil louis
# @Location: Beijing
# @File    : dataset.py


from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch

class OSADataset(Dataset):
    def __init__(self, oral_root, cranio_root, id_map,
                 cranio_transform=None, oral_transform=None, indices=None):
        self.oral_root = oral_root
        self.cranio_root = cranio_root
        self.id_map = id_map
        self.indices = indices
        self.cranio_transform = cranio_transform
        self.oral_transform = oral_transform

        self.cranio_files = [f for f in os.listdir(cranio_root)]
        self.oral_files = [f for f in os.listdir(oral_root)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        oral_front_path = os.path.join(self.oral_root, self.oral_files[idx * 2])
        oral_profile_path = os.path.join(self.oral_root, self.oral_files[idx * 2 + 1])
        cranio_front_path = os.path.join(self.cranio_root, self.cranio_files[idx * 2])
        cranio_profile_path = os.path.join(self.cranio_root, self.cranio_files[idx * 2 + 1])

        oral_front = np.array(Image.open(oral_front_path).convert("RGB"))
        oral_profile = np.array(Image.open(oral_profile_path).convert("RGB"))
        cranio_front = np.array(Image.open(cranio_front_path).convert("RGB"))
        cranio_profile = np.array(Image.open(cranio_profile_path).convert("RGB"))

        file_info = self.cranio_files[idx * 2].split("_")
        patient_id = int(file_info[0])
        gender = int(file_info[1])
        age = int(file_info[2])
        neck_circumference = int(file_info[3])
        bmi = int(file_info[4])

        front_area = self.id_map[patient_id]['front_area']
        front_length = self.id_map[patient_id]['front_length']
        front_width = self.id_map[patient_id]['front_width']
        profile_area = self.id_map[patient_id]['profile_area']
        profile_length = self.id_map[patient_id]['profile_length']
        profile_thickness = self.id_map[patient_id]['profile_thickness']
        profile_curvature = self.id_map[patient_id]['profile_curvature']

        label = self.id_map[patient_id]['label']
        if label == "正常":
            label = 0
        if label == "轻度":
            label = 0
        if label == "中度":
            label = 1
        if label == "重度":
            label = 1
        label = torch.tensor(label, dtype=torch.long)
        physio_dim = torch.tensor([gender, age, neck_circumference, bmi], dtype=torch.float)
        oral_dim = torch.tensor([front_area, front_length, front_width,
                                 profile_area, profile_length, profile_thickness, profile_curvature],
                                dtype=torch.float)

        if self.oral_transform is not None:
            augmented_front = self.oral_transform(image=np.array(oral_front))
            oral_front = augmented_front["image"]
            augmented_profile = self.oral_transform(image=np.array(oral_profile))
            oral_profile = augmented_profile["image"]

        if self.cranio_transform is not None:
            augmented_front = self.cranio_transform(image=np.array(cranio_front))
            cranio_front = augmented_front["image"]
            augmented_profile = self.cranio_transform(image=np.array(cranio_profile))
            cranio_profile = augmented_profile["image"]

        return (cranio_front, cranio_profile, oral_front, oral_profile), (physio_dim, oral_dim, label)
