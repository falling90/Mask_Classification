import platform
# import os
# import tqdm
from torch.utils.data import Dataset
# from PIL import Image

# from albumentations import *
# from albumentations.pytorch import ToTensorV2

# import numpy as np

# import random
# from collections import defaultdict
# from enum import Enum
# from typing import Tuple, List
# import torch
# from torch.utils.data import Dataset, Subset, random_split
# from torchvision import transforms
# from torchvision.transforms import *

class MyOS:
    @classmethod
    def path_join(cls, current_os, params):
        result = os.path.join(*params)
        if current_os == "Windows":
            result = result.replace("\\", "/")
        return result

class Data_Labeling():
    def __init__(self, image_name, gender, age):
        self.image_name = image_name
        self.gender = gender
        self.age = age
    
    @classmethod
    def get_mask_label(self):
        MASK = 0
        INCORRECT = 1
        NORMAL = 2

        if self.image_name.lower().startswith("mask"):
            return MASK
        elif self.image_name.lower().startswith("incorrect"):
            return INCORRECT
        elif self.image_name.lower().startswith("normal"):
            return NORMAL
        else:
            raise ValueError(f"Mask value should be either 'mask' or 'incorrect' or 'normal' // , {self.image_name}")

    def get_gender_label(self):
        MALE = 0
        FEMALE = 1

        if self.gender.lower() == "male":
            return MALE
        elif self.gender.lower() == "female":
            return FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {self.gender}")

    def get_age_label(self):
        YOUNG = 0
        MIDDLE = 1
        OLD = 2

        try:
            age = int(self.age)
        except Exception:
            raise ValueError(f"Age value should be numeric, {self.age}")

        if age < 30:
            return YOUNG
        elif age >= 30 and age < 60:
            return MIDDLE
        elif age >= 60:
            return OLD

    def get_final_label(self):
        return self.get_mask_label() * 6 + self.get_gender_label() * 3 + self.get_age_label()

class TrainDataset(Dataset):
    current_os = platform.system()
    num_classes = 3 * 2 * 3

    _file_names = {"incorrect_mask", "mask1", "mask2", "mask3", "mask4", "mask5", "normal"}

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, val_ratio=0.2):
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.transform = None
        self.scandir_and_labeling()

    def scandir_and_labeling(self):
        print("File Scan Start.")
        each_human_dirs = os.listdir(self.data_dir)
        for each_human_dir in tqdm(each_human_dirs):
            if each_human_dir.startswith("."):  # "." 으로 시작하는 숨김 파일 무시
                continue

            img_folder = MyPlatform.path_join(self.current_os, [self.data_dir, each_human_dir])
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # _file_names 변수 내 파일명이 아닌 invalid 파일 무시
                    continue

                img_path = MyPlatform.path_join(self.current_os, [self.data_dir, each_human_dir, file_name])
                id, gender, race, age = each_human_dir.split("_")

                labeling_info = Data_Labeling(_file_name, gender, age)

                mask_label = labeling_info.get_mask_class()
                gender_label = labeling_info.get_gender_class()
                age_label = labeling_info.get_age_class()

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
        print("File Scan Done.")

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in tqdm(self.image_paths):
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index):
        return self.mask_labels[index]

    def get_gender_label(self, index):
        return self.gender_labels[index]

    def get_age_label(self, index):
        return self.age_labels[index]

    def encode_multi_class(mask_label, gender_label, age_label):
        return mask_label * 6 + gender_label * 3 + age_label

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, "tranform 함수 無. set_transform 사용 必"

        image = np.array(Image.open(self.img_paths[index]))
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        label = self.encode_multi_class(mask_label, gender_label, age_label)

        image = self.transform(image)
        return image, label
