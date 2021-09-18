import os
import platform
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, random_split
#############################################
# from dotenv import load_dotenv

# from albumentations import *
# from albumentations.pytorch import ToTensorV2

# import random
# from collections import defaultdict
# from enum import Enum
# from typing import Tuple, List
# import torch
# from torch.utils.data import Dataset, Subset, random_split
# from torchvision import transforms
# from torchvision.transforms import *

class MyOS:
    current_os = platform.system()

    @classmethod
    def path_join(cls, params):
        result = os.path.join(*params)
        if cls.current_os == "Windows":
            result = result.replace("\\", "/")

        return result

class Data_Labeling():
    MASK_LABEL = dict(MASK = 0, INCORRECT = 1, NORMAL = 2)
    GENDER_LABEL = dict(MALE = 0, FEMALE = 1)
    AGE_LABEL = dict(YOUNG = 0, MIDDLE = 1, OLD = 2)
    
    @classmethod
    def get_mask_label(cls, image_name):
        if image_name.lower().startswith("mask"):
            return cls.MASK_LABEL["MASK"]
        elif image_name.lower().startswith("incorrect"):
            return cls.MASK_LABEL["INCORRECT"]
        elif image_name.lower().startswith("normal"):
            return cls.MASK_LABEL["NORMAL"]
        else:
            raise ValueError(f"Mask value should be either 'mask' or 'incorrect' or 'normal' // , {self.image_name}")
    
    @classmethod
    def get_gender_label(cls, gender):
        if gender.lower() == "male":
            return cls.GENDER_LABEL["MALE"]
        elif gender.lower() == "female":
            return cls.GENDER_LABEL["FEMALE"]
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {gender}")

    @classmethod
    def get_age_label(cls, age):
        try:
            age = int(age)
        except Exception:
            raise ValueError(f"Age value should be numeric, {age}")

        if age < 30:
            return cls.AGE_LABEL["YOUNG"]
        elif age >= 30 and age < 60:
            return cls.AGE_LABEL["MIDDLE"]
        elif age >= 60:
            return cls.AGE_LABEL["OLD"]
    
    @staticmethod
    def encode_final_label(mask_label, gender_label, age_label):
        return mask_label * 6 + gender_label * 3 + age_label
    
    @staticmethod
    def decode_final_label(final_label):
        mask_label = (final_label // 6) % 3
        gender_label = (final_label // 3) % 2
        age_label = final_label % 3
        return mask_label, gender_label, age_label

class MyDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {"incorrect_mask", "mask1", "mask2", "mask3", "mask4", "mask5", "normal"}

    image_paths = []
    final_labels = []

    def __init__(self, data_dir, val_ratio=0.2, transform=None):
        self.data_dir = data_dir
        self.val_ratio = val_ratio
        self.transform = transform
        self.mean = None
        self.std = None
        self.scandir_and_labeling()

    def scandir_and_labeling(self):
        print("File Scan Start.")
        self.data_dir = "./input/data/train/images/"
        each_human_dirs = os.listdir(self.data_dir)

        for each_human_dir in tqdm(each_human_dirs, desc = 'dirs'):
            if each_human_dir.startswith("."):  # "." 으로 시작하는 숨김 파일 무시
                continue

            img_folder = MyOS.path_join([self.data_dir, each_human_dir])
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # _file_names 변수 내 파일명이 아닌 invalid 파일 무시
                    continue

                img_path = MyOS.path_join([self.data_dir, each_human_dir, file_name])
                id, gender, race, age = each_human_dir.split("_")

                mask_label = Data_Labeling.get_mask_label(_file_name)
                gender_label = Data_Labeling.get_gender_label(gender)
                age_label = Data_Labeling.get_age_label(age)

                self.image_paths.append(img_path)
                self.final_labels.append(Data_Labeling.encode_final_label(mask_label, gender_label, age_label))
        print("File Scan Done.")

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("Image Analysis Start.('mean' & 'std')")
            sums = []
            squared = []
            for image_path in tqdm(self.image_paths):
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255
            print(f"Image Info > mean = {self.mean}, std = {self.std}")
            print("Image Analysis Done.('mean' & 'std')")

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
        print("Transform Setting 完.")

    def __getitem__(self, index):
        assert self.transform is not None, "tranform 함수 無. set_transform 사용 必"

        image = np.array(Image.open(self.image_paths[index]))
        label = self.final_labels[index]

        image = self.transform(image=image)['image']
        return image, label
    
    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set