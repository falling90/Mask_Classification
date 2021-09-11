import platform
import os
from torch.utils.data import Dataset
from PIL import Image
from albumentations import *
from albumentations.pytorch import ToTensorV2
import numpy as np

# import random
# from collections import defaultdict
# from enum import Enum
# from typing import Tuple, List


# import torch

# from torch.utils.data import Dataset, Subset, random_split
# from torchvision import transforms
# from torchvision.transforms import *

class config_info:
    main_dirpath = 'C:/Users/KJY/Desktop/input/data/train'
    image_dirpath = f'{main_dirpath}/images'
    csv_dirpath = f'{main_dirpath}/train.csv'
    new_csv_path = f'{main_dirpath}/new_train.csv'

    test_dir = f'{main_dirpath}/eval'
    testset_ratio = 0.1

    epochs=100
    k_fold_split = 5


class Data_Labeling():
    '''
    print(Data_Labeling("incorrect_mask.jpg", "male", 23)['class'])
    '''

    def __init__(self, image_name, gender, age):
        self.image_name = image_name
        self.gender = gender
        self.age = age

    def __getitem__(self, data):
        if data == "class":
            return 6 * self.get_mask_class(self.image_name) + 3 * self.get_gender_class(
                self.gender) + self.get_age_class(self.age)

    def get_mask_class(self):
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

    def get_gender_class(self):
        MALE = 0
        FEMALE = 1

        if self.gender.lower() == "male":
            return MALE
        elif self.gender.lower() == "female":
            return FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {self.gender}")

    def get_age_class(self):
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

class Path_Join_by_Platform():
    def __init__(self):
        self.current_os = platform.system()
        # self.params = params

    def path_join(self, params):
        result = os.path.join(*params)
        if self.current_os == "Windows":
            result = result.replace("\\", "/")
        return result

class TrainDataset(Dataset):
    # Mask(3 class) * Gender(2 class) * Age(3 class)
    num_classes = 3 * 2 * 3

    _file_names = {"incorrect_mask", "mask1", "mask2", "mask3", "mask4", "mask5", "normal"}

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.scandir_and_labeling()
        # self.calc_statistics()

    def scandir_and_labeling(self):
        pathjoin = Path_Join_by_Platform()
        each_human_dirs = os.listdir(self.data_dir)
        for each_human_dir in each_human_dirs:
            if each_human_dir.startswith("."):  # "." 으로 시작하는 숨김 파일 무시
                continue

            img_folder = pathjoin.path_join([self.data_dir, each_human_dir])
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # _file_names 변수 내 파일명이 아닌 invalid 파일 무시
                    continue

                img_path = pathjoin.path_join([self.data_dir, each_human_dir, file_name])
                id, gender, race, age = each_human_dir.split("_")

                labeling_info = Data_Labeling(_file_name, gender, age)

                mask_label = labeling_info.get_mask_class()
                gender_label = labeling_info.get_gender_class()
                age_label = labeling_info.get_age_class()

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

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

class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = None
        
    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, "tranform 함수 無. set_transform 사용 必"

        image = np.array(Image.open(self.img_paths[index]))

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)