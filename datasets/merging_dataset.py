import gc, random

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import SimpleITK as sitk
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import albumentations as A


class Merging_data_set:
    def __init__(
        self, df, subjects, Left=True, is_train=False, img_height=160, img_width=160
    ):

        self.subjects = subjects
        self.sub_id = df.Subject_num.values
        self.slice = df.slice.values

        if Left:
            HR_paths = subjects.HR_Left_path.values
            LR_paths = subjects.LR_Left_path.values
            label_paths = subjects.HRL_L_path.values
            subject_num = subjects.Subject_num.values
            HR = dict()
            LR = dict()
            im_label = dict()
            for i in range(len(HR_paths)):
                HR[subject_num[i]] = sitk.ReadImage(HR_paths[i])
                LR[subject_num[i]] = sitk.ReadImage(LR_paths[i])
                im_label[subject_num[i]] = sitk.ReadImage(label_paths[i], sitk.sitkInt8)

        else:
            HR_paths = subjects.HR_Right_path.values
            LR_paths = subjects.LR_Right_path.values
            label_paths = subjects.HRR_L_path.values
            subject_num = subjects.Subject_num.values
            HR = dict()
            LR = dict()
            im_label = dict()
            for i in range(len(HR_paths)):
                HR[subject_num[i]] = sitk.ReadImage(HR_paths[i])
                LR[subject_num[i]] = sitk.ReadImage(LR_paths[i])
                im_label[subject_num[i]] = sitk.ReadImage(label_paths[i], sitk.sitkInt8)

        self.LR = LR
        self.HR = HR
        self.label = im_label

        self.img_height = img_height
        self.img_width = img_width
        self.is_train = is_train

    def transform(self, HR, LR, mask):
        # Resize
        resize = A.Resize(
            height=self.img_height, width=self.img_width, always_apply=True
        )

        HR = resize(image=HR)
        LR = resize(image=LR, mask=mask)
        HR = HR["image"]
        mask = LR["mask"]
        LR = LR["image"]
        # Random RandomRotate90
        if (random.random() > 0.5) and self.is_train:
            aug = A.RandomRotate90(p=1)

            HR = aug(image=HR)
            HR = HR["image"]

            LR = aug(image=LR, mask=mask)
            mask = LR["mask"]
            LR = LR["image"]

        return HR, LR, mask

    def __len__(self):

        return len(self.slice)

    def __getitem__(self, item):
        out = dict()

        HR = (
            sitk.GetArrayFromImage(
                self.HR[int(self.sub_id[item])][:, :, int(self.slice[item])]
            )
            / 100
        )
        LR = (
            sitk.GetArrayFromImage(
                self.LR[int(self.sub_id[item])][:, :, int(self.slice[item])]
            )
            / 100
        )
        mask = sitk.GetArrayFromImage(
            self.label[int(self.sub_id[item])][:, :, int(self.slice[item])]
        )

        HR, LR, mask = self.transform(HR, LR, mask)

        mask = torch.tensor(mask, dtype=torch.long)
        x_label_0 = (mask == 0).type(torch.long).unsqueeze(0)
        x_label_1 = (mask == 1).type(torch.long).unsqueeze(0)
        x_label_2 = (mask == 2).type(torch.long).unsqueeze(0)
        x_label_3 = (mask == 3).type(torch.long).unsqueeze(0)
        x = torch.cat([x_label_0, x_label_1, x_label_2, x_label_3], dim=0)
        out["label"] = x

        HR = torch.tensor(HR, dtype=torch.float)
        out["HR"] = HR

        LR = torch.tensor(LR, dtype=torch.float)
        out["LR"] = LR

        return out


class Segmentation_data_set:
    def __init__(
        self, df, subjects, Left=True, is_train=False, img_height=160, img_width=160
    ):

        self.subjects = subjects
        self.sub_id = df.Subject_num.values
        self.slice = df.slice.values

        if Left:
            image_paths = subjects.LR_Left_path.values
            label_paths = subjects.HRL_L_path.values
            subject_num = subjects.Subject_num.values
            im = dict()
            im_label = dict()
            for i in range(len(image_paths)):
                im[subject_num[i]] = sitk.ReadImage(image_paths[i])
                im_label[subject_num[i]] = sitk.ReadImage(label_paths[i], sitk.sitkInt8)

        else:
            image_paths = subjects.LR_Right_path.values
            label_paths = subjects.HRR_L_path.values
            subject_num = subjects.Subject_num.values
            im = dict()
            im_label = dict()
            for i in range(len(image_paths)):
                im[subject_num[i]] = sitk.ReadImage(image_paths[i])
                im_label[subject_num[i]] = sitk.ReadImage(label_paths[i], sitk.sitkInt8)

        self.image = im
        self.label = im_label

        if is_train:

            self.aug = A.Compose(
                [
                    A.Resize(height=img_height, width=img_width, always_apply=True),
                    A.RandomRotate90(p=0.5),
                ]
            )
        else:

            self.aug = A.Resize(height=img_height, width=img_width, always_apply=True)

    def __len__(self):
        return len(self.slice)

    def __getitem__(self, item):
        out = dict()
        image = (
            sitk.GetArrayFromImage(
                self.image[int(self.sub_id[item])][:, :, int(self.slice[item])]
            )
            / 255
        )
        label = sitk.GetArrayFromImage(
            self.label[int(self.sub_id[item])][:, :, int(self.slice[item])]
        )

        augmented = self.aug(image=image, mask=label)
        label = torch.tensor(augmented["mask"], dtype=torch.long)
        x_label_0 = (label == 0).type(torch.long).unsqueeze(0)
        x_label_1 = (label == 1).type(torch.long).unsqueeze(0)
        x_label_2 = (label == 2).type(torch.long).unsqueeze(0)
        x_label_3 = (label == 3).type(torch.long).unsqueeze(0)
        x = torch.cat([x_label_0, x_label_1, x_label_2, x_label_3], dim=0)
        y = torch.tensor(augmented["image"], dtype=torch.float)
        out["image"] = y
        out["label"] = x

        return out
