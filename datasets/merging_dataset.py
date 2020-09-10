import gc, random

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import SimpleITK as sitk
from PIL import Image
import cv2

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import albumentations as A

def get_training_augmentation(resize_to=(160,160)):
    print('[get_training_augmentation]  resize_to:', resize_to) 

    train_transform = [
                
        
        A.ShiftScaleRotate(scale_limit=0.20, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.GridDistortion(p=0.5),
        A.Resize(*resize_to),
        
    ]

    return A.Compose(train_transform)

class Merging_data_set() : 
    
    def __init__(self, df , subjects , Left = True , is_train= False , img_height=160 , img_width=160 ) : 
    
        self.subjects = subjects
        self.sub_id = df.Subject_num.values 
        self.slice = df.slice.values
        
        if Left : 
            HR_paths = subjects.HR_Left_path.values
            LR_paths = subjects.LR_Left_path.values
            label_paths = subjects.HRL_L_path.values
            subject_num = subjects.Subject_num.values
            HR = dict()
            LR = dict()
            im_label = dict()
            for i in range(len(HR_paths)) : 
                HR[subject_num[i]]  = sitk.ReadImage(HR_paths[i]) 
                LR[subject_num[i]]  = sitk.ReadImage(LR_paths[i]) 
                im_label[subject_num[i]] = sitk.ReadImage(label_paths[i],sitk.sitkInt8)
      
        else : 
            HR_paths = subjects.HR_Right_path.values
            LR_paths = subjects.LR_Right_path.values
            label_paths = subjects.HRR_L_path.values
            subject_num = subjects.Subject_num.values
            HR = dict()
            LR = dict()
            im_label = dict()
            for i in range(len(HR_paths)) : 
                HR[subject_num[i]]  = sitk.ReadImage(HR_paths[i]) 
                LR[subject_num[i]]  = sitk.ReadImage(LR_paths[i]) 
                im_label[subject_num[i]] = sitk.ReadImage(label_paths[i],sitk.sitkInt8)
        
        self.LR = LR  
        self.HR = HR  
        self.label = im_label  
        
        self.img_height = img_height 
        self.img_width  = img_width 
        
        if is_train : 
            self.aug = get_training_augmentation((img_height,img_width))
        else :
            self.aug = A.Resize(img_height,img_width)
        

    
    def __len__(self) : 
        
        return len(self.slice)
    
    def __getitem__(self,item) : 
        out = dict()
        
        HR  = sitk.GetArrayFromImage(self.HR[int(self.sub_id[item])][:,:,int(self.slice[item])])/100
        LR  = sitk.GetArrayFromImage(self.LR[int(self.sub_id[item])][:,:,int(self.slice[item])])/100
        mask = sitk.GetArrayFromImage(self.label[int(self.sub_id[item])][:,:,int(self.slice[item])])  
        
        transformed = self.aug(image=HR,  masks=[LR, mask])
        HR = transformed['image']
        LR = transformed['masks'][0]
        mask = transformed['masks'][1]
    
       
        mask = torch.tensor( mask ,dtype = torch.long ) 
        x_label_0 = (mask==0).type(torch.long).unsqueeze(0)
        x_label_1 = (mask==1).type(torch.long).unsqueeze(0)
        x_label_2 = (mask==2).type(torch.long).unsqueeze(0)
        x_label_3 = (mask==3).type(torch.long).unsqueeze(0)
        x = torch.cat([x_label_0,x_label_1,x_label_2,x_label_3],dim = 0)
        out['label'] = x
        
        HR= torch.tensor(HR ,dtype = torch.float)
        out['HR'] = HR
        
        LR = torch.tensor(LR ,dtype = torch.float)
        out['LR'] = LR 
        
            
        return out