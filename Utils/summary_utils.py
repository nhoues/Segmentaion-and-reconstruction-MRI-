import gc ,random 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn import model_selection

import cv2
import SimpleITK as sitk
from ipywidgets import interact, fixed
from tqdm import tqdm 
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

def display(image_z, fixed):
    fig, axs = plt.subplots(figsize=(10,8))
    axs.imshow(sitk.GetArrayViewFromImage(fixed[:,:,image_z]), cmap='hot', interpolation='nearest');
    plt.show()

def display_images_with_Label(image_z , alpha , im_orginal , Label  ):
    fig, axs = plt.subplots(1,2,figsize=(20,8))
    im = (1-alpha) * im_orginal + alpha * (sitk.Cast(Label, sitk.sitkFloat32) )
    axs[0].imshow(sitk.GetArrayFromImage(im[:,:,image_z]),cmap=plt.cm.Greys_r);
    axs[1].imshow(sitk.GetArrayFromImage(sitk.LabelToRGB(Label)[:,:,image_z]));

def visualize_results(image_z   , label , label_hat): 
    fig, axs = plt.subplots(1,2,figsize=(20,8))

    axs[0].imshow(sitk.GetArrayFromImage(sitk.LabelToRGB(label)[:,:,image_z]));
    axs[0].set_title('Real mask')
  
    axs[1].imshow(sitk.GetArrayFromImage(sitk.LabelToRGB(label_hat)[:,:,image_z]));
    axs[1].set_title('predicted mask')

def soft_max_2d(pred) : 
    s = pred.shape 
    y_final = torch.randn((s[0],s[2],s[3]))
    for i in range(s[2]) : 
        for j in range(s[3]) : 
            y_final[:,i,j] = F.softmax(pred[:,:,i,j],dim = 1 ).argmax(dim=1)
    
    return y_final 

def dice(pred, target, smooth = 0.00001):
    
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    loss_label_1 = 1-loss[:,1].mean()
    loss_label_2 = 1-loss[:,2].mean()
    loss_label_3 = 1-loss[:,3].mean()
    loss_over_all = 1 - loss.mean().mean()
    
    return  loss_label_1 , loss_label_2 ,loss_label_3 , loss_over_all 

def loss_eval  (data_set, model , device , merging = False):
   
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=128,
        num_workers=8
    )

    model.eval()
    tr_loss = 0
    counter = 0
    
    label1_loss  = 0
    label2_loss  = 0 
    label3_loss  = 0 
   

    with torch.no_grad():
        
        for bi, d in enumerate(data_loader):
            
            y = d["label"].to(device, dtype=torch.float)
            x = d["LR"].to(device, dtype=torch.float) 
            y_hat   = model(x.unsqueeze(1)) #forward prop 
            
            if merging : 
                y_hat = y_hat[:,1:,:,:]

                
            if bi == 0 :
                full_x = x
                full_y = y
                y_final = y_hat 
            else :
                full_x = torch.cat([full_x , x] , dim = 0 ) 
                full_y = torch.cat([full_y , y] , dim = 0 )  
                y_final= torch.cat([y_final , y_hat] , dim = 0 )  
                
            
        y_final = soft_max_2d(y_final)
        y_final = torch.cat([ (y_final==0).type(torch.long).unsqueeze(1) ,(y_final==1).type(torch.long).unsqueeze(1),(y_final==2).type(torch.long).unsqueeze(1),(y_final==3).type(torch.long).unsqueeze(1) ],dim = 1)
        y_final = y_final.to(device)

        labels = dice(y_final, full_y) # Loss calaculation of batch i 
    return labels[0].item() ,  labels[1].item() , labels[2].item() , labels[3].item()

def construction(df , model , is_simple = True) : 
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        df,
        batch_size=8,
        num_workers=4
    )
    with torch.no_grad():
        
        for bi, d in tqdm(enumerate(data_loader) , total = len(data_loader)) :
       
            y = d["label"].to(device, dtype=torch.float)

            x = d["LR"].to(device, dtype=torch.float) 
            
            y_hat   = model(x.unsqueeze(1)) #forward prop
            if is_simple == False : 
                y_hat = y_hat[:,1:,:,:]
            if bi == 0 :
                
                full_x     = x
                full_y     = y
                full_y_hat =y_hat 
            
            else :
                
                full_x     = torch.cat([full_x , x] , dim = 0 ) 
                full_y     = torch.cat([full_y , y] , dim = 0 )  
                full_y_hat = torch.cat([full_y_hat , y_hat] , dim = 0 )  
                
    return full_x.cpu().numpy() , full_y , full_y_hat 

def write( x , path ) :  
    img = sitk.GetImageFromArray(x)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(img)