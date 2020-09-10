import gc, random

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import SimpleITK as sitk
from PIL import Image
import cv2


# define a function plot image and mask
def plot_image_and_mask(image, mask):
    '''
    Function to plot a single prediction:
    INPUT:
        image - PIL image 
        mask - PIL image with corresponding mask
    '''
    fig, axs = plt.subplots(1, 3, figsize=(20,10))

    #plot the original data
    axs[0].imshow(image , cmap=plt.cm.Greys_r)
    axs[0].axis('off')
    axs[0].set_title('Image')

    #plot the mask
    axs[1].imshow(mask, cmap = "Reds")
    axs[1].axis('off')   
    axs[1].set_title('Mask')
    
    #plot image and add the mask
    axs[2].imshow(image, cmap=plt.cm.Greys_r)
    axs[2].imshow(mask, alpha = 0.4, cmap = "Reds")
    axs[2].axis('off')   
    axs[2].set_title('Image with mask overlay')

    # set suptitle
    plt.suptitle('Image with mask')
    plt.show()
    
    
def plot_HR_LR__mask(HR,LR, mask):
  
    fig, axs = plt.subplots(1, 3, figsize=(20,10))

    #plot the original data
    axs[0].imshow(HR , cmap=plt.cm.Greys_r)
    axs[0].axis('off')
    axs[0].set_title('HR image')

    #plot the mask
    axs[1].imshow(LR , cmap=plt.cm.Greys_r)
    axs[1].axis('off')   
    axs[1].set_title('LR image')
    
    #plot image and add the mask
    axs[2].imshow(LR, cmap=plt.cm.Greys_r)
    axs[2].imshow(mask, alpha = 0.4, cmap = "Reds")
    axs[2].axis('off')   
    axs[2].set_title('LR with mask overlay')

    plt.show()