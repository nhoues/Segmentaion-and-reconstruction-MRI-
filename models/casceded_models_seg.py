import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim 

from models.UNet import * 
 
class ParallelCascadedUNet(nn.Module) : 
    def __init__(self) :
        super(ParallelCascadedUNet,self).__init__()
        
        self.segmentation_1  = UNet(1,4, segmentation = False) 
        self.segmentation_2  = UNet(2,4, segmentation = False)
        self.segmentation_3  = UNet(2,4, segmentation = False)
        
        self.reconstruction_1 = UNet(1,1, segmentation = False)
        self.reconstruction_2 = UNet(5,1, segmentation = False)
        self.reconstruction_3 = UNet(5,1, segmentation = False)
    
    def forward(self,image) :
        seg_1 = self.segmentation_1(image) 
        rec_1 = self.reconstruction_1(image)
        
        x_hr_lr = torch.cat([rec_1,image] , dim =1)
        seg_2  = self.segmentation_2(x_hr_lr) 
        
        c_seg = self.construct_seg(seg_1)
        x_merge = torch.cat([image,c_seg] , dim = 1 )
        rec_2 = self.reconstruction_2(x_merge)
        
        x_hr_lr = torch.cat([rec_2,image] , dim =1)
        seg_3  = self.segmentation_3(x_hr_lr) 
        
        c_seg = self.construct_seg(seg_2)
        x_merge = torch.cat([image,c_seg] , dim = 1 )
        rec_3 = self.reconstruction_3(x_merge)
        
        return seg_3
    def construct_seg(self,x) : 
        y_1 = torch.argmax(nn.Softmax2d()(x) , dim=1)
        x_label_0 = (y_1==0).type(torch.long).unsqueeze(1)
        x_label_1 = (y_1==1).type(torch.long).unsqueeze(1)
        x_label_2 = (y_1==2).type(torch.long).unsqueeze(1)
        x_label_3 = (y_1==3).type(torch.long).unsqueeze(1)
        generated_segmenation = torch.cat([x_label_0,x_label_1,x_label_2,x_label_3] , dim = 1)
        generated_segmenation = generated_segmenation.type(torch.float)
        return generated_segmenation 


class CascadedUNet(nn.Module) : 
    def __init__(self) :
        super(CascadedUNet,self).__init__()
        self.layer_1  = UNet(1,1, segmentation = False) 
        self.layer_2  = UNet(2,4, segmentation = False)
        self.layer_3  = UNet(6,1, segmentation = False)
        self.layer_4  = UNet(2,4, segmentation = False)
        self.layer_5  = UNet(6,1, segmentation = False)
    def forward(self,image) :
        
        hr_1 = self.layer_1(image)
        im = torch.cat([hr_1,image],dim=1)
        
        seg_1 = self.layer_2(im)
        
        seg_t = self.segmentation_gen(seg_1)
        im = torch.cat([hr_1,image,seg_t*image],dim=1)
        
        hr_2 = self.layer_3(im) 
        
        im = torch.cat([hr_2,image],dim=1)
        seg_2 = self.layer_4(im)
        
        seg_t = self.segmentation_gen(seg_2) 
        
        im = torch.cat([hr_2,image,seg_t*image],dim=1)
        hr_3 = self.layer_5(im)
        
        return seg_2
    def segmentation_gen(self,x) : 
        
        y_1 = torch.argmax(nn.Softmax2d()(x) , dim=1)
        x_label_0 = (y_1==0).type(torch.long).unsqueeze(1)
        x_label_1 = (y_1==1).type(torch.long).unsqueeze(1)
        x_label_2 = (y_1==2).type(torch.long).unsqueeze(1)
        x_label_3 = (y_1==3).type(torch.long).unsqueeze(1)
        y_1 = torch.cat([x_label_0,x_label_1,x_label_2,x_label_3] , dim = 1)
        y_1 = y_1.type(torch.float)
        
        return y_1 

class GANCascadedUNet(nn.Module) : 
    def __init__(self) :
        super(GANCascadedUNet,self).__init__()
        self.layer_1  = UNet(1,1, segmentation = False) 
        self.layer_2  = UNet(2,4, segmentation = False)
        self.layer_3  = UNet(6,1, segmentation = False)
        self.layer_4  = UNet(2,4, segmentation = False)
        self.layer_5  = UNet(6,1, segmentation = False)
    def forward(self,image) :
        
        hr_1 = self.layer_1(image)
        im = torch.cat([hr_1,image],dim=1)
        
        seg_1 = self.layer_2(im)
        
        seg_t = self.segmentation_gen(seg_1)
        im = torch.cat([hr_1,image,seg_t*image],dim=1)
        
        hr_2 = self.layer_3(im) 
        
        im = torch.cat([hr_2,image],dim=1)
        seg_2 = self.layer_4(im)
        
        seg_t = self.segmentation_gen(seg_2) 
        
        im = torch.cat([hr_2,image,seg_t*image],dim=1)
        hr_3 = self.layer_5(im)
        
        return seg_2  
    def segmentation_gen(self,x) : 
        
        y_1 = torch.argmax(nn.Softmax2d()(x) , dim=1)
        x_label_0 = (y_1==0).type(torch.long).unsqueeze(1)
        x_label_1 = (y_1==1).type(torch.long).unsqueeze(1)
        x_label_2 = (y_1==2).type(torch.long).unsqueeze(1)
        x_label_3 = (y_1==3).type(torch.long).unsqueeze(1)
        y_1 = torch.cat([x_label_0,x_label_1,x_label_2,x_label_3] , dim = 1)
        y_1 = y_1.type(torch.float)
        
        return y_1 
    
class GANParallelCascadedUNet(nn.Module) : 
    def __init__(self) :
        super(GANParallelCascadedUNet,self).__init__()
        
        self.segmentation_1  = UNet(1,4, segmentation = False) 
        self.segmentation_2  = UNet(2,4, segmentation = False)
        self.segmentation_3  = UNet(2,4, segmentation = False)
        
        self.reconstruction_1 = UNet(1,1, segmentation = False)
        self.reconstruction_2 = UNet(5,1, segmentation = False)
        self.reconstruction_3 = UNet(5,1, segmentation = False)
    
    def forward(self,image) :
        seg_1 = self.segmentation_1(image) 
        rec_1 = self.reconstruction_1(image)
        
        x_hr_lr = torch.cat([rec_1,image] , dim =1)
        seg_2  = self.segmentation_2(x_hr_lr) 
        
        c_seg = self.construct_seg(seg_1)
        x_merge = torch.cat([image,c_seg] , dim = 1 )
        rec_2 = self.reconstruction_2(x_merge)
        
        x_hr_lr = torch.cat([rec_2,image] , dim =1)
        seg_3  = self.segmentation_3(x_hr_lr) 
        
        c_seg = self.construct_seg(seg_2)
        x_merge = torch.cat([image,c_seg] , dim = 1 )
        rec_3 = self.reconstruction_3(x_merge)
       
        pred_seg = self.construct_seg(seg_3)
        return seg_3
    
    def construct_seg(self,x) : 
        y_1 = torch.argmax(nn.Softmax2d()(x) , dim=1)
        x_label_0 = (y_1==0).type(torch.long).unsqueeze(1)
        x_label_1 = (y_1==1).type(torch.long).unsqueeze(1)
        x_label_2 = (y_1==2).type(torch.long).unsqueeze(1)
        x_label_3 = (y_1==3).type(torch.long).unsqueeze(1)
        generated_segmenation = torch.cat([x_label_0,x_label_1,x_label_2,x_label_3] , dim = 1)
        generated_segmenation = generated_segmenation.type(torch.float)
        return generated_segmenation 