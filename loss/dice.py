import torch 
from torch import nn
import torch.nn.functional as F

def dice_loss(pred, target, smooth = 1.):
    
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    loss_label_1 = loss[:,0].mean()
    loss_label_2 = loss[:,1].mean()
    loss_label_3 = loss[:,2].mean()
    loss_label_4 = loss[:,3].mean()

    return ((loss_label_1+loss_label_2+loss_label_3+loss_label_4)/4 , (loss_label_1 , loss_label_2 ,loss_label_3,loss_label_4))