import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim 

class SegmenationAdeversarial(nn.Module) : 
    def __init__(self,in_channels = 5 ) : 
        super(SegmenationAdeversarial,self).__init__( ) 
        self.full_model = nn.Sequential(
                                            nn.Conv2d(in_channels , 64 , kernel_size = 5 ) , 
                                            nn.ReLU(inplace=True) , 
                                            nn.MaxPool2d(kernel_size = 2 , stride=2),

                                            nn.Conv2d(64, 128 , kernel_size = 5 ) , 
                                            nn.ReLU(inplace=True) , 
                                            nn.MaxPool2d(kernel_size = 2 , stride=2),

                                            nn.Conv2d(128, 128 , kernel_size = 3 ) , 
                                            nn.ReLU(inplace=True) , 
                                            nn.MaxPool2d(kernel_size = 2 , stride=2),

                                            nn.Conv2d(128, 256 , kernel_size = 3 ) , 
                                            nn.ReLU(inplace=True) , 
                                            nn.MaxPool2d(kernel_size = 2 , stride=2),

                                            nn.Conv2d(256, 512 , kernel_size = 3 ) , 
                                            nn.ReLU(inplace=True) ,

                                            nn.Conv2d(512, 2 , kernel_size = 3 ) ,
                                            nn.Sigmoid() , 
                                        )
    def forward(self,image ) : 
        x_hat = self.full_model(image)
        return(x_hat[:,:,0,0])

class Adeversarial(nn.Module) : 
    def __init__(self) : 
        super(Adeversarial, self).__init__()
        
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128 , kernel_size = 5), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size = 2 , stride=2),
        
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Sigmoid()
        
        )
        
    def forward(self, image):
        y = self.image_encoder(image)
        return(y[:,:,0,0])           