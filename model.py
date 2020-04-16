import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_f, out_f, kernel_size, padding, max_pool_layer=False):
    if max_pool_layer==False : 
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_f)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_f),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25)
        )
    
class CancerNet(nn.Module):
    def __init__(self, in_c, n_classes=1):
        super().__init__()
        self.conv_block1 = conv_block(in_c, 32, kernel_size=3, padding=1, max_pool_layer=True)       
        self.conv_block2 = conv_block(32, 64, kernel_size=3, padding=1)
        self.conv_block3 = conv_block(64, 64, kernel_size=3, padding=1, max_pool_layer=True)
        self.conv_block4 = conv_block(64, 128, kernel_size=3, padding=1)
        self.conv_block5 = conv_block(128, 128, kernel_size=3, padding=1)
        self.conv_block6 = conv_block(128, 128, kernel_size=3, padding=1, max_pool_layer=True)

        
        self.fc_block = nn.Sequential(
            nn.Linear(128*6*6, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.25)
        )
        
        self.final = nn.Linear(256, n_classes)

        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)

        x = x.view(x.size(0), -1) # flat
        
        x = self.fc_block(x)
        
        x = self.final(x)
        
        return x