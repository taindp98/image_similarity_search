import torch
import torch.nn as nn
from torchvision import models

class RecommendNet(nn.Module):
    """
    
    """
    def __init__(self):
        super(RecommendNet, self).__init__()
        self.net = models.resnet50(pretrained=True)

        ## freeze layers
        
        # for param in  self.net.parameters():
        #     param.requires_grad = False

        num_ftrs = self.net.fc.in_features

        out_features = 256

        self.net.fc = nn.Linear(num_ftrs, out_features)

    def forward(self, input):
        output = self.net(input)
        return output