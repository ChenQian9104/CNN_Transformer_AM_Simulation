from torchvision.models import resnet34, ResNet34_Weights
import torch.nn as nn

cnn = resnet34(weights=ResNet34_Weights.DEFAULT)

class BackBoneResNet(nn.Module): 

    def __init__(self, resNet=cnn, n=6): 
        super().__init__()

        self.layer_list = [layer for layer in resNet.children()]

        self.layers = nn.ModuleList([self.layer_list[i] for i in range(n)])

    def forward(self, x): 

        for layer in self.layers: 
            x = layer(x)
        return x 

