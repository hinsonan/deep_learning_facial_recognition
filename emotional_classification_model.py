from turtle import forward
from torch import nn
import torch.nn.functional as F

class EmotionalModel(nn.Module):

    def __init__(self) -> None:
        super(EmotionalModel,self).__init__()

        self.input_layer = nn.Conv2d(1,25,3,1,0)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1232100,8)

    def forward(self,x):
        out = F.relu(self.input_layer(x))
        out = self.flatten(out)
        out = self.linear(out)
        return out