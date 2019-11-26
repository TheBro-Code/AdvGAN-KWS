import torch
import torch.nn as nn
from src.layers import *

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv = nn.Conv2d(in_channels=1, out_channels=128,
                           kernel_size=(3,4), stride=(1,2),padding=(1,1))
    self.conv_gated = nn.Conv2d(in_channels=1, out_channels=128,
                           kernel_size=(3,4), stride=(1,2),padding=(1,1))
    self.downsample1 = DownSample_Disc(128,(4,4),(2,2),256)
    self.downsample2 = DownSample_Disc(256,(4,4),(2,2),256)
    self.downsample3 = DownSample_Disc(256,(5,4),(1,2),512)
    self.linear = nn.Linear(512,1)

  def forward(self,input):
    out = self.conv(input)
    out_gated = self.conv_gated(input)
    out = out * nn.Sigmoid()(out_gated)
    out = self.downsample1(out)
    out = self.downsample2(out)
    out = self.downsample3(out)
    out = out.view(out.shape[0],out.shape[1],-1)
    out = torch.mean(out,axis=2)
    out = self.linear(out)
    out = nn.Sigmoid()(out)
    return out
