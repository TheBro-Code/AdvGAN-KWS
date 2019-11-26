import torch
import torch.nn as nn
from src.layers import *

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.conv1 = nn.Conv1d(in_channels=128, out_channels=128,
                           kernel_size=15, stride=1,
                           padding=7)
    self.conv1_gated = nn.Conv1d(in_channels=128, out_channels=128,
                           kernel_size=15, stride=1,
                           padding=7)
    self.downsample1 = DownSample_Gen(128,6,2,256)
    self.downsample2 = DownSample_Gen(256,6,2,256)
    self.resblock1 = ResBlock(256,3,3,512,256,1,1)
    self.resblock2 = ResBlock(256,3,3,512,256,1,1)
    self.resblock3 = ResBlock(256,3,3,512,256,1,1)
    self.resblock4 = ResBlock(256,3,3,512,256,1,1)
    self.resblock5 = ResBlock(256,3,3,512,256,1,1)
    self.resblock6 = ResBlock(256,3,3,512,256,1,1)
    self.upsample1 = UpSample(256,5,1,512,2)
    self.upsample2 = UpSample(256,5,1,256,2)
    self.conv2 = nn.Conv1d(in_channels=128,out_channels=128,\
                           kernel_size=15,stride=1,\
                           padding=7)

  def forward(self,input):
    out = self.conv1(input)
    out_gated = self.conv1_gated(input)
    out = out * nn.Sigmoid()(out_gated)
    out = self.downsample1(out)
    out = self.downsample2(out)
    out = self.resblock1(out)
    out = self.resblock2(out)
    out = self.resblock3(out)
    out = self.resblock4(out)
    out = self.resblock5(out)
    out = self.resblock6(out)
    out = self.upsample1(out)
    out = self.upsample2(out)
    out = self.conv2(out)
    return out
