import torch
import torch.nn as nn

class ResBlock(nn.Module):
  def __init__(self,in_channels,kernel_size_1,kernel_size_2,channel_1,channel_2,stride_1,stride_2):
    super(ResBlock,self).__init__()
    ### initializing sub layers ###
    self.conv1 = nn.Conv1d(in_channels=in_channels,out_channels=channel_1,\
                           kernel_size=kernel_size_1,stride=stride_1,\
                           padding=int((kernel_size_1-1)/2))
    self.conv_gated1 = nn.Conv1d(in_channels=in_channels,out_channels=channel_1,\
                           kernel_size=kernel_size_1,stride=stride_1,\
                           padding=int((kernel_size_1-1)/2))
    self.instancenorm1 = nn.InstanceNorm1d(channel_1,affine=True)
    self.instancenorm_gated1 = nn.InstanceNorm1d(channel_1,affine=True)
    self.conv2 = nn.Conv1d(in_channels=channel_1,out_channels=channel_2,\
                           kernel_size=kernel_size_2,stride=stride_2,\
                           padding=int((kernel_size_2-1)/2))
    self.instancenorm2 = nn.InstanceNorm1d(channel_2,affine=True)

  def forward(self,input):
    x = input
    x = self.conv1(x)
    x = self.instancenorm1(x)
    x_gated = self.conv_gated1(input)
    x_gated = self.instancenorm_gated1(x_gated)
    x = x * nn.Sigmoid()(x_gated)
    x = self.conv2(x)
    x = self.instancenorm2(x)
    output = x+input
    return output

class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // self.upscale_factor
        w_new = input.shape[2] * self.upscale_factor
        return input.view(n, c_out, w_new)

class DownSample_Gen(nn.Module):
  def __init__(self, in_channels, kernel_size, stride, channels):
    super(DownSample_Gen, self).__init__()
    self.conv = nn.Conv1d(in_channels=in_channels,out_channels=channels,\
                        kernel_size=kernel_size,stride=stride,\
                        padding=int((kernel_size-stride)/2))
    self.instancenorm = nn.InstanceNorm1d(channels,affine=True)
    self.conv_gated = nn.Conv1d(in_channels=in_channels,out_channels=channels,\
                        kernel_size=kernel_size,stride=stride,\
                        padding=int((kernel_size-stride)/2))
    self.instancenorm_gated = nn.InstanceNorm1d(channels,affine=True)

  def forward(self,input):
    out = self.conv(input)
    out = self.instancenorm(out)
    out_gated = self.conv_gated(input)
    out_gated = self.instancenorm_gated(out_gated)
    return out*nn.Sigmoid()(out_gated)

class DownSample_Disc(nn.Module):
  def __init__(self, in_channels, kernel_size, stride, channels):
    super(DownSample_Disc, self).__init__()
    self.conv = nn.Conv2d(in_channels=in_channels,out_channels=channels,
                        kernel_size=kernel_size,stride=stride,
                        padding=(int((kernel_size[0]-stride[0])/2),int(((kernel_size[1]-stride[1])/2))))
    self.instancenorm = nn.InstanceNorm2d(channels,affine=True)
    self.conv_gated = nn.Conv2d(in_channels=in_channels,out_channels=channels,
                        kernel_size=kernel_size,stride=stride,
                        padding=(int((kernel_size[0]-stride[0])/2),int(((kernel_size[1]-stride[1])/2))))
    self.instancenorm_gated = nn.InstanceNorm2d(channels,affine=True)


  def forward(self,input):
    out = self.conv(input)
    out = self.instancenorm(out)
    out_gated = self.conv_gated(input)
    out_gated = self.instancenorm_gated(out_gated)
    return out*nn.Sigmoid()(out_gated)

class UpSample(nn.Module):
  def __init__(self, in_channels, kernel_size, stride, channels, upsample_factor):
    super(UpSample, self).__init__()
    self.conv = nn.Conv1d(in_channels=in_channels,out_channels=channels,\
                        kernel_size=kernel_size,stride=stride,\
                        padding=int((kernel_size-stride)/2))
    self.pixelshuffle = PixelShuffle(upsample_factor)
    self.instancenorm = nn.InstanceNorm1d(channels//upsample_factor,affine=True)
    self.conv_gated = nn.Conv1d(in_channels=in_channels,out_channels=channels,\
                        kernel_size=kernel_size,stride=stride,\
                        padding=int((kernel_size-stride)/2))
    self.instancenorm_gated = nn.InstanceNorm1d(channels//upsample_factor,affine=True)

  def forward(self,input):
    out = self.conv(input)
    out = self.pixelshuffle(out)
    out = self.instancenorm(out)
    out_gated = self.conv_gated(input)
    out_gated = self.pixelshuffle(out_gated)
    out_gated = self.instancenorm_gated(out_gated)
    return out*nn.Sigmoid()(out_gated)
