import torch
import torchvision
import torch.nn as nn
from torchvision.ops.stochastic_depth import StochasticDepth

#ConvNext Block
#utilising the pytorch equivalent implementation of the code as it runs slightly quikcer
#(2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
class ConvNext_Block(nn.Module):
  def __init__(self, in_channels, stochastic_depth_prob= 0.0, layer_scale_init_value = 1e-6):
    super().__init__()

    self.dw_conv = nn.Conv2d(in_channels = in_channels,
                             out_channels = in_channels, #same number of channels in as out
                             kernel_size = 7,
                             padding = 3,
                             groups = in_channels, #hyperparameter that can be change based on version of ConvNext used
                             bias = True)

     # depthwise convolution, no_groups = no_channels in input

        #layer_norm is done over the final dimension if an integer is given , so permute in order to do normilisations over channel dimension ]
    self.layer_norm  = nn.LayerNorm(normalized_shape = in_channels, eps=1e-6)
    self.conv1 = nn.Linear(in_channels, (in_channels*4) ) # pointwise conv 1x1 applied using linear layers
                     #1x1 convolutional layer , following the inverted bottleneck design
    self.gelu = nn.GELU()
    self.conv2 = nn.Linear(in_channels*4, in_channels)

    self.stochastic_depth_prob = stochastic_depth_prob
    self.layer_scale = nn.Parameter(torch.ones((in_channels))*layer_scale_init_value, requires_grad = True)
    self.stochastic_depth_rate  = StochasticDepth(p = self.stochastic_depth_prob, mode ='row')

  def forward(self, x):
    residual =  x # skip connection
    x = self.dw_conv(x)
    x = x.permute(0, 2, 3, 1)# tensor shape now (N,H,W,C)
    x = self.layer_norm(x)
    x = self.conv1(x)
    x = self.gelu(x)
    x = self.conv2(x)
    x = self.layer_scale*x
    x = self.stochastic_depth_rate(x)
    x = x.permute(0, 3, 1, 2)# tensor shape now (N,C,H,W)

    x += residual

    return x

#Downsampler 
class DownSampler(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.layer_norm = nn.LayerNorm(normalized_shape = in_channels, eps = 1e-6)
    self.conv_layer = nn.Conv2d(in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size=2,
                                stride=2)


  def forward(self,x):
    x = x.permute(0,2,3,1)
    x = self.layer_norm(x)
    x = x.permute(0,3,2,1)

    return self.conv_layer(x)
  

#stem of network

class ConvNext_stem(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv_block  = nn.Conv2d(in_channels = in_channels,
                                 out_channels = out_channels,
                                 kernel_size = 4,
                                 stride = 4)

    self.layer_norm = nn.LayerNorm(normalized_shape = out_channels, eps=1e-6)

  def forward(self,x):
    x = self.conv_block(x)
    x = x.permute(0,2,3,1)
    x = self.layer_norm(x)
    x = x.permute(0,3,1,2)

    return x
  

### Building the model

class ConvNext(nn.Module):
  def __init__(self,
              in_channels = 5,
              channels = [96,192,384,768],
              num_blocks = [3,3,9,3],
              stochastic_depth_prob =  0.1, #from table5 in paper based convnext tiny
              num_classes = 32,
              out_channels = 3,
              head_init_scale = 1):

    super().__init__()
    self.stem =  ConvNext_stem(in_channels = in_channels,
                               out_channels = channels[0])

    self.res2 = self._make_layer(in_channels = channels[0], number_of_blocks = num_blocks[0], stochastic_depth_prob = stochastic_depth_prob)
    self.downsample_2_3 = DownSampler( in_channels = channels[0], out_channels = channels[1])
    self.res3 = self._make_layer(in_channels = channels[1], number_of_blocks = num_blocks[1],stochastic_depth_prob = stochastic_depth_prob)
    self.downsample_3_4 = DownSampler(in_channels = channels[1], out_channels= channels[2])
    self.res4 = self._make_layer(in_channels = channels[2], number_of_blocks = num_blocks[2], stochastic_depth_prob = stochastic_depth_prob)
    self.downsample_4_5 = DownSampler(in_channels = channels[2], out_channels = channels[3])
    self.res5 = self._make_layer(in_channels = channels[3], number_of_blocks = num_blocks[3], stochastic_depth_prob = stochastic_depth_prob)

    self.global_pool = nn.AdaptiveAvgPool2d(1) # global average pooling as implemented in the pytorch implementation
    #permute
    self.final_layernorm = nn.LayerNorm(normalized_shape = channels[-1], eps=1e-6)
    #permute
    self.head = nn.Linear(in_features = (channels[-1]), out_features = out_channels)

    self.apply(self._init_weights)
    self.head.weight.data.mul_(head_init_scale)
    self.head.bias.data.mul_(head_init_scale)

  def _init_weights(self, m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
      nn.init.trunc_normal_(m.weight, std=.02)
      nn.init.constant_(m.bias, 0)

  def _make_layer(self, in_channels, number_of_blocks, stochastic_depth_prob):
    layers = []
    for i in range(number_of_blocks):
      layers.append(ConvNext_Block(in_channels = in_channels, stochastic_depth_prob=stochastic_depth_prob))
      return nn.Sequential(*layers)

  def forward(self, x):
    x = self.stem(x)
    x = self.res2(x)
    x = self.downsample_2_3(x)
    x = self.res3(x)
    x = self.downsample_3_4(x)
    x = self.res4(x)
    x = self.downsample_4_5(x)
    x = self.res5(x)
    x = self.global_pool(x)
    x = x.permute(0, 2, 3, 1)
    x = self.final_layernorm(x)
    x = x.permute(0, 3, 1 , 2)
    #print(x.shape)
    # shape after permute is torch.Size([80, 768, 1, 1])
    x = x.flatten(start_dim=1)
    #print(x.shape)
    #flatten (in the pytorch implementation, the tensor is flattened starting from dim[1])
    #then fully connected
    x = self.head(x)

    return x



