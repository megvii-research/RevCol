import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

class UpSampleConvnext(nn.Module):
    def __init__(self, ratio, inchannel, outchannel):
        super().__init__()
        self.ratio = ratio
        self.channel_reschedule = nn.Sequential(  
                                        # LayerNorm(inchannel, eps=1e-6, data_format="channels_last"),
                                        nn.Linear(inchannel, outchannel),
                                        LayerNorm(outchannel, eps=1e-6, data_format="channels_last"))
        # self.upsample  = nn.Upsample(scale_factor=2**ratio, mode='nearest')
        
    def forward(self, x, shape):
        x = x.permute(0, 2, 3, 1)
        x = self.channel_reschedule(x)
        x = x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, shape, mode='nearest')
        return x
        
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, in_channel, hidden_dim, out_channel, kernel_size=3, layer_scale_init_value=1e-6, drop_path= 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=in_channel) # depthwise conv
        self.norm = nn.LayerNorm(in_channel, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channel, hidden_dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_channel)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channel)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x