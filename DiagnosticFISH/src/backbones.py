import torch
import einops
import warnings
import torch.nn as nn
from warnings import warn
import torch.nn.functional as F
#from mmselfsup.models.builder import BACKBONES

from typing import List, Optional, Callable
   
#*************************************************************************************************************************************************************************************#

class ConvNormActivation(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        #_log_api_usage_once(self)
        self.out_channels = out_channels

#*************************************************************************************************************************************************************************************#

class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    #@timeit(debug=DEBUG)
    def forward(self, x):
        return x.permute(self.dims)

#*************************************************************************************************************************************************************************************#

class ChannelScaler(nn.Module):
    
    def __init__(self, groups, scale):
        super().__init__()
        
        self.groups = groups
        self.scales = nn.Parameter(torch.ones(groups, 1, 1, 1) * scale)
        
    #@timeit(debug=DEBUG)
    def forward(self, input):
        
        X = input.reshape(input.shape[0], self.groups, input.shape[1]//self.groups, *input.shape[2:])
        
        return (X * self.scales).reshape(input.shape)

#*************************************************************************************************************************************************************************************#

class BlockSetting():
    def __init__(
        self,
        input_channels: int,
        out_channels: int,
        expansion: int,
        num_layers: int,
        downscale: bool,
    ) -> None:
        
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.num_layers = num_layers
        self.downscale = downscale

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", expansion={expansion}"
        s += ", num_layers={num_layers}"
        s += ", downscale={downscale}"

        s += ")"
        return s.format(**self.__dict__)
    
#*************************************************************************************************************************************************************************************#


def create_convneXt(       
    in_channels,
    features,
    initial_expansion,
    expansions,
    block_expansion,
    num_layers):

    block_settings = []
    prev_expansion = 1
    
    expanded_channels = in_channels*initial_expansion
    
    for expansion in expansions:
        
        block_settings.append(BlockSetting(expanded_channels*prev_expansion, expanded_channels*expansion, block_expansion, num_layers, True))
        
        prev_expansion = expansion
        
    block_settings.append(BlockSetting(expanded_channels*prev_expansion, features, block_expansion, num_layers, False))

    return ConvNeXt(
        in_channels=in_channels,
        initial_expansion=initial_expansion,
        block_settings=block_settings
        )

class GroupedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        groups=1,
        norm=False
    ) -> None:
        super().__init__()

        assert in_features % groups == 0
        assert out_features % groups == 0

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.norm = norm
        
        self._linear_layers = nn.ModuleList()
        
        for _ in range(groups):
            layers = []
            if norm:
                layers.append(nn.LayerNorm(in_features // groups, eps=1e-6))
            layers.append(nn.Linear(in_features // groups, out_features // groups, bias=bias))
            self._linear_layers.append(nn.Sequential(*layers))
            
    #@timeit(debug=DEBUG)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        x = input.reshape(self.groups, *input.shape[1:], input.shape[0]//self.groups)
        
        return torch.cat(
            [linear_layer(x_) for x_, linear_layer in zip(x, self._linear_layers)],
            dim=-1
        ).permute([3, 0, 1, 2])
        
    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "in_features={in_features}"
        s += ", out_features={out_features}"
        s += ", groups={groups}"
        s += ", norm={norm}"
        s += ")"
        return s.format(**self.__dict__)

class ConvNeXtBlock(nn.Module):

    def __init__(
        self, 
        in_channels, 
        expansion, 
        scale,
        groups
    ):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, stride=1, groups=groups, bias=True),
            Permute([1, 0, 2, 3]),
            GroupedLinear(in_channels, in_channels*expansion, groups=groups, norm=False, bias=True),
            nn.GELU(),
            GroupedLinear(in_channels*expansion, in_channels, groups=groups, norm=False, bias=True),
            Permute([1, 0, 2, 3])
        )
        
        self.channel_scale = ChannelScaler(in_channels, scale)
    
    #@timeit(debug=DEBUG)
    def forward(self, input):
        
        X_skip = input
        X = self.block(input)
        X = self.channel_scale(X)

        return X + X_skip
    

class ConvNeXt(nn.Module):
    
    def __init__(
        self, 
        in_channels: int,
        initial_expansion: int,
        block_settings: List[BlockSetting],
        block: Optional[Callable[..., nn.Module]] = ConvNeXtBlock,
        groups: Optional[int] = 1
    ) -> None:
        
        super().__init__()
        
        self.in_channels = in_channels
        self.initial_expansion = initial_expansion
        
        if block is None:
            block = ConvNeXtBlock
        
        layers: List[nn.Module] = []
            
        layers.append(
            ConvNormActivation(
                in_channels, 
                self.in_channels * self.initial_expansion, 
                kernel_size=7, 
                padding=3, 
                groups=in_channels, 
                norm_layer=None,
                activation_layer=None,
                bias=False
            )
        )
        
        for config in block_settings:
            print(config)
            stage_ = []
            for _ in range(config.num_layers):
                stage_.append(
                    block(
                        in_channels=config.input_channels, 
                        expansion=config.expansion, 
                        groups=groups, 
                        scale=1e-6
                    )
                )
            
            layers.append(nn.Sequential(*stage_))
            
            layers.append(
                nn.Conv2d(
                    config.input_channels, 
                    config.out_channels, 
                    kernel_size=2 if config.downscale else 1,   # down scaling with kernel_size = 2 and stride = 2
                    stride=2 if config.downscale else 1,        # no down scaling with kernel_size = 1 and stride = 1, its just to reduce dimension
                    groups=in_channels, 
                    bias=False)
            )

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.ReLU()) #activations only between 0 and 1
    
        self.backbone = nn.Sequential(*layers)
        
        self.feature_size = block_settings[-1].out_channels
        
    #@timeit(debug=DEBUG)
    def forward(self, input):
                
        return self.backbone(input)

def ConvNeXt_tiny(in_channels):
    
    features = 256
    initial_expansion = 32
    expansions = [2, 4]
    block_expansion = 4
    num_layers = 1
    
    return create_convneXt(
        in_channels,
        features,
        initial_expansion,
        expansions,
        block_expansion,
        num_layers)