import numpy as np
import torch
import torch.nn as nn

from .revcol_module import ConvNextBlock, LayerNorm, UpSampleConvnext
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger

from .revcol_function import ReverseFunction
from mmcv.cnn import constant_init, trunc_normal_init
from mmcv.runner import BaseModule, _load_checkpoint
from torch.utils.checkpoint import checkpoint

class Fusion(nn.Module):
    def __init__(self, level, channels, first_col) -> None:
        super().__init__()
        
        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
                nn.Conv2d(channels[level-1], channels[level], kernel_size=2, stride=2),
                LayerNorm(channels[level], eps=1e-6, data_format="channels_first"),
            ) if level in [1, 2, 3] else nn.Identity()
        if not first_col:
            self.up = UpSampleConvnext(1, channels[level+1], channels[level]) if level in [0, 1, 2] else nn.Identity()
            

    def forward(self, *args):
        c_down, c_up = args
        
        if self.first_col:
            x = self.down(c_down)
            return x
        
        if self.level == 3:
            x = self.down(c_down)
        else:
            x = self.down(c_down)
            shape = x.shape[-2:]
            x = x + self.up(c_up, shape)
        return x

class Level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0) -> None:
        super().__init__()
        countlayer = sum(layers[:level])
        expansion = 4
        self.fusion = Fusion(level, channels, first_col)
        modules = [ConvNextBlock(channels[level], expansion*channels[level], channels[level], kernel_size = kernel_size,  layer_scale_init_value=1e-6, drop_path=dp_rate[countlayer+i]) for i in range(layers[level])]
        self.blocks = nn.Sequential(*modules)
    def forward(self, *args):
        x = self.fusion(*args)
        x = self.blocks(x)
        return x

class SubNet(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_col, dp_rates, save_memory) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 

        self.level0 = Level(0, channels, layers, kernel_size, first_col, dp_rates)

        self.level1 = Level(1, channels, layers, kernel_size, first_col, dp_rates)

        self.level2 = Level(2, channels, layers, kernel_size,first_col, dp_rates)

        self.level3 = Level(3, channels, layers, kernel_size, first_col, dp_rates)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3= args

        c0 = (self.alpha0)*c0 + self.level0(x, c1)
        c1 = (self.alpha1)*c1 + self.level1(c0, c2)
        c2 = (self.alpha2)*c2 + self.level2(c1, c3)
        c3 = (self.alpha3)*c3 + self.level3(c2, None)
        # if dist.get_rank() == 0:
        #     print(f'x mean: {x.mean()}, c0 mean: {c0.mean()}, c1 mean: {c1.mean()}, c2 mean: {c2.mean()}, c3 mean: {c3.mean()}')
        #     print(f'x var: {x.var()}, c0 var: {c0.var()}, c1 var: {c1.var()}, c2 var: {c2.var()}, c3 var: {c3.var()}')
        #     print(f'alpha0: {(self.alpha0).min()}, alpha1: {(self.alpha1).min()}, alpha2: {(self.alpha2).min()}, alpha3: {(self.alpha3).min()}')
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        # if dist.get_rank() == 0:
        #     print(f'c0 mean: {c0.mean()}, c1 mean: {c1.mean()}, c2 mean: {c2.mean()}, c3 mean: {c3.mean()}')
        #     print(f'c0 var: {c0.var()}, c1 var: {c1.var()}, c2 var: {c2.var()}, c3 var: {c3.var()}')
        #     print(f'alpha0: {self.alpha0.max()}, alpha1: {self.alpha1.max()}, alpha2: {self.alpha2.max()}, alpha3: {self.alpha3.max()}')
        return c0, c1, c2, c3

    def forward(self, *args):
        
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        
        #self.alpha0.data.clamp_(1e-3)
        #self.alpha1.data.clamp_(1e-3)
        #self.alpha2.data.clamp_(1e-3)
        #self.alpha3.data.clamp_(1e-3)
        
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign=data.sign()
            data.abs_().clamp_(value)
            data*=sign

@BACKBONES.register_module()
class RevCol(BaseModule):
    def __init__(self, channels=[32, 64, 96, 128], layers=[2, 3, 6, 3], num_subnet=5, kernel_size = 3, num_classes=1000, drop_path = 0.0, save_memory=False, single_head=True, out_indices=[0, 1, 2, 3], init_cfg=None) -> None:
        super().__init__(init_cfg)
        self.num_subnet = num_subnet
        self.single_head = single_head
        self.out_indices = out_indices
        self.init_cfg = init_cfg

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=4, stride=4),
            LayerNorm(channels[0], eps=1e-6, data_format="channels_first")
        )

        # dp_rate = self.cal_dp_rate(sum(layers), num_subnet, drop_path)

        dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(layers))] 
        for i in range(num_subnet):
            first_col = True if i == 0 else False
            self.add_module(f'subnet{str(i)}', SubNet(
                channels,layers, kernel_size, first_col, dp_rates=dp_rate, save_memory=save_memory))

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            

            state_dict = _state_dict
            # print(state_dict.keys())
            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # load state_dict
            self.load_state_dict(state_dict, False)
        

    def forward(self, x):
        x = self.stem(x)        
        c0, c1, c2, c3 = 0, 0, 0, 0
        for i in range(self.num_subnet):
            # c0, c1, c2, c3 = checkpoint(getattr(self, f'subnet{str(i)}'), x, c0, c1, c2, c3 )
            c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2, c3)
        return c0, c1, c2, c3
    
    def cal_dp_rate(self, depth, num_subnet, drop_path):
        dp = np.zeros((depth, num_subnet))
        dp[:,0]=np.linspace(0, depth-1, depth)
        dp[0,:]=np.linspace(0, num_subnet-1, num_subnet)
        for i in range(1, depth):
            for j in range(1, num_subnet):
                dp[i][j] = min(dp[i][j-1], dp[i-1][j])+1
        ratio = dp[-1][-1]/drop_path
        dp_matrix = dp/ratio
        return dp_matrix
