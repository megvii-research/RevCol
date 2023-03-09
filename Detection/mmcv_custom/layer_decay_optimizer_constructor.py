# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import re
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.runner import get_dist_info
import numpy as np

def cal_model_depth(depth, num_subnet):
    dp = np.zeros((depth, num_subnet))
    dp[:,0]=np.linspace(0, depth-1, depth)
    dp[0,:]=np.linspace(0, num_subnet-1, num_subnet)
    for i in range(1, depth):
        for j in range(1, num_subnet):
            dp[i][j] = min(dp[i][j-1], dp[i-1][j])+1
    dp = dp.astype(int)
    # col = [x for x in np.linspace(0, sum(self.layers)-1, sum(self.layers))]
    # dp = np.transpose(np.array([col]*self.num_subnet, dtype=int))
    dp = dp+1 ## make layer id starts from 1
    return dp

def get_num_layer_layer_wise(n, layers, num_subnet=12):
    dp=cal_model_depth(sum(layers), num_subnet)
    # def get_layer_id(n, dp, layers):
    if n.startswith("backbone.subnet"):
        n=n[9:]
        name_part = n.split('.')
        subnet = int(name_part[0][6:])
        if name_part[1].startswith("alpha"):
            id = dp[0][subnet]
        else:
            level = int(name_part[1][-1])
            if name_part[2].startswith("blocks"):
                sub = int(name_part[3])
                if sub>layers[level]-1:
                    sub = layers[level]-1
                block = sum(layers[:level])+sub

            if name_part[2].startswith("fusion"):
                block = sum(layers[:level])
            id = dp[block][subnet]
    elif n.startswith("backbone.stem"):
        id = 0
    else:
        id = dp[-1][-1]+1
    return id

        

@OPTIMIZER_BUILDERS.register_module()
class LearningRateDecayOptimizerConstructor(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}
        print(self.paramwise_cfg)
        num_layers = cal_model_depth(sum(self.paramwise_cfg.get('layers')), self.paramwise_cfg.get('num_subnet'))[-1][-1]+2
        # num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', "layer_wise")
        print("Build LearningRateDecayOptimizerConstructor %s %f - %d" % (decay_type, decay_rate, num_layers))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in ('pos_embed', 'cls_token') or  re.match('(.*).alpha.$', name):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            if decay_type == "layer_wise":
                layer_id = get_num_layer_layer_wise(name, self.paramwise_cfg.get('layers'), self.paramwise_cfg.get('num_subnet'))
                
            group_name = "layer_%d_%s" % (layer_id, group_name)

            if group_name not in parameter_groups:
                scale = decay_rate ** (num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "param_names": [], 
                    "lr_scale": scale, 
                    "group_name": group_name, 
                    "lr": scale * self.base_lr, 
                }

            parameter_groups[group_name]["params"].append(param)
            parameter_groups[group_name]["param_names"].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    "param_names": parameter_groups[key]["param_names"], 
                    "lr_scale": parameter_groups[key]["lr_scale"], 
                    "lr": parameter_groups[key]["lr"], 
                    "weight_decay": parameter_groups[key]["weight_decay"], 
                }
            print("Param groups = %s" % json.dumps(to_display, indent=2))
        
        params.extend(parameter_groups.values())