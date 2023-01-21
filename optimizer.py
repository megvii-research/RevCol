# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import numpy as np
from torch import optim as optim

def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    elif config.MODEL.TYPE.startswith("revcol"):
        parameters = param_groups_lrd(model, weight_decay=config.TRAIN.WEIGHT_DECAY, no_weight_decay_list=[], layer_decay=config.TRAIN.OPTIMIZER.LAYER_DECAY)
    else:
        parameters = set_weight_decay(model, skip, skip_keywords)


    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR)
    
    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad or name in ["linear_eval.weight", "linear_eval.bias"]:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def cal_model_depth(columns, layers):
    depth = sum(layers)
    dp = np.zeros((depth, columns))
    dp[:,0]=np.linspace(0, depth-1, depth)
    dp[0,:]=np.linspace(0, columns-1, columns)
    for i in range(1, depth):
        for j in range(1, columns):
            dp[i][j] = min(dp[i][j-1], dp[i-1][j])+1
    dp = dp.astype(int)
    return dp


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    dp = cal_model_depth(model.num_subnet, model.layers)+1
    num_layers = dp[-1][-1] + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:# or re.match('(.*).alpha.$', n):
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id(n, dp, model.layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    
    return list(param_groups.values())

def get_layer_id(n, dp, layers):
    if n.startswith("subnet"):
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
    elif n.startswith("stem"):
        id = 0
    else:
        id = dp[-1][-1]+1
    return id