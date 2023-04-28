# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import torch
from models.revcol import *
from models.revcol_huge import revcol_huge


def build_model(config):
    model_type = config.MODEL.TYPE    

    ##-------------------------------------- revcol tiny ----------------------------------------------------------------------------------------------------------------------#
  
    if model_type == "revcol_tiny":
        model = revcol_tiny(save_memory=config.REVCOL.SAVEMM, inter_supv=config.REVCOL.INTER_SUPV, drop_path = config.REVCOL.DROP_PATH, num_classes=config.MODEL.NUM_CLASSES, kernel_size = config.REVCOL.KERNEL_SIZE)
    
    ##-------------------------------------- revcol small ----------------------------------------------------------------------------------------------------------------------#
    
    elif model_type == "revcol_small":
        model = revcol_small(save_memory=config.REVCOL.SAVEMM, inter_supv=config.REVCOL.INTER_SUPV, drop_path = config.REVCOL.DROP_PATH, num_classes=config.MODEL.NUM_CLASSES, kernel_size = config.REVCOL.KERNEL_SIZE)

    ##-------------------------------------- revcol base ----------------------------------------------------------------------------------------------------------------------#
    
    elif model_type == "revcol_base":
        model = revcol_base(save_memory=config.REVCOL.SAVEMM, inter_supv=config.REVCOL.INTER_SUPV, drop_path = config.REVCOL.DROP_PATH, num_classes=config.MODEL.NUM_CLASSES  ,  kernel_size = config.REVCOL.KERNEL_SIZE)

    ##-------------------------------------- revcol large ----------------------------------------------------------------------------------------------------------------------#

    elif model_type == "revcol_large":
        model = revcol_large(save_memory=config.REVCOL.SAVEMM, inter_supv=config.REVCOL.INTER_SUPV, drop_path = config.REVCOL.DROP_PATH, num_classes=config.MODEL.NUM_CLASSES , head_init_scale=config.REVCOL.HEAD_INIT_SCALE,  kernel_size = config.REVCOL.KERNEL_SIZE)
    
    ##-------------------------------------- revcol xlarge ----------------------------------------------------------------------------------------------------------------------#

    elif model_type == "revcol_xlarge":
        model = revcol_xlarge(save_memory=config.REVCOL.SAVEMM, inter_supv=config.REVCOL.INTER_SUPV, drop_path = config.REVCOL.DROP_PATH, num_classes=config.MODEL.NUM_CLASSES , head_init_scale=config.REVCOL.HEAD_INIT_SCALE,  kernel_size = config.REVCOL.KERNEL_SIZE)
    
    elif model_type == "revcol_huge":
        model = revcol_huge(save_memory=config.REVCOL.SAVEMM, inter_supv=config.REVCOL.INTER_SUPV, drop_path = config.REVCOL.DROP_PATH, num_classes=config.MODEL.NUM_CLASSES , head_init_scale=config.REVCOL.HEAD_INIT_SCALE,  kernel_size = config.REVCOL.KERNEL_SIZE)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model




