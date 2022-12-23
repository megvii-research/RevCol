# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

import io
import os
import re
from typing import List
from timm.utils.model_ema import ModelEma
import torch
import torch.distributed as dist
from timm.utils import get_state_dict
import subprocess




def load_checkpoint(config, model, optimizer, logger, model_ema=None):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        logger.info("Already loaded checkpoint to memory..")
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if config.MODEL_EMA:
        if 'state_dict_ema' in checkpoint.keys():
            model_ema.ema.load_state_dict(checkpoint['state_dict_ema'], strict=False)

            logger.info("Loaded state_dict_ema")
        else:
            model_ema.ema.load_state_dict(checkpoint['model'], strict=False)
            logger.warning("Failed to find state_dict_ema, starting from loaded model weights")

    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()

        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
    # del checkpoint
    # torch.cuda.empty_cache()
    return max_accuracy

def load_checkpoint_finetune(config, model, logger, model_ema=None):
    logger.info(f"==============> Finetune {config.MODEL.FINETUNE}....................")
    checkpoint = torch.load(config.MODEL.FINETUNE, map_location='cpu')['model']
    converted_weights = {}
    keys = list(checkpoint.keys())
    for key in keys:
        if re.match(r'cls.*', key):
        # if re.match(r'cls.classifier.1.*', key): 
            print(f'key: {key} is used for pretrain, discarded.')
            continue
        else:
            converted_weights[key] = checkpoint[key]
    msg = model.load_state_dict(converted_weights, strict=False)
    logger.info(msg)
    if model_ema is not None:
        ema_msg = model_ema.ema.load_state_dict(converted_weights, strict=False)
        logger.info(f"==============> Loaded Pretraind statedict into EMA....................")
        logger.info(ema_msg)
    del checkpoint
    torch.cuda.empty_cache()
    

def save_checkpoint(config, epoch, model, epoch_accuracy, max_accuracy, optimizer, logger, model_ema=None):
    if model_ema is not None:
        logger.info("Model EMA is not None...")
        save_state = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'max_accuracy': max(max_accuracy, epoch_accuracy),
                    'epoch': epoch,
                    'state_dict_ema': get_state_dict(model_ema),
                    'input': input,
                    'config': config}
    else:
        save_state = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'max_accuracy': max(max_accuracy, epoch_accuracy),
                    'epoch': epoch,
                    'state_dict_ema': None,
                    'input': input,
                    'config': config}
    
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    best_path = os.path.join(config.OUTPUT, f'best.pth')
    
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    if epoch_accuracy>max_accuracy:
        torch.save(save_state, best_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir,logger):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth') and ckpt.startswith('ckpt_')]
    logger.info(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def denormalize(tensor: torch.Tensor, mean: List[float], std: List[float], inplace: bool = False) -> torch.Tensor:
    """Denormalize a float tensor image with mean and standard deviation.
    This transform does not support PIL Image.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Denormalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if not tensor.is_floating_point():
        raise TypeError('Input tensor should be a float tensor. Got {}.'.format(tensor.dtype))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean).clip_(0.0, 1.0)
    return tensor

