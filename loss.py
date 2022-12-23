# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------

from dis import dis
import torch
from torch import nn
import torch.distributed as dist
from torch.functional import Tensor
import torch.nn.functional as F



def compound_loss(coe, output_feature, image:Tensor, output_label, targets, criterion_bce, criterion_ce, epoch):
    f_coe, c_coe = coe
    image.clamp_(0.01, 0.99)
    multi_loss = []
    for i, feature in enumerate(output_feature):
        ratio_f = 1 - i / len(output_feature)
        ratio_c = (i+1) / (len(output_label))

        ihx = criterion_bce(feature, image) * ratio_f * f_coe 
        ihy = criterion_ce(output_label[i], targets) * ratio_c * c_coe
        # if dist.get_rank() == 0:
        #     print(f'ihx: {ihx}, ihy: {ihy}')
        multi_loss.append(ihx + ihy)
        # feature_loss.append(torch.dist(output_feature[i], teacher_feature) *  feature_coe)
    multi_loss.append(criterion_ce(output_label[-1], targets))
    # print(feature_loss)
    loss = torch.sum(torch.stack(multi_loss), dim=0)
    # +torch.mean(torch.stack(feature_loss), dim=0)
    return loss, multi_loss
