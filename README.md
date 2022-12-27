# Reversible Column Networks 
This repo is the official implementation of:

### [Reversible Column Networks](https://arxiv.org/abs/2212.11696)
[Yuxuan Cai](https://nightsnack.github.io), [Yizhuang Zhou](https://scholar.google.com/citations?user=VRSGDDEAAAAJ), [Qi Han](https://hanqer.github.io), Jianjian Sun, Xiangwen Kong, Jun Li, [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ) \
[MEGVII Technology](https://en.megvii.com)\
[\[arxiv\]](https://arxiv.org/abs/2212.11696) 



## Updates
***12/23/2022***\
Initial commits: codes for ImageNet-1k and ImageNet-22k classification are released.


## To Do List


- [x] ImageNet-1K and 22k Training Code  
- [ ] ImageNet-1K and 22k Model Weights
- [ ] Cascade Mask R-CNN COCO Object Detection Code & Model Weights
- [ ] ADE20k Semantic Segmentation Code & Model Weights


## Introduction
RevCol is composed of multiple copies of subnetworks, named columns respectively, between which multi-level reversible connections are employed. RevCol coud serves as a foundation model backbone for various tasks in computer vision including classification, detection and segmentation.

<p align="center">
<img src="figures/title.png" width=100% height=100% 
class="center">
</p>

## Main Results on ImageNet with Pre-trained Models

| name | pretrain | resolution | #params |FLOPs | acc@1 | pretrained model | finetuned model |
|:---------------------:| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RevCol-T | ImageNet-1K | 224x224 | 30M | 4.5G | 82.2 | [baidu]()/[github]() | - |
| RevCol-S | ImageNet-1K | 224x224 | 60M | 9.0G | 83.5 | [baidu]()/[github]() | - |
| RevCol-B | ImageNet-1K | 224x224 | 138M | 16.6G | 84.1 |  [baidu]()/[github]() |  - |
| RevCol-B<sup>\*</sup> | ImageNet-22K | 224x224 | 138M | 16.6G | 85.6 |[baidu]()/[github]()| [baidu]()/[github]()|
| RevCol-B<sup>\*</sup> | ImageNet-22K | 384x384 | 138M | 48.9G | 86.7 |[baidu]()/[github]()| [baidu]()/[github]()|
| RevCol-L<sup>\*</sup> | ImageNet-22K | 224x224 | 273M | 39G | 86.6 |[baidu]()/[github]()| [baidu]()/[github]()|
| RevCol-L<sup>\*</sup> | ImageNet-22K | 384x384 | 273M | 116G | 87.6 |[baidu]()/[github]()| [baidu]()/[github]()|

## Getting Started
Please refer to [INSTRUCTIONS.md](INSTRUCTIONS.md) for setting up, training and evaluation details.


## Acknowledgement
This repo was inspired by several open source projects. We are grateful for these excellent projects and list them as follows:
- [timm](https://github.com/rwightman/pytorch-image-models)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- [beit](https://github.com/microsoft/unilm/tree/master/beit)

## License
RevCol is released under the [Apache 2.0 license](LICENSE).

## Contact Us
If you have any questions about this repo or the original paper, please contact Yuxuan at caiyuxuan@megvii.com.


## Citation
```
@article{cai2022reversible,
  title={Reversible Column Networks},
  author={Cai, Yuxuan and Zhou, Yizhuang and Han, Qi and Sun, Jianjian and Kong, Xiangwen and Li, Jun and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2212.11696},
  year={2022}
}
```