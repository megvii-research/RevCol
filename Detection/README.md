# COCO Object detection with RevCol

## Getting started 

We build RevCol object detection model based on [mmdetection](https://github.com/open-mmlab/mmdetection/tree/3e2693151add9b5d6db99b944da020cba837266b) commit `3e26931`. We add RevCol model and config files to [the original repo](https://github.com/open-mmlab/mmdetection/tree/3e2693151add9b5d6db99b944da020cba837266b). Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/3e2693151add9b5d6db99b944da020cba837266b/docs/en/get_started.md) for installation and dataset preparation instructions.


## Results and Fine-tuned Models

All checkpoints are uploaded to [huggingface](https://huggingface.co/LarryTsai/RevCol). 

| name | Pretrained Model | Method | Lr Schd | box mAP | mask mAP | #params | FLOPs | Fine-tuned Model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:| :---:|
| RevCol-T | [ImageNet-1K](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/classification/revcol_tiny_1k.pth) | Cascade Mask R-CNN | 3x | 50.8 | 44.0 | 88M | 741G | [model](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/detection/cmr_tiny_3x_in1k_AP508.pth) |
| RevCol-S | [ImageNet-1K](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/classification/revcol_small_1k.pth) | Cascade Mask R-CNN | 3x | 52.6 | 45.5 | 118M | 833G | [model](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/detection/cmr_revcol_small_3x_in1k.pth) |
| RevCol-B | [ImageNet-1K](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/classification/revcol_base_1k.pth) | Cascade Mask R-CNN | 3x | 53.0 | 45.9 | 196M | 988G | [model](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/detection/cmr_base_3x_in1k.pth) |
| RevCol-B | [ImageNet-22K](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/classification/revcol_base_22k.pth) | Cascade Mask R-CNN | 3x | 55.0 | 47.5 | 196M | 988G | [model](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/detection/cmr_base_3x_in22k_AP55.pth) |
| RevCol-L | [ImageNet-22K](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/classification/revcol_large_22k.pth) | Cascade Mask R-CNN | 3x | 55.9 | 48.4 | 330M | 1453G | [model](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/detection/cmr_large_3x_in22k_AP559.pth) |



## Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [other optional arguments] 
```
For example, to train a Cascade Mask R-CNN model with a `RevCol-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/revcol/cascade_mask_rcnn_revcol_tiny_3x_in1k.py 8 --cfg-options pretrained=<PRETRAIN_MODEL>
```

More config files can be found at [`configs/revcol`](configs/revcol).

## Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

## Acknowledgment 

This code is built using [mmdetection](https://github.com/open-mmlab/mmdetection), [timm](https://github.com/rwightman/pytorch-image-models) libraries, and [BeiT](https://github.com/microsoft/unilm/tree/f8f3df80c65eb5e5fc6d6d3c9bd3137621795d1e/beit), [Swin Transformer](https://github.com/microsoft/Swin-Transformer) repositories.