# ADE20k Semantic segmentation with RevCol

## Getting started 

We add RevCol model and config files to [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/4eaa8e69191cc293b64dafe47f1f88a7d468c93c).
Our code has been tested with commit [4eaa8e6](https://github.com/open-mmlab/mmsegmentation/tree/4eaa8e69191cc293b64dafe47f1f88a7d468c93c). Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/4eaa8e69191cc293b64dafe47f1f88a7d468c93c/docs/en/get_started.md#installation) for installation and dataset preparation instructions.

## Results and Fine-tuned Models

| name | Pretrained Model | Method | Crop Size | Lr Schd | mIoU | mIoU (ms+flip) | #params | FLOPs | Fine-tuned Model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:| :---:|:---:|
| RevCol-T | [ImageNet-1K](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/classification/revcol_tiny_1k.pth) | UPerNet | 512x512 | 160K | 47.4 | 47.6 | 60M | 937G | [model](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/segmentation/upernet_revcol_tiny_fp16_512x512_160k_ade20k.pth) |
| RevCol-S | [ImageNet-1K](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/classification/revcol_small_1k.pth) | UPerNet | 512x512 | 160K | 47.9 | 49.0 | 90M | 1031G | [model](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/segmentation/upernet_revcol_small_fp16_512x512_160k_ade20k.pth) |
| RevCol-B | [ImageNet-1K](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/classification/revcol_base_1k.pth) | UPerNet | 512x512 | 160K | 49.0 | 50.1 | 122M | 1169G | [model](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/segmentation/upernet_revcol_base_512x512_160k_ade20k.pth) |
| RevCol-B | [ImageNet-22K](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/classification/revcol_base_22k.pth) | UPerNet | 640x640 | 160K | 52.7 | 53.3 | 122M | 1827G | [model](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/segmentation/upernet_revcol_base_640x640_160k_ade20k.pth) |
| RevCol-L | [ImageNet-22K](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/classification/revcol_large_22k.pth) | UPerNet | 640x640 | 160K | 53.4 | 53.7 | 306M | 2610G | [model](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/segmentation/upernet_revcol_large_fp16_640x640_160k_ade20k.pth) |
| RevCol-H | [MegData-168M](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/segmentation/revcol_huge_k7padded.pth) (Pad conv kernel to 7x7 after pretraining) | UPerNet | 640x640 | 160K | 57.8 | 58.0 | 2421M | -    | [model](https://huggingface.co/LarryTsai/RevCol/blob/main/revcol_models/segmentation/upernet_revcol_huge_fp16_640x640_160k_ade20k.pth) |

### Training

Command format:
```
tools/dist_train.sh <CONFIG_PATH> <NUM_GPUS> --work-dir <SAVE_PATH> --seed 0 --deterministic --options model.pretrained=<PRETRAIN_MODEL>
```

For example, using a `RevCol-T` backbone with UperNet:
```bash
bash tools/dist_train.sh \
    configs/revcol/upernet_revcol_tiny_fp16_512x512_160k_ade20k.py 8 \
    --work-dir /path/to/save --seed 0 --deterministic \
    --options model.pretrained=<PRETRAIN_MODEL>
```

More config files can be found at [`configs/RevCol`](configs/revcol).


## Evaluation

Command format for multi-scale testing:
```
tools/dist_test.sh <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU --aug-test
```

For example, evaluate a `RevCol-T` backbone with UperNet (multi-scale):
```bash
bash tools/dist_test.sh configs/RevCol/upernet_revcol_tiny_fp16_512x512_160k_ade20k.py \ 
    <CHECKPOINT_PATH> 4 --eval mIoU --aug-test
```

Command format for single-scale testing:
```
tools/dist_test.sh <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```

For example, evaluate a `RevCol-T` backbone with UperNet:
```bash
bash tools/dist_test.sh configs/RevCol/upernet_revcol_tiny_fp16_512x512_160k_ade20k.py \ 
    <CHECKPOINT_PATH> 4 --eval mIoU 
```

## Acknowledgment 

This code is built using [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [timm](https://github.com/huggingface/pytorch-image-models) repositories.
