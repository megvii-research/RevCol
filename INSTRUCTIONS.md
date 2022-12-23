# Installation, Training and Evaluation Instructions for Image Classification

We provide installation, training and evaluation instructions for image classification here.

## Installation Instructions

- Clone this repo:

```bash
git clone https://github.com/megvii-research/RevCol.git
cd RevCol
```

- Create a conda virtual environment and activate it:

```bash
conda create --name revcol python=3.7 -y
conda activate revcol
```

- Install `CUDA>=11.3` with `cudnn>=8` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.11.0` and `torchvision>=0.12.0` with `CUDA>=11.3`:

```bash
conda install pytorch=1.11.0 torchvision=0.12.0 torchaudio=0.11.0 cudatoolkit=11.3 -c pytorch
```

- Install `timm==0.5.4`:

```bash
pip install timm==0.5.4
```

- Install other requirements:

```bash
pip install -r requirements.txt
```

## Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard imagenet-1k dataset, the file structure should look like:
  ```bash
  path-to-imagenet-1k
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```

- For ImageNet-22K dataset, the file structure should look like:

  ```bash
  path-to-imagenet-22k
  ├── class1
  │   ├── img1.jpeg
  │   ├── img2.jpeg
  │   └── ...
  ├── class2
  │   ├── img3.jpeg
  │   └── ...
  └── ...
  ```

- As imagenet-22k has no val set, one way is to use imagenet-1k val set as the evaluation for imagenet 22k dataset. Please remember to map the imagenet-1k label to imagenet-22k.
  ```bash
  path-to-imagenet-22k-custom-eval-set
  ├── class1
  │   ├── img1.jpeg
  │   ├── img2.jpeg
  │   └── ...
  ├── class2
  │   ├── img3.jpeg
  │   └── ...
  └── ...
  ```

## Evaluation

To evaluate a pre-trained `RevCol` on ImageNet validation set, run:

```bash
torchrun --nproc_per_node=<num-of-gpus-to-use> --master_port=23456 main.py --cfg <config-file.yaml> --resume <checkpoint_path> --data-path <imagenet-path> --eval
```

For example, to evaluate the `RevCol-T` with a single GPU:

```bash
torchrun --nproc_per_node=8 --master_port=23456 main.py --cfg configs/revcol_tiny_1k.yaml --resume path_to_your_model.pth --eval
```

## Training from scratch on ImageNet-1K

To train a `RevCol` on ImageNet from scratch, run:

```bash
torchrun --nproc_per_node=<num-of-gpus-to-use> --master_port=23456 main.py  \ 
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

**Notes**:

For example, to train `RevCol` with 8 GPU on a single node for 300 epochs, run:

`RevCol-T`:

```bash
torchrun --nproc_per_node=8 --master_port=23456 main.py --cfg configs/revcol_tiny_1k.yaml --batch-size 128 --data-path <imagenet-path>
```

`RevCol-S`:

```bash
torchrun --nproc_per_node=8 --master_port=23456 main.py --cfg configs/revcol_small_1k.yaml --batch-size 128 --data-path <imagenet-path>
```

`RevCol-B`:

```bash
torchrun --nproc_per_node=8 --master_port=23456 main.py --cfg configs/revcol_base_1k.yaml --batch-size 128 --data-path <imagenet-path>
```

## Pre-training on ImageNet-22K

For example, to pre-train a `RevCol-B` model on ImageNet-22K:

```bash
torchrun --nproc_per_node=8 --master_port=23456 main.py --cfg configs/revcol_large_22k_pretrain.yaml --batch-size 128 --data-path <imagenet-22k-path> --opt DATA.EVAL_DATA_PATH <imagenet-22k-custom-eval-path>
```


## Fine-tuning from a ImageNet-22K(21K) pre-trained model

For example, to fine-tune a `RevCol-B` model pre-trained on ImageNet-22K(21K):

```bashs
torchrun --nproc_per_node=8 --master_port=23456 main.py --cfg configs/revcol_base_1k_384_finetune.yaml --batch-size 64 --data-path <imagenet-22k-path> --finetune revcol_base_22k_pretrained.pth
```

