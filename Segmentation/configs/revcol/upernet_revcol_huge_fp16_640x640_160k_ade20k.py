_base_ = [
    '../_base_/models/upernet_revcol_huge.py',
    '../_base_/datasets/ade20k_640x640.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (640, 640)
checkpoint_file = 'cls_model/ckpt_huge_k7_finetune.pth'  # noqa
model = dict(
    backbone=dict(
        type='FullNet',
        channels=[360, 720, 1440, 2880],
        layers=[1, 2, 6, 2],
        num_subnet=8,
        drop_path = 0.3,
        kernel_size=7,
        save_memory=False,
        out_indices=[0, 1, 2, 3],
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        in_channels=[360, 720, 1440, 2880],
        num_classes=150,
        channels = 1440
    ),
    auxiliary_head=dict(in_channels=1440, num_classes=150),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(426, 426)),
)

optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructorGLOM',
    _delete_=True,
    type='AdamW',
    lr=0.00004,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg={'decay_rate': 0.9,
                'decay_type': 'layer_wise',
                'layers': [1, 2, 6, 2],
                'num_subnet': 8}
    )

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=1)
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()
