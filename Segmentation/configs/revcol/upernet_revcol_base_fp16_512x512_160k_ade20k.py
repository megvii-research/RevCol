_base_ = [
    '../_base_/models/upernet_revcol.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
checkpoint_file = 'cls_model/revcol_base_1k.pth'  # noqa
model = dict(
    backbone=dict(
        type='RevCol',
        channels=[72, 144, 288, 576],
        layers=[1, 1, 3, 2],
        num_subnet=16,
        drop_path = 0.3, 
        save_memory=False, 
        out_indices=[0, 1, 2, 3],
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        in_channels=[72, 144, 288, 576],
        num_classes=150,
    ),
    auxiliary_head=dict(in_channels=288, num_classes=150),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(426, 426)),
)

optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructorRevCol',
    _delete_=True,
    type='AdamW',
    lr=0.00004,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg={'decay_rate': 1.0,
                'decay_type': 'layer_wise',
                'layers': [1, 1, 3, 2],
                'num_subnet': 16}
    )

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()
