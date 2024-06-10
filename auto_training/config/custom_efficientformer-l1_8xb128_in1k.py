RES = 128
auto_scale_lr = dict(base_batch_size=1024)
bgr_mean = [
    103.53,
    116.28,
    123.675,
]
bgr_std = [
    57.375,
    57.12,
    58.395,
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=3,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'AutoDataset'
default_hooks = dict(
    checkpoint=dict(interval=100, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=True, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='l1',
        drop_path_rate=0,
        init_cfg=[
            dict(
                bias=0.0,
                layer=[
                    'Conv2d',
                    'Linear',
                ],
                std=0.02,
                type='TruncNormal'),
            dict(bias=0.0, layer=[
                'GroupNorm',
            ], type='Constant', val=1.0),
            dict(layer=[
                'LayerScale',
            ], type='Constant', val=1e-05),
        ],
        resolution=4,
        type='EfficientFormer'),
    head=dict(
        distillation=False,
        in_channels=448,
        num_classes=3,
        type='EfficientFormerClsHead'),
    init_cfg=dict(
        checkpoint=
        'https://download.openmmlab.com/mmclassification/v0/efficientformer/efficientformer-l1_3rdparty_in1k_20220803-d66e61df.pth',
        type='Pretrained'),
    neck=dict(dim=1, type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-08,
        lr=0.0001,
        type='AdamW',
        weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
        flat_decay_mult=0.0,
        norm_decay_mult=0.0))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=10,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=10,
        by_epoch=True,
        end=100,
        eta_min=1e-05,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_prefix='test',
        data_root='/data/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                crop_ratio_range=(
                    1,
                    1,
                ),
                interpolation='bicubic',
                scale=128,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type=dataset_type),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(prefix='test', topk=(1, ), type='Accuracy'),
    dict(prefix='test', type='AveragePrecision'),
    dict(prefix='test', type='SingleLabelMetric'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        crop_ratio_range=(
            1,
            1,
        ),
        interpolation='bicubic',
        scale=128,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=10)
train_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_prefix='train',
        data_root='/data/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                crop_ratio_range=(
                    0.3,
                    1,
                ),
                interpolation='bicubic',
                scale=128,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(magnitude_range=(
                0,
                0.3,
            ), type='Shear'),
            dict(angle=360, prob=0.8, type='Rotate'),
            dict(
                erase_prob=0.25,
                fill_color=[
                    103.53,
                    116.28,
                    123.675,
                ],
                fill_std=[
                    57.375,
                    57.12,
                    58.395,
                ],
                max_area_ratio=0.3333333333333333,
                min_area_ratio=0.02,
                mode='rand',
                type='RandomErasing'),
            dict(type='PackInputs'),
        ],
        type=dataset_type),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        crop_ratio_range=(
            0.3,
            1,
        ),
        interpolation='bicubic',
        scale=128,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(magnitude_range=(
        0,
        0.3,
    ), type='Shear'),
    dict(angle=360, prob=0.8, type='Rotate'),
    dict(
        erase_prob=0.25,
        fill_color=[
            103.53,
            116.28,
            123.675,
        ],
        fill_std=[
            57.375,
            57.12,
            58.395,
        ],
        max_area_ratio=0.3333333333333333,
        min_area_ratio=0.02,
        mode='rand',
        type='RandomErasing'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_prefix='val',
        data_root='/data/dataset/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                crop_ratio_range=(
                    1,
                    1,
                ),
                interpolation='bicubic',
                scale=128,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        type=dataset_type),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(prefix='val', topk=(1, ), type='Accuracy'),
    dict(prefix='val', type='AveragePrecision'),
    dict(prefix='val', type='SingleLabelMetric'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
