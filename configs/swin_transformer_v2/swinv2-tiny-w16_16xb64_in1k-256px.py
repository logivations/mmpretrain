_base_ = [
    '../_base_/models/swin_transformer_v2/tiny_256_custom.py',
    '../_base_/datasets/imagenet_bs64_swin_256_custom.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

model = dict(backbone=dict(window_size=[16, 16, 16, 8]))
