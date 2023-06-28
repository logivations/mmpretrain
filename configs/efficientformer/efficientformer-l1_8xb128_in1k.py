_base_ = [
    '../_base_/models/efficientformer-l1_custom_128.py',
    '../_base_/datasets/imagenet_bs128_poolformer_small_224_custom.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]
