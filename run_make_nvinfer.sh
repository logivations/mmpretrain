#!/bin/bash

MODEL_DIR=$1
MODEL_DIR=${MODEL_DIR%%/}
shift

CLASSES=( "$@" )
CLASSES=$(IFS=';' ; echo "${CLASSES[*]}")

echo "[property]
gpu-id=0

# preprocessing parameters.
net-scale-factor=0.01742919389
offsets=123.675;116.128;103.53
model-color-format=0
scaling-filter=1 # 0=Nearest, 1=Bilinear

onnx-file=state_classifier.onnx
model-engine-file=state_classifier.onnx_b1_gpu0_fp16.engine

# model config
infer-dims=3;128;128
batch-size=1
network-mode=2 # 0=FP32, 1=INT8, 2=FP16
network-type=100 # >3 disables post-processing
cluster-mode=4 # 1=DBSCAN 4=No Clustering
gie-unique-id=3
process-mode=2 # 1=Primary, 2=Secondary
output-tensor-meta=1
maintain-aspect-ratio=0
tensor-meta-pool-size=200

operate-on-class-ids=0;

[custom]
labels=$CLASSES
" > "$MODEL_DIR/classifier-config.txt"


