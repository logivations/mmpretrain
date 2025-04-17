#!/bin/bash

MODEL_DIR=$1
MODEL_DIR=${MODEL_DIR%%/}
shift

CLASSES=( "$@" )
CLASSES=$(IFS=';' ; echo "${CLASSES[*]}")

echo "[property]
onnx-file=state_classifier.onnx
model-engine-file=state_classifier.onnx_b1_gpu0_fp16.engine

# model config
infer-dims=3;128;128
gie-unique-id=3

operate-on-class-ids=0;
operated-on-class-name=choice

[custom]
labels=$CLASSES
" > "$MODEL_DIR/classifier-config.txt"


