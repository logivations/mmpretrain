#!/bin/bash

MODEL_DIR=""
CLASSES=()
RES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --classes)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                CLASSES+=("$1")
                shift
            done
            ;;
        --res)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                RES+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

MODEL_DIR=${MODEL_DIR%%/}
CLASSES=$(IFS=';' ; echo "${CLASSES[*]}")
RES=$(IFS=';' ; echo "${RES[*]}")

echo "[property]
onnx-file=end2end.onnx

# model config
infer-dims=3;128;128

[custom]
res=$RES
operate-on-class-names=forklift
labels=$CLASSES
" > "$MODEL_DIR/classifier-config.txt"
