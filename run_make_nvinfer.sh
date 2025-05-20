#!/bin/bash

NVINFER_FILE=""
ONNX_FILENAME=""
CLASSES=()
RES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nvinfer-file)
            NVINFER_FILE="$2"
            shift 2
            ;;
        --onnx-filename)
            ONNX_FILENAME="$2"
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

CLASSES=$(IFS=';' ; echo "${CLASSES[*]}")
RES=$(IFS=';' ; echo "${RES[*]}")

echo "[property]
onnx-file=$ONNX_FILENAME

# model config
infer-dims=3;$RES

[custom]
operate-on-class-names=forklift
labels=$CLASSES
" > "$NVINFER_FILE"
