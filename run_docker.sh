docker run \
    --gpus all \
    --shm-size=8g \
    --rm \
    -it \
    -v /data:/data \
    -v /data/mmpretrain:/mmpretrain \
    -w /mmpretrain \
    quay.io/logivations/ml_all:LS_mmpret_latest