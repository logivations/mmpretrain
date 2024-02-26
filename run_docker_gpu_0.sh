docker run \
    --gpus '"device=0"' \
    --shm-size=8g \
    --rm \
    -it \
    -v /data:/data \
    -w /data/mmpretrain \
    mmpretrain
