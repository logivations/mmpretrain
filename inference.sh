python tools/inference.py /path/to/model.pth \
 auto_training/config/custom_efficientformer-l1_8xb128_in1k.py \
 /path/to/images \
 /path/to/out --classes no_vest unknown vest --silent \
  --ann-dir /path/to/raw_annotations/ 
