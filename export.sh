python auto_training/export_model.py \
  auto_training/config/custom_efficientformer-l1_8xb128_in1k.py \
  /data/regr_test/new.pth \
  --output-file /data/regr_test/classifier_fix.onnx \
  --shape 1 3 128 128
