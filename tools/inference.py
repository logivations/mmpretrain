from pathlib import Path
from argparse import ArgumentParser
import os
import json
import time

import cv2
import numpy as np
import torch
from mmpretrain import ImageClassificationInferencer
from sklearn.metrics import classification_report

# PyTorch 2.6+ uses weights_only=True by default which breaks mmengine checkpoints.
# Patch torch.load to use weights_only=False (safe for trusted internal checkpoints).
_original_torch_load = torch.load
def _torch_load_weights_only_false(f, *args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(f, *args, **kwargs)
torch.load = _torch_load_weights_only_false


def _draw_prediction(img_path, pred_class, score, out_path):
    """Draw the predicted class onto the image and save it (single-image predict)."""
    img = cv2.imread(img_path)
    if img is None:
        return
    bar_h = max(30, img.shape[0] // 20)
    bar = np.zeros((bar_h, img.shape[1], 3), dtype=np.uint8)
    out_img = np.vstack((bar, img))
    cv2.putText(
        out_img,
        f"{pred_class} ({score:.2f})",
        (5, int(bar_h * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        bar_h / 45,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out_img)


def main(args):
    inference = ImageClassificationInferencer(
        model=args.config,
        pretrained=args.checkpoint,
        classes=args.classes,
    )
    print(f"Inference classes: {args.classes}")
    if args.silent:
        inference.show_progress = False

    images: Path = args.images_dir

    # The annotation file is keyed by image name; its keys decide WHICH images to
    # process (value is the GT class, or None when the task has no annotation).
    # Fall back to globbing the folder when no annotation file is provided.
    labels = {}
    if args.ann_file and os.path.exists(args.ann_file):
        with open(args.ann_file) as f:
            labels = json.load(f)
    image_names = list(labels.keys()) if labels else [p.name for p in images.glob("**/*.jpg")]

    predictions = {}
    y_true, y_pred = [], []
    start_time = time.time()

    for image_name in image_names:
        img_path = os.path.join(images, image_name)
        if not os.path.exists(img_path):
            print(f"Image listed in annotations is missing, skipping: {img_path}")
            continue
        try:
            result = inference(img_path)[0]
            pred_class = result["pred_class"]
            score = float(result.get("pred_score", 1.0))

            predictions[image_name] = {
                "result": [
                    {
                        "type": "choices",
                        "value": {"choices": [pred_class]},
                        "score": score,
                        "origin": "manual",
                        "to_name": "image",
                        "from_name": "choice",
                    }
                ],
            }

            if args.vis_dir:
                _draw_prediction(
                    img_path, pred_class, score, os.path.join(args.vis_dir, image_name)
                )

            gt_class = labels.get(image_name)
            if gt_class is not None:
                y_true.append(gt_class)
                y_pred.append(pred_class)

        except Exception as e:
            print(f"Failed with {img_path}. {e}")

    print(f"Inference time: {round(time.time() - start_time, 2)} s.")

    with open(args.out_file, "w") as f:
        json.dump(predictions, f)

    # Metrics only over annotated images (entries whose value was not None).
    if args.metrics_file and y_true:
        report = classification_report(
            y_true, y_pred, labels=args.classes, output_dict=True, zero_division=0
        )
        os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
        with open(args.metrics_file, "w") as f:
            json.dump(report, f, indent=2)
        print("Classification metrics saved.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("config")
    parser.add_argument("images_dir", type=Path)
    parser.add_argument(
        "--out-file", type=str, required=True,
        help="Single JSON file with predictions, keyed by image name.")
    parser.add_argument(
        "--ann-file", type=str, default=None,
        help="JSON keyed by image name -> GT class (or null). Its keys select "
             "which images to process; non-null values are used for metrics.")
    parser.add_argument(
        "--metrics-file", type=str, default=None,
        help="Path to write the classification report (JSON).")
    parser.add_argument(
        "--vis-dir", type=str, default=None,
        help="If set, draw each prediction onto the image and save it here "
             "(used for single-image predict to show the result in Label Studio).")
    parser.add_argument(
        "--silent", action="store_true",
        help="suppress progress bars and verbose output")
    parser.add_argument(
        "--classes", nargs="+", required=True,
        help="list of classes for inference")
    config = parser.parse_args()
    main(config)
