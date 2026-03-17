from pathlib import Path
from argparse import ArgumentParser
import os
import json
import time
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

class FPSLogger:
    def __init__(self, num_of_images):
        self.tottime = 0.0
        self.count = 0
        self.last_record = 0.0
        self.last_print = time.time()
        self.interval = 3
        self.num_of_images = num_of_images

    def start_record(self):
        self.last_record = time.time()

    def end_record(self):
        self.tottime += time.time() - self.last_record
        self.count += 1
        self.print_fps()

    def print_fps(self):
        if time.time() - self.last_print > self.interval:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - mmpret - INFO - Predict({self.count}/{self.num_of_images}) "
                  f"- Inference running at {self.count / self.tottime:.3f} FPS")
            self.last_print = time.time()

def main(args):
    fps_logger = FPSLogger(len(os.listdir(args.images_dir)))
    inference = ImageClassificationInferencer(
        model=args.config,
        pretrained=args.checkpoint,
        classes=args.classes
    )
    print(f"Inference classes: {args.classes}")
    if args.silent:
        inference.show_progress = False

    images: Path = args.images_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True) 

    y_true, y_pred = ([], []) if args.ann_dir else (None, None)

    start_time = time.time()
    image_paths = list(images.glob("**/*.jpg"))
    for i, p in enumerate(image_paths, 1):
        try:
            fps_logger.start_record()
            result = inference(str(p))
            fps_logger.end_record()

            pred_class = result[0]['pred_class']

            file_name = p.stem + ".json"
            pred_path = os.path.join(output_dir, file_name)
            prediction = {
                "result": [
                    {
                        "type": "choices",
                        "value": {"choices": [pred_class]},
                        "origin": "manual",
                        "to_name": "image",
                        "from_name": "choice",
                    }
                ],
            }

            with open(pred_path, "w") as f:
                json.dump(prediction, f)

            if args.ann_dir is not None:
                ann_path = args.ann_dir / file_name
                try:
                    with open(ann_path) as f:
                        ann = json.load(f)
                    gt_class = ann["result"][0]["value"]["choices"][0]
                    y_true.append(gt_class)
                    y_pred.append(pred_class)
                except Exception as e:
                    print(f"Warning: could not load GT for {p.name}: {e}")

        except Exception as e:
            print(f"Failed with {p}. {e}")

    print(f"Inference time: {round(time.time() - start_time, 2)} s.")

    if args.ann_dir is not None and y_true:
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, labels=args.classes))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("config")
    parser.add_argument("images_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--ann-dir", type=Path, default=None,
        help="Directory with ground truth annotation JSONs (same filename as images). "
             "When provided, precision/recall/F1-score are printed after inference.")
    parser.add_argument(
        "--silent",
        action="store_true",
        help="suppress progress bars and verbose output")
    parser.add_argument(
        '--classes',
        nargs='+',
        required=True,
        help='list of classes for the training'
    )
    config = parser.parse_args()
    main(config)
