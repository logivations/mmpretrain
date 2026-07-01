import json
import os

from mmpretrain.datasets.custom import CustomDataset
from mmpretrain.registry import DATASETS


@DATASETS.register_module()
class JsonDataset(CustomDataset):
    """CustomDataset variant that reads a JSON annotation file instead of using
    a folder-per-class layout.

    The annotation file (``ann_file``) is ``{image_name: class_name}`` and all
    images live flat in ``data_prefix`` (mounted read-only, never copied). The
    label is resolved against the (sorted) class list so ``gt_label`` indices
    match the classification head — same class ordering as ``AutoDataset``.
    """

    def __init__(
        self,
        target_class_map=None,
        classes: list = None,
        **kwargs,
    ):
        self.target_class_map = target_class_map or {}
        # Build the final class list exactly like AutoDataset: apply the
        # target_class_map remap, drop classes mapped to None.
        final_classes = []
        for cls in classes or []:
            if cls in self.target_class_map:
                if self.target_class_map[cls] is not None:
                    final_classes.append(self.target_class_map[cls])
            else:
                final_classes.append(cls)

        super().__init__(classes=sorted(final_classes), **kwargs)

    def load_data_list(self):
        """Read {image_name: class_name} JSON and return img_path + gt_label."""
        with open(self.ann_file, "r") as f:
            annotations = json.load(f)

        class_to_idx = self.class_to_idx  # {class_name: idx} over sorted CLASSES
        data_list = []
        for image_name, class_name in annotations.items():
            # Apply the same remap used to build CLASSES.
            if class_name in self.target_class_map:
                class_name = self.target_class_map[class_name]
            if class_name is None or class_name not in class_to_idx:
                continue

            img_path = os.path.join(self.img_prefix, image_name)
            if not os.path.exists(img_path):
                continue

            data_list.append(
                {"img_path": img_path, "gt_label": int(class_to_idx[class_name])}
            )
        return data_list
