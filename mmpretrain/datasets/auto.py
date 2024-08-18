import json
import os

from mmpretrain.datasets.custom import CustomDataset, get_samples
from mmpretrain.registry import DATASETS

from typing import Dict, List, Optional, Tuple

from mmengine.fileio import (BaseStorageBackend, get_file_backend,
                             list_from_file)
from mmengine.logging import MMLogger




@DATASETS.register_module()
class AutoDataset(CustomDataset):
    """
        A extended CustomDataset from mmpretrain.datasets.custom.py
    """

    def __init__(
        self,
        target_class_map=None,
        classes: list = None,
        **kwargs,
    ):
        print(f"Found classes: {classes}")
        self.target_class_map = target_class_map
        CLASSES = []
        for cls in classes:
            if cls in target_class_map:
                if target_class_map[cls] is not None:
                    CLASSES.append(target_class_map[cls])
            else:
                CLASSES.append(cls)

        print(f"Classes for training: {CLASSES}")
        print(f"Target_class_map: {target_class_map}")

        super().__init__(classes=sorted(CLASSES), **kwargs)

    def _find_samples(self):
        """find samples from ``data_prefix``."""
        if self.with_label:
            classes, folder_to_idx = find_folders(self.img_prefix, target_class_map=self.target_class_map)
            samples, empty_classes = get_samples(
                self.img_prefix,
                folder_to_idx,
                is_valid_file=self.is_valid_file,
            )

            self.folder_to_idx = folder_to_idx

            if self.CLASSES is not None:
                assert len(self.CLASSES) == len(classes), \
                    f"The number of subfolders ({len(classes)}) doesn't " \
                    f'match the number of specified classes ' \
                    f'({len(self.CLASSES)}). Please check the data folder.'
            else:
                self._metainfo['classes'] = tuple(classes)
        else:
            samples, empty_classes = get_samples(
                self.img_prefix,
                None,
                is_valid_file=self.is_valid_file,
            )

        if len(samples) == 0:
            raise RuntimeError(
                f'Found 0 files in subfolders of: {self.data_prefix}. '
                f'Supported extensions are: {",".join(self.extensions)}')

        if empty_classes:
            logger = MMLogger.get_current_instance()
            logger.warning(
                'Found no valid file in the folder '
                f'{", ".join(empty_classes)}. '
                f"Supported extensions are: {', '.join(self.extensions)}")

        return samples

def find_folders(
    root: str,
    backend: Optional[BaseStorageBackend] = None,
    target_class_map: Optional[dict] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders
        backend (BaseStorageBackend | None): The file backend of the root.
            If None, auto infer backend from the root path. Defaults to None.

    Returns:
        Tuple[List[str], Dict[str, int]]:

        - folders: The name of sub folders under the root.
        - folder_to_idx: The map from folder name to class idx.
    """
    # Pre-build file backend to prevent verbose file backend inference.
    backend = backend or get_file_backend(root, enable_singleton=True)
    folders = list(
        backend.list_dir_or_file(
            root,
            list_dir=True,
            list_file=False,
            recursive=False,
        ))
    folders.sort()
    if target_class_map:
        for key, value in target_class_map.items():
            if value is None and key in folders:
                folders.remove(key)

    folder_to_idx = {folders[i]: i for i in range(len(folders))}

    if target_class_map:
        for folder in folders:
            if folder in target_class_map and target_class_map[folder] in folders:
                folder_to_idx[folder] = folder_to_idx[target_class_map[folder]]
    return folders, folder_to_idx

@DATASETS.register_module()
class ConfidenceDataset(CustomDataset):
    """
        A extended CustomDataset from mmpretrain.datasets.custom.py
    """

    def __init__(
        self,
        ann_file: str = "/data/dataset/annotations.json",
        **kwargs,
    ):
        CLASSES = ["loaded", "confidence"]
        self.ann_file = ann_file
        if not self.ann_file:
            self.ann_file = "/data/dataset/annotations.json",
        print(f"Classes for training: {CLASSES}")

        super().__init__(ann_file=ann_file, classes=sorted(CLASSES), **kwargs)

    def _find_samples(self):
        """find samples from ``data_prefix``."""
        samples = []

        with open(self.ann_file, "r") as f:
            data = json.load(f)
        for path, ann in data.items():
            full_path = os.path.join(self.data_prefix['img_path'], path)
            if os.path.exists(full_path):
                samples.append((path, ann['loaded'], ann['confidence']))

        print(f"samples: {samples}")
        return samples

    def load_data_list(self):
        """Load image paths and gt_labels."""
        samples = self._find_samples()

        # Pre-build file backend to prevent verbose file backend inference.
        backend = get_file_backend(self.img_prefix, enable_singleton=True)
        data_list = []
        for sample in samples:
            filename, state, confidence = sample
            img_path = backend.join_path(self.img_prefix, filename)
            info = {'img_path': img_path, 'gt_score': {"loaded": [state, confidence], "confidence": [confidence]}}
            data_list.append(info)
        return data_list