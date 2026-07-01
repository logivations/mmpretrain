from mmengine.config import Config

def make_mmpret_config(config_path: str, target_class_map: dict, classes: list) -> Config:
    """Setup classes in custom_vit_uper"""
    cfg = Config.fromfile(config_path)

    # Datasets config
    cfg.train_dataloader.dataset.target_class_map = target_class_map
    cfg.train_dataloader.dataset.classes = classes
    cfg.test_dataloader.dataset.target_class_map = target_class_map
    cfg.test_dataloader.dataset.classes = classes
    cfg.val_dataloader.dataset.target_class_map = target_class_map
    cfg.val_dataloader.dataset.classes = classes

    # Derive num_classes from the final class list (after target_class_map
    # remap / exclusions) instead of the hardcoded config value.
    final_classes = []
    for cls in classes:
        if cls in target_class_map:
            if target_class_map[cls] is not None:
                final_classes.append(target_class_map[cls])
        else:
            final_classes.append(cls)
    num_classes = len(set(final_classes))
    cfg.model.head.num_classes = num_classes
    cfg.data_preprocessor.num_classes = num_classes

    return cfg