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

    return cfg