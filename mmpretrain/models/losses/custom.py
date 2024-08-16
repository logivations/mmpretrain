import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from mmpretrain.models.losses import CrossEntropyLoss
@MODELS.register_module()
class ConfidenceLoss(nn.Module):
    """Cross entropy loss.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (List[float], optional): The positive weight for each
            class with shape (C), C is the number of classes. Only enabled in
            BCE loss when ``use_sigmoid`` is True. Default None.
    """

    def __init__(self, *args, **kwargs):
        super(ConfidenceLoss, self).__init__()
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self,
                cls_score,
                gt_score,
                **kwargs):
        pred_loaded, pred_confidence = cls_score[:, 0], cls_score[:, 1]
        gt_loaded, gt_confidence = gt_score[:, 0], gt_score[:, 1]

        loaded_loss = self.mse_loss(pred_loaded, gt_loaded)
        loaded_loss = loaded_loss * gt_confidence
        print(f"loaded_loss: {loaded_loss}")

        confidence_loss = self.mae_loss(
            pred_confidence,
            gt_confidence,
        )
        print(f"confidence_loss: {confidence_loss}")

        loss = confidence_loss + loaded_loss
        print(f"final loss: {loss}")
        return loss