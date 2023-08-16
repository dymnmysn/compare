# Simdilik silebilirsin
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from .utils import weighted_loss


@weighted_loss
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """A Wrapper of MSE loss.
    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: loss Tensor
    """
    return F.mse_loss(pred, target, reduction='none')


@MODELS.register_module()
class FieldLoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    @staticmethod
    def _getfields(mask):
        if not mask.any():
            return torch.zeros((mask.shape[0],mask.shape[1],3))
        
        positive_indices = torch.where(mask)
        min_row, min_col = torch.min(positive_indices[0]),torch.min(positive_indices[1])
        max_row, max_col = torch.max(positive_indices[0]),torch.max(positive_indices[1])
        y_center, x_center = (min_col + max_col) // 2, (min_row + max_row) // 2
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        deviation = (width / 3, height / 3)
        x, y = torch.meshgrid([torch.arange(0, mask.shape[0]), torch.arange(0, mask.shape[1])])
        exponent = -((x - x_center)**2 / (2 * deviation[1]**2) + (y - y_center)**2 / (2 * deviation[0]**2))
        kernel = torch.exp(exponent)  

        distances_x = (y - y_center) / (width + 1)
        distances_y = (x - x_center) / (height + 1)

        return torch.stack((distances_x*mask, distances_y*mask, kernel*mask), dim=-1)

    def _getfields_classwise(self, masks, labels):
        dummy = self._getfields(masks[0]) * 0.0
        fields = {i: torch.zeros_like(dummy) for i in range(8)}
        for label, mask in zip(labels, masks):
            fields[int(label)] += self._getfields(mask)
        out_tensor = torch.cat(tuple(fields.values()), dim=-1).permute(2,0,1)
        return out_tensor
    
    """
    masks = results['gt_masks']
    labels = results['gt_bboxes_labels']
    gt_fields = self._getfields_classwise(masks.to_tensor(device='cpu', dtype = torch.bool), labels)
    """

    def forward(self,
                fields: Tensor,
                masks: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function of loss.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: The calculated loss.
        """

        target = self._getfields_classwise(masks)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * mse_loss(
            fields, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
