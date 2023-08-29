# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from ..utils import interpolate_as
from .panoptic_fpn_head import PanopticFPNHead
from typing import Dict, Tuple, Union
from ..layers import ConvUpsample

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class FieldHead(PanopticFPNHead):
    def __init__(self,
                 num_things_classes: int = 8,
                 num_stuff_classes: int = 21,
                 in_channels: int = 256,
                 inner_channels: int = 128,
                 start_level: int = 0,
                 end_level: int = 4,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 loss_seg: ConfigType = dict(
                     type='MSELoss',
                     loss_weight=1.0),
                 init_cfg: OptMultiConfig = None):
        
        super().__init__(num_things_classes,
                 num_stuff_classes,
                 in_channels,
                 inner_channels,
                 start_level,
                 end_level,
                 conv_cfg,
                 norm_cfg,
                 loss_seg,
                 init_cfg)
        self.up2 = ConvUpsample(
            128,
            32,
            num_layers=2,
            num_upsample=2,
            conv_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        )
        self.loss_seg = MODELS.build(loss_seg)
        self.fieldout = nn.Conv2d(32, 24, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @staticmethod
    def _getfields(mask):
        device = mask.device
        if not mask.any():
            return torch.zeros((mask.shape[0],mask.shape[1],3),device=device)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        positive_indices = torch.where(mask)
        min_row, min_col = torch.min(positive_indices[0]),torch.min(positive_indices[1])
        max_row, max_col = torch.max(positive_indices[0]),torch.max(positive_indices[1])
        y_center, x_center = (min_col + max_col) // 2, (min_row + max_row) // 2
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        deviation = (width / 3, height / 3)
        x, y = torch.meshgrid([torch.arange(0, mask.shape[0]), torch.arange(0, mask.shape[1])])
        x, y = x.to(device), y.to(device)
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

    def loss(self, x: Union[Tensor, Tuple[Tensor]],
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """
        Args:
            x (Union[Tensor, Tuple[Tensor]]): Feature maps.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            Dict[str, Tensor]: The loss of semantic head.
        """
        seg_fields = self(x)['seg_fields']
        
        gt_fields_list = [
            self._getfields_classwise(data_sample.gt_instances.masks.to_tensor(dtype = torch.float32, device = self.device), data_sample.gt_instances.labels.to(self.device))
            for data_sample in batch_data_samples
        ]
        gt_fields = torch.stack(gt_fields_list)

        loss_fields = self.loss_seg(
            seg_fields, 
            gt_fields)
        
        if torch.isnan(loss_fields).any():
            raise ValueError("Nan detected")

        return dict(loss_fields=loss_fields)

    def init_weights(self) -> None:
        """Initialize weights."""
        super().init_weights()
        nn.init.normal_(self.fieldout.weight.data, 0, 0.01)
        self.fieldout.bias.data.zero_()

    def forward(self, x: Tuple[Tensor]) -> Dict[str, Tensor]:
        """Forward.

        Args:
            x (Tuple[Tensor]): Multi scale Feature maps.

        Returns:
            dict[str, Tensor]: semantic segmentation predictions and
                feature maps.
        """
        # the number of subnets must be not more than
        # the length of features.
        assert self.num_stages <= len(x)

        feats = []
        for i, layer in enumerate(self.conv_upsample_layers):
            f = layer(x[self.start_level + i])
            feats.append(f)

        seg_feats = torch.sum(torch.stack(feats, dim=0), dim=0)
        field_feats = self.up2(seg_feats)
        seg_fields = self.fieldout(field_feats)
        out = dict(seg_fields = seg_fields)
        return out
    
    def predict(self, *args, **kwargs):
        return self.forward(*args,**kwargs)