# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Sequence, Union

from mmseg.registry import DATASETS
from .api_wrappers import COCOPanoptic
from .coco_panoptic import CocoPanopticDataset


@DATASETS.register_module()
class CityPanopticDataset(CocoPanopticDataset):
    """Waymo dataset for Panoptic segmentation.
    """

    METAINFO = {
        'classes':
        ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky'),
        'thing_classes':
        ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'),
        'stuff_classes':
        ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky'),
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0)]
    }
    COCOAPI = COCOPanoptic
    # ann_id is not unique in coco panoptic dataset.
    ANN_ID_UNIQUE = False

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = METAINFO,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=None, ann=None, seg=None),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 backend_args: dict = None,
                 **kwargs) -> None:
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            backend_args=backend_args,
            **kwargs)
