import mmcv
from mmdet.registry import TRANSFORMS
from mmdet.datasets import PIPELINES
from mmcv.transforms import BaseTransform
import numpy as np
import torch


def getcenterfield(mask):
    if not mask.any():
        return np.zeros_like(mask)
    # Calculate the bounding box around the mask
    positive_indices = np.where(mask)
    min_row, min_col = np.min(positive_indices, axis=1)
    max_row, max_col = np.max(positive_indices, axis=1)
    center = ((min_col + max_col) // 2, (min_row + max_row) // 2)
    width = max_col - min_col + 1
    height = max_row - min_row + 1

    # Generate the Gaussian kernel
    deviation = (width /3, height / 3)
    def generate_2d_gaussian_kernel(shape, center, deviation):
        x, y = np.indices(shape)
        y_center, x_center = center
        exponent = -((x - x_center)**2 / (2 * deviation[1]**2) + (y - y_center)**2 / (2 * deviation[0]**2))
        gaussian_kernel = np.exp(exponent)
        return gaussian_kernel
    kernel = generate_2d_gaussian_kernel(mask.shape, center, deviation)
    filtered_mask = kernel * mask
    return filtered_mask

def getdistancefield(mask):
    if not mask.any():
        distances_x = np.zeros_like(mask)
        distances_y = np.zeros_like(mask)
        distances = np.stack((distances_x, distances_y), axis=-1)
        return distances

    # Find the indices of positive elements in the mask
    positive_indices = np.where(mask)

    # Calculate the bounding box around the positive elements
    min_row, min_col = np.min(positive_indices, axis=1)
    max_row, max_col = np.max(positive_indices, axis=1)

    # Calculate the center point of the bounding box
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2

    # Calculate the width and height of the bounding box
    width = max_col - min_col + 1
    height = max_row - min_row + 1

    # Calculate the coordinates of each pixel in the image
    rows, cols = np.indices(mask.shape[:2])

    # Calculate the distances between each pixel and the center point
    # -1 yerine +1 yazdim burda. Bizim durumda farketmez diye
    distances_x = (cols - center_col) / (width + 1)
    distances_y = (rows - center_row) / (height + 1)

    # Stack the distances along the channel axis
    distances = np.stack((distances_x, distances_y), axis=-1)
    masked_tensor = distances * mask[..., None]
    return masked_tensor

def getfields(mask):
    distancefield = getdistancefield(mask)
    centerfield = getcenterfield(mask)
    return np.concatenate((distancefield, centerfield[..., None]), axis=-1)

thinglabels={2,3,4,5,8,9,10,11}
#labels = gt_instance_mask_labels
#masks = gt_instance_masks
def getfields_classwise(masks_in,tensorlabels):
    masks = masks_in.to_ndarray()
    labels = tensorlabels.cpu().numpy()
    dummy = getfields(masks[0])*0.0
    fields = {i:torch.from_numpy(dummy.copy()) for i in thinglabels}
    for label,mask in zip(labels,masks):
        fields[int(label)] += torch.from_numpy(getfields(mask))
    out_tensor = torch.cat(tuple(fields.values()),dim = -1).permute([2,0,1])
    return out_tensor


#@TRANSFORMS.register_module()
class AddFields(BaseTransform):
    """Add vector fields"""

    def __transform__(self, results: dict) -> dict:
        """Transform function to add vector fields from instance masks from file.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains instance masks.
        """

        gt_instances = results['gt_instances']
        gt_fields = getfields_classwise(gt_instances.masks,gt_instances.labels).to(torch.float32)
        results['gt_fields'] = gt_fields
        return results