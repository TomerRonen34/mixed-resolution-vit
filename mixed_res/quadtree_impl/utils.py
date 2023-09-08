import math
from functools import cache
from typing import Optional

import numpy as np
import torch
from torch import FloatTensor, LongTensor, Tensor


def assert_valid_quadtree_params(num_patches: int, min_patch_size: int, max_patch_size: int, image_size: int) -> None:
    assert image_size % max_patch_size == 0, f"image_size {image_size} must be divisible by max_patch_size {max_patch_size}"
    assert max_patch_size % min_patch_size == 0, f"max_patch_size {max_patch_size} must be divisible by min_patch_size {min_patch_size}"
    max_num_patches = (image_size // min_patch_size)**2
    assert num_patches <= max_num_patches, \
        f"num_patches {num_patches} is too large, max_num_patches is {max_num_patches} = ({image_size} / {min_patch_size})**2"
    initial_num_patches = (image_size // max_patch_size)**2
    floored_num_patches = (
        num_patches - initial_num_patches) // 3 * 3 + initial_num_patches
    assert floored_num_patches == num_patches, \
        f"Illegal num_patches {num_patches} for image_size {image_size}, try {floored_num_patches} or {floored_num_patches + 3}"


def extract_model_inputs(model_inputs_by_scale: list[FloatTensor],
                         patch_ids: LongTensor,
                         size_ids: LongTensor) -> FloatTensor:
    batch_size, num_patches = patch_ids.shape
    some_inputs = model_inputs_by_scale[0]
    inputs_dim = some_inputs.shape[-1]
    dtype = some_inputs.dtype
    model_inputs = torch.zeros([batch_size, num_patches, inputs_dim],
                               device=patch_ids.device, dtype=dtype)

    for curr_size_id, scale_features in enumerate(model_inputs_by_scale):
        scale_mask = (curr_size_id == size_ids)
        i_sample, i_patch = torch.nonzero(scale_mask, as_tuple=True)
        scale_patch_ids = patch_ids[scale_mask]
        extracted_scale_inputs = scale_features[i_sample, scale_patch_ids, :]
        model_inputs[i_sample, i_patch, :] = extracted_scale_inputs

    return model_inputs


def create_box_metadata(min_patch_size: int, patch_size: int, image_size: int, batch_size: int, device: torch.device, dtype: torch.dtype, mode: str = "center") -> Tensor:
    position_ids = _create_position_ids(
        min_patch_size, patch_size, image_size, batch_size, device, dtype, mode)
    num_patches = position_ids.shape[1]
    size_ids = _create_size_ids(
        min_patch_size, patch_size, batch_size, num_patches, device, dtype)
    box_metadata = torch.cat([position_ids, size_ids], dim=-1)
    return box_metadata


def _create_position_ids(min_patch_size: int, patch_size: int, image_size: int, batch_size: int, device: torch.device, dtype: torch.dtype, mode: str) -> Tensor:
    boxes = grid_boxes_tensor(patch_size, image_size, device, dtype)
    if mode == "center":
        pixel_coords = (boxes[:, :2] + boxes[:, 2:]) / 2
    elif mode == "left_top":
        pixel_coords = boxes[:, :2]
    else:
        raise ValueError(mode)
    grid_coords = pixel_coords / min_patch_size
    grid_coords = grid_coords.to(dtype)
    position_ids = grid_coords.unsqueeze(0).expand(batch_size, -1, -1)
    return position_ids


def _create_size_ids(min_patch_size: int, patch_size: int, batch_size: int, num_patches: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    size_id_value = _size_id_value(patch_size, min_patch_size)
    size_ids = size_id_value * \
        torch.ones((batch_size, num_patches, 1), device=device, dtype=dtype)
    return size_ids


def _size_id_value(patch_size: int, min_patch_size: int) -> int:
    size_ratio = patch_size / min_patch_size
    if isinstance(size_ratio, Tensor):
        size_ratio = size_ratio.item()
    return int(math.log2(size_ratio))


@cache
def grid_boxes_tuples(patch_size: int,
                      full_width: int,
                      full_height: Optional[int] = None,
                      x_shift: int = 0,
                      y_shift: int = 0,
                      ) -> list[tuple[int, int, int, int]]:
    """ returns list of (left, top, right, bottom) tuples in raster order: scan the image left-to-right, top-to-bottom """
    if full_height is None:
        full_height = full_width
    grid = [(left + x_shift, top + y_shift,
             left + x_shift + patch_size, top + y_shift + patch_size)
            for top in range(0, full_height, patch_size)
            for left in range(0, full_width, patch_size)]
    return grid


@torch.jit.script
def grid_boxes_tensor(patch_size: int, full_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    arange = torch.arange(0, full_size, patch_size, dtype=dtype, device=device)
    left_top = torch.fliplr(torch.cartesian_prod(arange, arange))
    right_bottom = left_top + patch_size
    boxes = torch.concat([left_top, right_bottom], dim=1)
    return boxes


def patch_arange(min_patch_size: int, max_patch_size: int) -> Tensor:
    min_logsize = torch.log2(torch.tensor(min_patch_size))
    max_logsize = torch.log2(torch.tensor(max_patch_size))
    patch_arange = 2 ** torch.arange(min_logsize, max_logsize + 1)
    patch_arange = patch_arange.round().long()
    return patch_arange


def sort_by_meta(quadtree_output):
    """ Useful for comparisons between model_output from different Quadtree implementations. """
    return torch.stack([
        x[torch.tensor(np.lexsort(x[:, -3:].T.cpu().numpy()),
                       dtype=torch.long, device=x.device)]
        for x in quadtree_output
    ])


def is_power_of_2(x: int) -> bool:
    return math.log2(x) % 1 == 0


def split_model_inputs(model_inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    flat_patches, centers, size_ids = (model_inputs[..., :-3],
                                       model_inputs[..., -3:-1],
                                       model_inputs[..., -1:])
    return flat_patches, centers, size_ids
