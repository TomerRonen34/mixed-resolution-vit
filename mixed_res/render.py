import math
from typing import Optional, Union

import torch
from PIL import Image, ImageDraw
from torch import FloatTensor, LongTensor, Tensor
from torchvision.transforms.functional import to_pil_image

from mixed_res.quadtree_impl.utils import split_model_inputs


def render_quadtree(image: Union[FloatTensor, Image.Image], model_inputs: FloatTensor,
                    blur: bool = True, show_lines: bool = True, mini_patches: bool = False,
                    line_color_rgba: Optional[tuple] = (0, 0, 0, 85)) -> Image.Image:
    assert model_inputs.ndim == 2  # [num_patches, x-y-size]
    assert not (blur and mini_patches)

    if isinstance(image, Tensor):
        image = tensor_to_pil_image(image)

    boxes, min_patch_size = _extract_boxes_from_model_inputs(model_inputs)

    vis = Image.new('RGB', (image.width, image.height), (255, 255, 255))
    draw = ImageDraw.Draw(vis, "RGBA")
    line_kwargs = dict(width=1, fill=line_color_rgba)
    for l, t, r, b in boxes.tolist():
        patch = image.crop((l, t, r, b))

        if mini_patches or blur:
            patch_size = patch.size
            patch = patch.resize(
                (min_patch_size, min_patch_size), resample=Image.BILINEAR)
            if blur:
                patch = patch.resize(patch_size, resample=Image.BILINEAR)

        if not mini_patches:
            vis.paste(patch, (l, t))
        else:
            vis.paste(patch, ((l + r - patch.width) //
                              2, (t + b - patch.height) // 2))

        if show_lines:
            border_points = ((r - 1, t), (r - 1, b - 1), (l, b - 1))
            draw.line(border_points, **line_kwargs)

    if show_lines:
        border_points = ((vis.width - 1, 0), (0, 0), (0, vis.height - 1))
        draw.line(border_points, **line_kwargs)

    return vis


def _extract_boxes_from_model_inputs(model_inputs: FloatTensor) -> tuple[LongTensor, int]:
    flat_patches, centers, size_ids = split_model_inputs(model_inputs)
    min_patch_size = int(math.sqrt(flat_patches.shape[1] / 3))
    is_floored_centers = ((centers % 1) == 0).all()
    if is_floored_centers:
        centers = torch.where(size_ids != 0, centers, centers + 0.5)
    centers = centers * min_patch_size
    patch_sizes = min_patch_size * (2 ** size_ids)
    boxes = torch.cat([centers - patch_sizes / 2, centers + patch_sizes / 2],
                      dim=1).long()
    return boxes, min_patch_size


def tensor_to_pil_image(image: FloatTensor) -> Image.Image:
    assert image.ndim == 3  # [C,H,W]
    if image.min() < 0:
        image = image - image.min()
        image = image / image.max()
    image = to_pil_image(image)
    return image
