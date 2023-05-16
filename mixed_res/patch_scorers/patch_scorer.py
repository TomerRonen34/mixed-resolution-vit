import abc
import math

import torch
from torch import FloatTensor, Tensor, nn
from torch.nn.functional import avg_pool2d
from torchvision.transforms.functional import resize

from mixed_res.patchify import patchify_flat

INF_SCORE = 1e6


class PatchScorer(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self.current_images = None
        self.importance_maps = None
        self.force_recalculate = False  # can be ovverriden by children
        self.leaf_scores = None

    def __call__(self, *args,
                 **kwargs) -> tuple[list[list[float]], FloatTensor]:
        return self._calculate_scores_and_small_patches(*args, **kwargs)

    def _calculate_scores_and_small_patches(
        self,
        images: FloatTensor,
        small_patch_size: int,
        patch_size: int,
        **kwargs,
    ) -> tuple[FloatTensor, FloatTensor]:
        """
        patches are in raster order: left-to-right, top-to-bottom.
        returns:
            scores: [batch, num_patches]
            small_patches: [batch, num_patches, h*w*3]
        """
        scores = self._calculate_scores(
            images, small_patch_size, patch_size, **kwargs)
        small_patches = calculate_small_patches(
            images, small_patch_size, patch_size)
        return scores, small_patches

    def _calculate_scores(
        self,
        images: FloatTensor,
        small_patch_size: int,
        patch_size: int,
        **kwargs,
    ) -> list[list[float]]:
        if patch_size == small_patch_size:
            return self._get_leaf_scores(images, small_patch_size)

        if isinstance(patch_size, Tensor):
            patch_size = patch_size.item()
        batch_size, _, image_size, _ = images.shape
        grid_size = image_size // patch_size
        num_patches = grid_size**2

        if (self.current_images is not images) or self.force_recalculate or (self.importance_maps is None):
            self.current_images = images
            self._calculate_importance_maps(
                images=images, small_patch_size=small_patch_size, patch_size=patch_size, **kwargs)

        scale = patch_size // small_patch_size

        assert self.importance_maps is not None, "implement _calculate_importance_maps()"
        importance_maps = self.importance_maps
        scale_idx = int(math.log2(scale) - 1)
        if isinstance(importance_maps, list):
            importance_maps = importance_maps[scale_idx]
        elif importance_maps.ndim == 4:  # [bsz, scale, grid, grid]
            importance_maps = importance_maps[:, scale_idx, :, :]

        map_size = importance_maps.shape[-1]
        if map_size < grid_size:
            raise ValueError(
                f"map_size ({map_size}) < grid_size ({grid_size})")
        elif map_size == grid_size:
            importance_per_patch = importance_maps
        elif (map_size % grid_size == 0):
            pool_size = map_size // grid_size
            importance_per_patch = avg_pool2d(
                importance_maps, kernel_size=pool_size, stride=pool_size)
        else:
            importance_per_patch = resize(
                importance_maps, [grid_size], antialias=False)

        scores = -importance_per_patch.view(batch_size, num_patches)
        return scores

    def _calculate_importance_maps(self, images: FloatTensor, **kwargs) -> None:
        """ sets self.importance_maps """
        raise NotImplementedError()

    def _get_leaf_scores(self, images: FloatTensor, small_patch_size: int) -> FloatTensor:
        batch_size, _, image_size, _ = images.shape
        num_patches = (image_size // small_patch_size)**2
        desired_params = ([batch_size, num_patches],
                          images.device, images.dtype)
        self_params = None if (self.leaf_scores is None) else (
            list(self.leaf_scores.shape), self.leaf_scores.device, self.leaf_scores.dtype)
        if desired_params != self_params:
            self.leaf_scores = torch.full(
                (batch_size, num_patches), INF_SCORE, device=images.device)
        return self.leaf_scores


def calculate_small_images(
    images: FloatTensor,
    small_patch_size: int,
    patch_size: int,
    antialias: bool = None,
) -> FloatTensor:

    if patch_size == small_patch_size:
        return images

    full_size = images.shape[-1]
    scale = patch_size // small_patch_size
    small_size = full_size // scale
    small_images = resize(images, [small_size], antialias=antialias)
    return small_images


def calculate_small_patches(
    images: FloatTensor,
    small_patch_size: int,
    patch_size: int
) -> FloatTensor:
    small_images = calculate_small_images(images, small_patch_size, patch_size)
    small_patches = patchify_flat(small_images, small_patch_size)
    return small_patches
