import torch
from torch import FloatTensor
from torch.nn.functional import avg_pool3d
from torchvision.transforms.functional import resize

from mixed_res.patch_scorers.patch_scorer import (PatchScorer,
                                                  calculate_small_images)
from mixed_res.patchify import patchify_flat


class PixelBlurPatchScorer(PatchScorer):
    def _calculate_scores_and_small_patches(
        self,
        images: FloatTensor,
        small_patch_size: int,
        patch_size: int
    ) -> tuple[FloatTensor, FloatTensor]:
        bsz, channels, image_size, _ = images.shape
        num_patches = (image_size // patch_size)**2

        if patch_size != small_patch_size:
            if isinstance(patch_size, torch.Tensor):
                patch_size = patch_size.item()

            small_images = calculate_small_images(
                images, small_patch_size, patch_size)
            blurry_images = resize(small_images, [image_size], antialias=None)
            squared_error = (blurry_images - images)**2

            mse = avg_pool3d(squared_error,
                             kernel_size=(channels, patch_size, patch_size),
                             stride=patch_size)
            mse = mse.view(bsz, num_patches)

            scores = -torch.sqrt(mse * patch_size)
        else:
            small_images = images
            scores = self._get_leaf_scores(images, small_patch_size)

        small_patches = patchify_flat(small_images, small_patch_size)

        return scores, small_patches

    def _calculate_scores(self, images: FloatTensor, small_patch_size: int, patch_size: int, **kwargs) -> list[list[float]]:
        scores, _ = self._calculate_scores_and_small_patches(
            images, small_patch_size, patch_size)
        return scores

    def _calculate_importance_maps(self, images: FloatTensor, **kwargs) -> None:
        """ No need for an implementation since _calculate_scores_and_small_images is overloaded """
        pass
