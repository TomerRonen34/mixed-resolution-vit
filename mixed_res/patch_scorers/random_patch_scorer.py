from typing import Optional

import torch
from torch import FloatTensor

from mixed_res.patch_scorers.patch_scorer import PatchScorer


class RandomPatchScorer(PatchScorer):
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.seed = seed

    def _calculate_scores(self, images: FloatTensor, small_patch_size: int, patch_size: int, **kwargs) -> list[list[float]]:
        if patch_size == small_patch_size:
            return self._get_leaf_scores(images, small_patch_size)

        bsz, _, _, full_size = images.shape
        grid_size = full_size // patch_size
        num_patches = grid_size**2
        patch_size = patch_size.item() if torch.is_tensor(patch_size) else patch_size
        rng = None if (self.seed is None) else torch.Generator(
            device=images.device).manual_seed(self.seed + patch_size)
        scores = -torch.rand((bsz, num_patches),
                             device=images.device, generator=rng)
        return scores
