import math

import torch
from torch import LongTensor, Tensor

from mixed_res.patch_scorers.patch_scorer import INF_SCORE
from mixed_res.quadtree_impl.utils import (assert_valid_quadtree_params,
                                           is_power_of_2, patch_arange)


def z_curve(n, device="cpu"):
    reverse_inds = mortonify(n, device)
    z_curve_inds = torch.argsort(reverse_inds)
    return z_curve_inds


def mortonify(n, device="cpu"):
    """(i,j) index to linear morton code"""
    arange = torch.arange(n, device=device)
    i, j = torch.cartesian_prod(arange, arange).T
    z = torch.zeros(i.shape[0], dtype=torch.long, device=device)
    for pos in range(32):
        z = (z | ((j & (1 << pos)) << pos) | ((i & (1 << pos)) << pos + 1))
    return z


class QuadtreePatchChooser:
    def __init__(self, num_patches: int, min_patch_size: int, max_patch_size: int, image_size: int, batch_size: int, device: torch.device) -> None:
        assert is_power_of_2(image_size), \
            "Image size must be a power of 2 to use z_curve Quadtree implementation. Try quadtree_tensor_lookup.py instead."
        assert_valid_quadtree_params(
            num_patches, min_patch_size, max_patch_size, image_size)

        max_grid_size = image_size // max_patch_size
        largest_scale = int(math.log2(max_grid_size))
        self.lower_levels_pad_size = torch.sum(
            2 ** (2 * torch.arange(largest_scale))).item()

        reverse_patch_arange = torch.flip(
            patch_arange(min_patch_size, max_patch_size), (0,))

        ids_pad = torch.zeros(self.lower_levels_pad_size,
                              device=device, dtype=torch.long)

        self.z_curves = [z_curve(image_size // patch_size, device=device)
                         for patch_size in reverse_patch_arange]
        z_curves_concat = torch.concat([ids_pad] + self.z_curves)

        size_ids = torch.log2(reverse_patch_arange)
        size_ids = size_ids - size_ids.min()
        size_ids = [torch.full_like(morton, fill_value=scale)
                    for morton, scale in zip(self.z_curves, size_ids)]
        size_ids_concat = torch.concat([ids_pad] + size_ids)

        self.ids_table = torch.stack(
            [z_curves_concat, size_ids_concat], dim=-1).unsqueeze(0)

        self.num_patches = num_patches
        self.initial_num_patches = max_grid_size ** 2
        self.num_splits = (num_patches - self.initial_num_patches) // 3
        self.child_arange = torch.arange(1, 5, device=device)[None, :]

        self.scores_pad = torch.full(
            (batch_size, self.lower_levels_pad_size), INF_SCORE, device=device)
        self.heap = torch.empty(
            (batch_size, self.ids_table.shape[1]), dtype=torch.bool, device=device)
        self.inf_score = torch.tensor(INF_SCORE, device=device)

    def run(self,
            scores: list[Tensor],
            ) -> tuple[LongTensor, LongTensor]:
        scores = self._prepare_scores(scores)
        patch_ids, size_ids = self._select_patches(scores)
        return patch_ids, size_ids

    def _prepare_scores(self, scores: list[Tensor]) -> Tensor:
        scores = list(scores)[::-1]
        for i_scale in range(len(scores) - 1):
            scores[i_scale] = scores[i_scale][:, self.z_curves[i_scale]]

        scores = torch.concat([self.scores_pad] + scores, dim=1)
        return scores

    def _select_patches(self, scores: Tensor):
        """
        errors are ordered s.t child(i) = 4*i+1:4*i+4
        max_k: in each iteration expand no more than max_k leaves
        """
        self.heap[:] = 0
        self.heap[:, self.lower_levels_pad_size:self.lower_levels_pad_size +
                  self.initial_num_patches] = 1

        for _ in range(self.num_splits):
            active_scores = torch.where(
                self.heap.bool(), scores, self.inf_score)
            to_expand = active_scores.argmin(dim=1, keepdim=True)
            self.heap.scatter_(dim=1, index=to_expand, value=0)
            children = 4 * to_expand + self.child_arange
            self.heap.scatter_(dim=1, index=children, value=1)

        mask = self.heap.unsqueeze(-1).bool()
        ids = self.ids_table.masked_select(mask).view(
            (scores.shape[0], self.num_patches, 2))
        patch_ids, size_ids = ids.unbind(dim=-1)
        return patch_ids, size_ids


class QuadtreePatchChooserForFlopCounting(QuadtreePatchChooser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heap = self.heap.to(torch.int32)

    def _run(self, scores: Tensor):
        scores = self._prepare_scores(scores)
        patch_ids, size_ids = self._select_patches(scores)

        _for_flops = 0 * scores.sum().long()
        patch_ids = patch_ids + _for_flops
        return patch_ids, size_ids
