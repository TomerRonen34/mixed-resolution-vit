from contextlib import nullcontext
from typing import Any, Callable, Union

import torch
from torch import FloatTensor, Tensor

from mixed_res.patch_scorers.patch_scorer import PatchScorer
from mixed_res.quadtree_impl.quadtree_runner import QuadtreeRunner
from mixed_res.quadtree_impl.utils import (assert_valid_quadtree_params,
                                           create_box_metadata,
                                           extract_model_inputs, patch_arange)


def _create_box_metadata_and_index_grid(min_patch_size: int, max_patch_size: int, image_size: int, device: str) -> tuple:
    box_metadata = _create_box_metadata_for_indexing(
        min_patch_size, max_patch_size, image_size, device)
    index_grid = _calculate_index_grid(
        box_metadata, min_patch_size, max_patch_size, image_size)
    return box_metadata, index_grid


def _create_box_metadata_for_indexing(min_patch_size: int, max_patch_size: int, image_size: int, device: str) -> Tensor:
    multires_patch_metadata = [
        create_box_metadata(
            min_patch_size, patch_size, image_size, batch_size=1, device=device, dtype=torch.long, mode="left_top")
        for patch_size in patch_arange(min_patch_size, max_patch_size)
    ]
    box_metadata = torch.concat([meta.squeeze(0)
                                for meta in multires_patch_metadata], dim=0)
    return box_metadata


@torch.jit.script
def _calculate_index_grid(box_metadata: Tensor, min_patch_size: int, max_patch_size: int, image_size: int) -> Tensor:
    device = box_metadata.device
    i_left, i_top, i_size = box_metadata[:,
                                         0], box_metadata[:, 1], box_metadata[:, 2]
    num_scales = int(
        1 + torch.log2(torch.tensor(max_patch_size // min_patch_size)).item())
    grid_size = image_size // min_patch_size
    index_grid = -100 * \
        torch.ones((num_scales, grid_size, grid_size),
                   dtype=torch.long, device=device)
    index_grid[i_size, i_top, i_left] = torch.arange(
        box_metadata.shape[0], device=device)
    return index_grid


class BoxIndexer:
    def __init__(self, min_patch_size: int, max_patch_size: int, image_size: int, device: torch.device):
        self.box_metadata, self.index_grid = _create_box_metadata_and_index_grid(
            min_patch_size, max_patch_size, image_size, device)
        self.one = torch.ones(1, dtype=torch.long, device=device)

    def get_children(self, index: Tensor) -> Tensor:
        return _get_children(index, self.box_metadata, self.index_grid, self.one)


@torch.jit.script
def _get_children(index: Tensor, box_metadata: Tensor, index_grid: Tensor, one: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    box = box_metadata[index]
    i_left, i_top, i_size = box[:, 0], box[:, 1], box[:, 2]
    i_size_minus_1 = i_size - 1
    two_exp_i_size_minus_1 = one << i_size_minus_1
    tl_index = index_grid[i_size_minus_1, i_top, i_left]
    tr_index = index_grid[i_size_minus_1,
                          i_top, i_left + two_exp_i_size_minus_1]
    bl_index = index_grid[i_size_minus_1,
                          i_top + two_exp_i_size_minus_1, i_left]
    br_index = index_grid[i_size_minus_1, i_top +
                          two_exp_i_size_minus_1, i_left + two_exp_i_size_minus_1]
    children_indices = (tl_index, tr_index, bl_index, br_index)
    return children_indices


_float_inf = 1e10
_float_subinf = 1e9


class MinHeap:
    def __init__(self, max_cumulative_length: int, bsz: int, device: torch.device):
        """ max_cumulative_length includes all the items that were ever in the heap, including popped ones. """
        self.inf = torch.tensor(_float_inf, device=device)

        keys_shape = (bsz, max_cumulative_length)
        self.keys = torch.empty(keys_shape, device=device, dtype=torch.float)
        self.indices = torch.empty(keys_shape, device=device, dtype=torch.long)
        self.cursor = None
        self.len = None

        self.max_cumulative_length = max_cumulative_length
        self.bsz = bsz
        self.arange = torch.arange(self.bsz, device=device)

    def reset(self) -> None:
        self.keys[:] = self.inf
        self.cursor = 0
        self.len = 0

    def pop(self) -> Any:
        # if self.len == 0:
        #     raise IndexError("pop from empty heap")
        i = self.keys.argmin(dim=1)
        index = self.indices[self.arange, i]
        self.keys[self.arange, i] = self.inf
        self.len -= 1
        return index

    def push(self, key: float, index: Any) -> None:
        # if self.len == self.max_cumulative_length:
        #     raise IndexError(
        #         f"heap full, len=max_cumulative_length={self.max_cumulative_length}")
        self.keys[:, self.cursor] = key
        self.indices[:, self.cursor] = index
        self.cursor += 1
        self.len += 1

    def __len__(self) -> int:
        return self.len

    def get_active_indices(self) -> list:
        active_mask = (self.keys < self.inf)
        flat_active_indices = self.indices[active_mask]
        active_indices = flat_active_indices.view((self.bsz, self.len))
        return active_indices


class TensorLookupQuadtreeRunner(QuadtreeRunner):
    def __init__(self, num_patches: int, min_patch_size: int, max_patch_size: int, no_grad: bool = True):
        super().__init__(num_patches, min_patch_size, max_patch_size, no_grad)
        self.patch_arange = patch_arange(min_patch_size, max_patch_size)
        self.input_params = None

    def _init_input_specific_members(self, images: FloatTensor):
        self.input_params = (images.shape, images.device)
        batch_size, _, image_size, _ = images.shape
        device = images.device
        num_patches, min_patch_size, max_patch_size = self.num_patches, self.min_patch_size, self.max_patch_size
        assert_valid_quadtree_params(num_patches, min_patch_size, max_patch_size, image_size)

        initial_num_patches = (image_size // max_patch_size)**2
        self.num_splits = (num_patches - initial_num_patches) // 3
        max_cumulative_length = initial_num_patches + 4 * self.num_splits

        self.box_indexder = BoxIndexer(
            min_patch_size, max_patch_size, image_size, device)
        self.initial_box_indices = self.box_indexder.box_metadata.shape[0] + torch.arange(
            -initial_num_patches, 0)
        self.heap = MinHeap(max_cumulative_length, batch_size, device=device)
        num_leaves = (image_size // self.min_patch_size)**2
        self.leaf_scores = torch.full(
            (batch_size, num_leaves), fill_value=_float_subinf, device=device)
        self.arange = torch.arange(batch_size, device=device)
        self.patch_ids_table = torch.concat([torch.arange((image_size // patch_size)**2, device=device, dtype=torch.long)
                                             for patch_size in patch_arange(min_patch_size, max_patch_size)])
        self.box_metadata_by_scale = [create_box_metadata(min_patch_size, patch_size, image_size, batch_size, device, torch.float)
                                      for patch_size in self.patch_arange]

    def run_batch_quadtree(self, images: FloatTensor, score_func: Union[PatchScorer, Callable]) -> FloatTensor:
        """
        Like multires.quadtree.run_batch_quadtree but faster
        """
        with self.context_manager_func():
            image_params = (images.shape, images.device)
            if self.input_params != image_params:
                self._init_input_specific_members(images)

            scores, small_patches_by_scale = zip(*[
                score_func(images, self.min_patch_size, patch_size)
                for patch_size in self.patch_arange])
            model_inputs_by_scale = [torch.cat([small_patches, box_metadata], dim=-1)
                                     for small_patches, box_metadata in zip(small_patches_by_scale, self.box_metadata_by_scale)]

            # no need for the leaf scores, we set them inside _run_quadtrees
            scores = scores[1:]

            patch_ids, size_ids = self._run_quadtrees(scores)

            model_inputs = extract_model_inputs(
                model_inputs_by_scale, patch_ids, size_ids)

            _for_flops_count = 0. * sum([x.sum() for x in scores])
            model_inputs = model_inputs + _for_flops_count

        return model_inputs

    def _run_quadtrees(self, scores: list[Tensor]) -> tuple[Tensor, Tensor]:
        scores = torch.concat((self.leaf_scores,) + scores, dim=1)

        self.heap.reset()
        for index in self.initial_box_indices:
            score = scores[:, index]
            self.heap.push(score, index)

        for _ in range(self.num_splits):
            chosen = self.heap.pop()
            children_indices = self.box_indexder.get_children(chosen)
            for child_indices in children_indices:
                child_scores = scores[self.arange, child_indices]
                self.heap.push(child_scores, child_indices)

        chosen = self.heap.get_active_indices()
        size_ids = self.box_indexder.box_metadata[chosen, 2]
        patch_ids = self.patch_ids_table[chosen]

        return patch_ids, size_ids
