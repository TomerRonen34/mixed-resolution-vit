from contextlib import nullcontext
from heapq import heappop, heappush
from itertools import repeat
from typing import Callable, Union

import torch
from torch import FloatTensor, Tensor, tensor

from mixed_res.patch_scorers.patch_scorer import PatchScorer
from mixed_res.quadtree_impl.quadtree_runner import QuadtreeRunner
from mixed_res.quadtree_impl.utils import (_size_id_value,
                                           assert_valid_quadtree_params,
                                           create_box_metadata,
                                           extract_model_inputs,
                                           grid_boxes_tuples, patch_arange)


class DictLookupQuadtreeRunner(QuadtreeRunner):
    def run_batch_quadtree(self, images: FloatTensor, score_func: Union[PatchScorer, Callable]) -> FloatTensor:
        return run_batch_quadtree(self.num_patches, self.min_patch_size, self.max_patch_size, images, score_func, self.no_grad)



def run_batch_quadtree(num_patches: int,
                        min_patch_size: int,
                        max_patch_size: int,
                        images: FloatTensor,
                        score_func: Union[PatchScorer, Callable],
                        no_grad: bool = True) -> FloatTensor:
    """
    images: [batch, channels, height, width]
    """
    assert (images.ndim == 4) and (images.shape[-2] == images.shape[-1])
    image_size = images.shape[-1]
    assert image_size % max_patch_size == 0
    assert max_patch_size % min_patch_size == 0

    context_manager = torch.no_grad if no_grad else nullcontext
    with context_manager():
        patches_lookup_tables, model_inputs_by_scale = _build_patches_lookup_tables(
            images, min_patch_size, max_patch_size, score_func)

        patch_ids, size_ids = zip(*[
            Quadtree(num_patches, min_patch_size, max_patch_size, image_size,
                     patches_lookup_table).run()
            for patches_lookup_table in patches_lookup_tables
        ])
        patch_ids = tensor(patch_ids, device=images.device)
        size_ids = tensor(size_ids, device=images.device)

        model_inputs = extract_model_inputs(model_inputs_by_scale, patch_ids,
                                            size_ids)
        return model_inputs


class Quadtree:

    def __init__(self, num_patches: int, min_patch_size: int,
                 max_patch_size: int, image_size: int,
                 patches_lookup_table: dict[tuple[int, int, int, int], tuple]):
        self.num_patches = num_patches
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.image_size = image_size
        self.patches_lookup_table = patches_lookup_table

        self.heap = []
        self.leaves = []
        self._initialize_heap()
        assert_valid_quadtree_params(
            num_patches, min_patch_size, max_patch_size, image_size)

    def _initialize_heap(self) -> None:
        initial_quads = [
            Quad(box, self.patches_lookup_table)
            for box in grid_boxes_tuples(self.max_patch_size, self.image_size)
        ]
        for quad in initial_quads:
            self.push(quad)

    def run(self) -> tuple[list[int], list[int]]:
        """
        Every split adds 3 quads, so the actual num_patches may be slightly smaller than self.num_patches

        Returns:
            model_input: [num_patches, channels*min_patch_size**2 + 3] (flat_patch, pos_x, pos_y, size_id)
        """
        while (len(self.heap) + len(self.leaves)) < self.num_patches:
            self.split_worst_quad()

        patch_ids, size_ids = zip(*[(quad.patch_id, quad.size_id)
                                    for quad in self.quads])
        return patch_ids, size_ids

    @property
    def quads(self) -> list["Quad"]:
        return [quad for score, quad in self.heap
                ] + [leaf for leaf in self.leaves if leaf is not None]

    def push(self, quad: "Quad") -> None:
        heappush(self.heap, (quad.score, quad))

    def pop(self) -> "Quad":
        score, quad = heappop(self.heap)
        return quad

    def split_worst_quad(self):
        quad = self.pop()
        children = quad.split()
        for child in children:
            if child.is_leaf:
                self.leaves.append(child)
            else:
                self.push(child)


class Quad:
    __slots__ = ("box", "patches_lookup_table", "score", "is_leaf", "patch_id",
                 "size_id")

    def __init__(self, box: tuple[int, int, int, int],
                 patches_lookup_table: dict):
        self.box = box
        self.patches_lookup_table = patches_lookup_table
        self.score, self.is_leaf, self.patch_id, self.size_id = patches_lookup_table[
            box]

    def split(self) -> tuple["Quad"]:
        l, t, r, b = self.box
        lr = l + (r - l) // 2
        tb = t + (b - t) // 2
        tl = Quad((l, t, lr, tb), self.patches_lookup_table)
        tr = Quad((lr, t, r, tb), self.patches_lookup_table)
        bl = Quad((l, tb, lr, b), self.patches_lookup_table)
        br = Quad((lr, tb, r, b), self.patches_lookup_table)
        children = (tl, tr, bl, br)
        return children

    def __lt__(self, other):
        """ Dummy fallback in case of equal scores. We use tuple comparison instead of object comparison because it's faster. """
        return False


BOX_TYPE = tuple[int, int, int, int]


def _build_patches_lookup_tables(
    images: FloatTensor, min_patch_size: int, max_patch_size: int,
    score_func: Union[PatchScorer, Callable]
) -> tuple[dict[BOX_TYPE, tuple], list[FloatTensor]]:
    """
    returns:
        patches_lookup_tables: for each sample in the batch, a dictionary that maps a (left,right,top,bottom) box tuple to (score, is_leaf, patch_id, size_id).
        model_inputs_by_scale: a list of tensors, one for each scale. shape of each tensor is [batch, num_patches_in_that_scale, input_dim]
                               where input_dim = (h*w*3 + 2 + 1) since it's a concatenation of small_patch, position_id, size_id.
    """
    batch_size, _, image_size, _ = images.shape
    patches_lookup_tables = [{} for _ in range(batch_size)]
    model_inputs_by_scale = []
    for patch_size in patch_arange(min_patch_size, max_patch_size).tolist():
        scores, model_inputs = _create_model_inputs(
            images, min_patch_size, patch_size, score_func)

        if isinstance(scores, Tensor):
            _for_gmacs_count = 0 * scores.mean().to(model_inputs.device)
            model_inputs = model_inputs + _for_gmacs_count
            scores = scores.tolist()

        model_inputs_by_scale.append(model_inputs)
        _update_lookup_tables(
            min_patch_size, patches_lookup_tables, patch_size, image_size, scores)

    return patches_lookup_tables, model_inputs_by_scale


def _update_lookup_tables(min_patch_size, patches_lookup_tables, patch_size, image_size, scores):
    boxes_tuples = grid_boxes_tuples(patch_size, image_size)
    num_patches = len(boxes_tuples)
    patch_ids = range(num_patches)
    size_id_value = _size_id_value(patch_size, min_patch_size)
    is_leaf = (patch_size == min_patch_size)
    batch_size = len(scores)
    for i in range(batch_size):
        dict_values = list(
            zip(scores[i], repeat(is_leaf), patch_ids, repeat(size_id_value)))
        patches_lookup_tables[i].update(zip(boxes_tuples, dict_values))


def _create_model_inputs(images: Tensor, min_patch_size: int, patch_size: int, score_func: Union[PatchScorer, Callable]) -> Tensor:
    scores, small_patches = score_func(images, min_patch_size, patch_size)
    batch_size, _, image_size, _ = images.shape
    box_metadata = create_box_metadata(
        min_patch_size, patch_size, image_size, batch_size, images.device, images.dtype)
    model_inputs = torch.cat([small_patches, box_metadata], dim=-1)
    return scores, model_inputs
