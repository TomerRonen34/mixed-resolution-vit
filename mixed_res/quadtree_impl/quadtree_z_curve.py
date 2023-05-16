from contextlib import nullcontext
from typing import Callable, Union

import torch
from torch import FloatTensor

from mixed_res.patch_scorers.patch_scorer import PatchScorer
from mixed_res.quadtree_impl.quadtree_runner import QuadtreeRunner
from mixed_res.quadtree_impl.utils import (create_box_metadata,
                                           extract_model_inputs, patch_arange)
from mixed_res.quadtree_impl.z_curve_patch_chooser import QuadtreePatchChooser


class ZCurveQuadtreeRunner(QuadtreeRunner):
    def __init__(self, num_patches: int, min_patch_size: int, max_patch_size: int, no_grad: bool = True):
        super().__init__(num_patches, min_patch_size, max_patch_size, no_grad)
        self.patch_arange = patch_arange(min_patch_size, max_patch_size)
        self.quadtree_patch_chooser = None
        self.box_metadata_by_scale = None
        self.input_params = None

    def _init_input_specific_members(self, images: FloatTensor):
        self.input_params = (images.shape, images.device, images.dtype)
        batch_size, _, image_size, _ = images.shape
        device = images.device
        dtype = images.dtype
        del self.quadtree_patch_chooser
        del self.box_metadata_by_scale
        self.quadtree_patch_chooser = QuadtreePatchChooser(
            self.num_patches, self.min_patch_size, self.max_patch_size, image_size, batch_size, device)
        self.box_metadata_by_scale = [create_box_metadata(self.min_patch_size, patch_size, image_size, batch_size, device, dtype)
                                      for patch_size in self.patch_arange]

    def run_batch_quadtree(self, images: FloatTensor, score_func: Union[PatchScorer, Callable]) -> FloatTensor:
        with self.context_manager_func():
            image_params = (images.shape, images.device, images.dtype)
            if self.input_params != image_params:
                self._init_input_specific_members(images)

            scores, small_patches_by_scale = zip(*[
                score_func(images, self.min_patch_size, patch_size)
                for patch_size in self.patch_arange])

            model_inputs_by_scale = [torch.cat([small_patches, box_metadata], dim=-1)
                                     for small_patches, box_metadata in zip(small_patches_by_scale, self.box_metadata_by_scale)]

            patch_ids, size_ids = self.quadtree_patch_chooser.run(scores)

            model_inputs = extract_model_inputs(
                model_inputs_by_scale, patch_ids, size_ids)
            return model_inputs
