import abc
from contextlib import nullcontext
from typing import Callable, Union

import torch
from torch import FloatTensor

from mixed_res.patch_scorers.patch_scorer import PatchScorer


class QuadtreeRunner(metaclass=abc.ABCMeta):
    def __init__(self, num_patches: int, min_patch_size: int, max_patch_size: int, no_grad: bool = True):
        self.num_patches = num_patches
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.no_grad = no_grad
        self.context_manager_func = torch.no_grad if no_grad else nullcontext

    @abc.abstractmethod
    def run_batch_quadtree(self, images: FloatTensor, score_func: Union[PatchScorer, Callable]) -> FloatTensor:
        pass
