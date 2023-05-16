import abc
from typing import Callable, Optional, Union

import torch
from torch import Tensor

from mixed_res.patch_scorers.patch_scorer import PatchScorer
from mixed_res.quadtree_impl.quadtree_runner import QuadtreeRunner
from mixed_res.tokenization.patch_embed import FlatPatchEmbed, PatchEmbed
from mixed_res.tokenization.pos_embed import build_sinusoidal_embeddings


class ImageTokenizer(metaclass=abc.ABCMeta):
    def __init__(self,
                 patch_embed: PatchEmbed,
                 cls_token: Optional[Tensor],
                 ):
        self.patch_embed = patch_embed
        self.cls_token = cls_token

    @abc.abstractmethod
    def tokenize(self, images: Tensor) -> Tensor:
        pass

    @staticmethod
    def _create_pos_embeds(token_embeds: Tensor, xy_pos: Tensor) -> Tensor:
        batch_size, num_patches, embed_dim = token_embeds.shape
        if xy_pos.shape[0] == 1:
            xy_pos.expand(batch_size, -1, -1)
        x_pos, y_pos = xy_pos.unbind(dim=-1)
        x_embeds = build_sinusoidal_embeddings(
            x_pos, embed_dim // 2)
        y_embeds = build_sinusoidal_embeddings(
            y_pos, embed_dim // 2)
        pos_embeds = torch.cat([x_embeds, y_embeds], dim=-1)
        return pos_embeds

    def _concat_cls_token(self, token_embeds: Tensor):
        if self.cls_token is not None:
            batch_size = token_embeds.shape[0]
            cls_token = self.cls_token.view(
                1, 1, -1).expand(batch_size, -1, -1)
            token_embeds = torch.cat([cls_token, token_embeds], dim=1)
        return token_embeds


class QuadtreeTokenizer(ImageTokenizer):
    def __init__(self,
                 patch_embed: FlatPatchEmbed,
                 cls_token: Optional[Tensor],
                 quadtree_runner: QuadtreeRunner,
                 score_func: Union[PatchScorer, Callable],
                 ):
        assert isinstance(patch_embed, FlatPatchEmbed)
        super().__init__(patch_embed, cls_token)
        self.quadtree_runner = quadtree_runner
        self.score_func = score_func

    def tokenize(self, images: Tensor) -> Tensor:
        model_inputs = self.quadtree_runner.run_batch_quadtree(
            images, self.score_func)
        flat_patches = model_inputs[:, :, :-3]
        xy_pos = model_inputs[:, :, -3:-1]  # discard size id

        token_embeds = self.patch_embed(flat_patches)

        token_embeds = token_embeds + \
            self._create_pos_embeds(token_embeds, xy_pos)

        token_embeds = self._concat_cls_token(token_embeds)

        return token_embeds


class VanillaTokenizer(ImageTokenizer):
    def __init__(self,
                 patch_embed: FlatPatchEmbed,
                 cls_token: Optional[Tensor]):
        super().__init__(patch_embed, cls_token)
        self._xy_pos = self._create_uniform_grid_positions()

    def tokenize(self, images: Tensor) -> Tensor:
        if self._xy_pos.device != images.device:
            self._xy_pos = self._xy_pos.to(images.device)

        token_embeds = self.patch_embed(images)

        token_embeds = token_embeds + \
            self._create_pos_embeds(token_embeds, self._xy_pos)

        token_embeds = self._concat_cls_token(token_embeds)

        return token_embeds

    def _create_uniform_grid_positions(self):
        image_size = self.patch_embed.img_size[0]
        patch_size = self.patch_embed.patch_size[0]
        grid_size = image_size // patch_size

        arange = torch.arange(grid_size)
        xy_pos = torch.fliplr(torch.cartesian_prod(arange, arange))
        xy_pos = xy_pos.unsqueeze(0)  # [1, num_patches, 2]
        return xy_pos
