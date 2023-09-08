import numpy as np
import torch
from torch import Tensor

from mixed_res.quadtree_impl.utils import split_model_inputs


def reverse_quadtree(model_inputs: Tensor) -> Tensor:
    """
    This operation turns the results of the Quadtree algorithm into a full-size feature volume,
    effectively replicating the token embedding of each Quadtree token to match its original patch size.
    This can be useful when you want to apply Quadtree-style processing (e.g. with a mixed-resolution ViT or an MLP)
    followed by a more standard model (like a CNN).

    Args:
        model_inputs: [bsz, num_quadtree_patches, embed_dim + 3]
                      A tensor of token embeddings, created by the Quadtree algorithm, and possibly processed
                      by an appropriate network such as a mixed-resolution ViT.

    Returns:
        full features: [bsz, embed_dim, image_size / min_patch_size, image_size / min_patch_size, embed_dim]
                       A full-size feature volume, where each token embedding is duplicated to completely fill its
                       corresponding area. Patches of scale 0 stay the same, patches of scale 1 are replicated 2x2
                       times, patches of scale 2 are replicated 4x4 times, etc.
    """
    flat_patches, centers, size_ids = split_model_inputs(model_inputs)
    bsz, num_patches, dim = flat_patches.shape
    num_scales = len(size_ids.unique())
    quad_left_top = centers - 2 ** (size_ids - 1)
    quad_left, quad_top = quad_left_top[:, :, 0], quad_left_top[:, :, 1]
    quad_inds = torch.arange(num_patches)[None, :].expand(bsz, num_patches)
    batch_inds = torch.arange(bsz)[:, None].expand(bsz, num_patches)
    size_ids, quad_left, quad_top, quad_inds, batch_inds = list(map(
        torch.flatten, [size_ids, quad_left, quad_top, quad_inds, batch_inds]))

    full_i_left, full_i_top, full_i_quad, full_i_batch, full_mask_inds = [], [], [], [], []
    for size_id in range(num_scales):
        size_mask = (size_ids == size_id)
        mask_inds, = torch.nonzero(size_mask, as_tuple=True)
        i_left = quad_left[mask_inds]
        i_top = quad_top[mask_inds]
        i_quad = quad_inds[mask_inds]
        i_batch = batch_inds[mask_inds]
        if size_id != 0:
            arange = torch.arange(2 ** size_id)
            left_diff, top_diff = torch.meshgrid(arange, arange, indexing="xy")
            left_diff, top_diff = left_diff.flatten(), top_diff.flatten()
            i_left = (i_left[:, None] + left_diff[None, :]).flatten()
            i_top = (i_top[:, None] + top_diff[None, :]).flatten()
            i_quad = torch.repeat_interleave(i_quad, (2 ** size_id) ** 2)
            i_batch = torch.repeat_interleave(i_batch, (2 ** size_id) ** 2)
            mask_inds = torch.repeat_interleave(mask_inds, (2 ** size_id) ** 2)
        full_i_left.append(i_left)
        full_i_top.append(i_top)
        full_i_quad.append(i_quad)
        full_i_batch.append(i_batch)
        full_mask_inds.append(mask_inds)

    full_i_left, full_i_top, full_i_quad, full_i_batch, full_mask_inds = list(map(
        lambda x: torch.concat(x, axis=0),
        [full_i_left, full_i_top, full_i_quad, full_i_batch, full_mask_inds]))

    sort_keys = torch.vstack([full_i_left, full_i_top, full_i_batch])
    sort_inds = np.lexsort(sort_keys.cpu().numpy())
    full_features = flat_patches[full_i_batch, full_i_quad, :]
    full_features = full_features[sort_inds, :]
    width = int((full_features.numel() / bsz / dim) ** 0.5)
    full_features = full_features.reshape((bsz, width, width, dim)).permute(0, 3, 1, 2)
    return full_features
