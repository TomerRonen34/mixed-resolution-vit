from torch import FloatTensor, Tensor
from torch.nn.functional import unfold


def patchify(images: Tensor, patch_size: int) -> Tensor:
    """
    returns [batch, num_patches, channels, patch_size, patch_size] in raster scan order.
    much faster than using torch.unfold().
    """
    bsz, embed_dim, height, width = images.shape
    assert width == height, f"Only square tensors are supported, got height={height}, width={width}"
    grid_size = width // patch_size
    num_patches = int(grid_size**2)
    patches = images.view(bsz, embed_dim, grid_size, patch_size, -1
                          ).transpose(-2, -1
                                      ).reshape(bsz, embed_dim, num_patches, patch_size, patch_size
                                                ).permute(0, 2, 1, 4, 3)
    return patches


def patchify_flat(images: Tensor, patch_size: int) -> Tensor:
    """
    returns [batch, num_patches, flat_patch_dim] in raster scan order.
    flat_patch_dim = channels * (patch_size ** 2)
    much faster than using torch.unfold().
    """
    patches = patchify(images, patch_size)
    flat_patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
    return flat_patches


def patchify_with_unfold(images: FloatTensor, patch_size: int) -> FloatTensor:
    """ returns [batch, num_patches, channels, patch_size, patch_size] in raster scan order """
    flat_patches = patchify_flat_with_unfold(images, patch_size)
    batch_size, channels = images.shape[:2]
    num_patches = flat_patches.shape[1]
    patches = flat_patches.view(batch_size, num_patches, channels, patch_size,
                                patch_size)
    return patches


def patchify_flat_with_unfold(images: FloatTensor, patch_size: int) -> FloatTensor:
    """
    returns [batch, num_patches, flat_patch_dim] in raster scan order.
    flat_patch_dim = channels * (patch_size ** 2)
    """
    flat_patches = unfold(images, kernel_size=patch_size, stride=patch_size)
    flat_patches = flat_patches.transpose(-1, -2)
    return flat_patches
