from torch import nn

from helpers import to_2tuple


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding.
    Based on the implementation in timm:
    https://github.com/huggingface/pytorch-image-models/blob/20a1fa63f8ea999dab29d927d5e1866ed3b67348/timm/layers/patch_embed.py#L24

    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class FlatPatchEmbed(PatchEmbed):
    """
    Flattened patches to patch embedding - used in Quadformer models.
    Instead of using a Linear layer, we reshape the flat vectors to small patch images and use a convolution layer.
    This may seem weird, but it's useful for using pretrained weights from a vanilla Transformer.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten=True)

    def forward(self, x):
        """ x: [batch_size, num_patches, flat_patch_dim] """
        batch_size, num_patches = x.shape[:2]
        patches_as_images = x.view(
            (batch_size * num_patches, self.in_chans, self.patch_size[0], self.patch_size[1]))
        projected = super().forward(patches_as_images)
        projected = projected.view((batch_size, num_patches, self.embed_dim))
        return projected
