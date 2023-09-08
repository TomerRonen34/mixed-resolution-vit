import math

import torch
from PIL import Image
from torchvision import transforms

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def transforms_imagenet_eval(
        img_size=224,
        crop_pct=DEFAULT_CROP_PCT,
        interpolation=transforms.InterpolationMode.BILINEAR,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):
    """
    Based on the implementation in timm:
    https://github.com/huggingface/pytorch-image-models/blob/20a1fa63f8ea999dab29d927d5e1866ed3b67348/timm/data/transforms_factory.py#L130
    """

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        transforms.Resize(
            scale_size, interpolation=interpolation),
        transforms.CenterCrop(img_size),
    ]
    tfl += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(mean),
            std=torch.tensor(std))
    ]

    return transforms.Compose(tfl)


def hstack_images(images: list[Image.Image], gap: int = 20) -> Image.Image:
    prev_concat = images[0]
    for image in images[1:]:
        concat = Image.new("RGB", (prev_concat.width + gap +
                           image.width, image.height), (255, 255, 255))
        concat.paste(prev_concat, (0, 0))
        concat.paste(image, (prev_concat.width + gap, 0))
        prev_concat = concat
    return concat


def vstack_images(images: list[Image.Image], gap: int = 20) -> Image.Image:
    prev_concat = images[0]
    for image in images[1:]:
        concat = Image.new(
            "RGB", (prev_concat.width, prev_concat.height + gap + image.height), (255, 255, 255))
        concat.paste(prev_concat, (0, 0))
        concat.paste(image, (0, prev_concat.height + gap))
        prev_concat = concat
    return concat
