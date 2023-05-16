from typing import Literal, Optional

import torch
import torchvision
from torch import FloatTensor, nn
from torch.nn.functional import cosine_similarity
from torchvision.models.shufflenetv2 import (ShuffleNet_V2_X0_5_Weights,
                                             ShuffleNetV2, shufflenet_v2_x0_5)

from mixed_res.patch_scorers.patch_scorer import PatchScorer


class FeatureBasedPatchScorer(PatchScorer):
    def __init__(self,
                 num_scales: int = 3,
                 small_patch_size: int = 16,
                 model: Optional[nn.Module] = None,
                 device: str = "cuda",
                 downscale_factor: int = 32,
                 error_method: Literal["mse", "cosine_distance"] = "mse",
                 high_res_fraction: float = 1.0,
                 ):
        super().__init__()
        device = torch.ones(1, device=device).device
        self.device = device

        self.num_scales = num_scales
        self.small_patch_size = small_patch_size
        self.downscale_factor = downscale_factor
        self.error_method = error_method
        self.high_res_fraction = high_res_fraction

        self._setup_model(model)

        self.blur_factors = [2**scale for scale in range(1, self.num_scales)]

    def _setup_model(self, model: Optional[nn.Module]):
        if model is not None:
            self.model = model
        else:
            self.model = shufflenet_v2_x0_5(
                weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)

        if isinstance(self.model, ShuffleNetV2) and hasattr(self.model, "fc"):
            del self.model.fc  # saves GPU space

        model_device = next(self.model.parameters()).device
        if self.device != model_device:
            self.model = self.model.to(self.device)

        self.model = self.model.eval()

    def _extract_features(self, images: FloatTensor) -> FloatTensor:
        if isinstance(self.model, ShuffleNetV2):
            features = _extract_features_shufflenet(
                self.model, images, self.downscale_factor)
        else:
            features = self.model(images)

        assert features.shape[-1] == (images.shape[-1] //
                                      self.downscale_factor)
        return features

    @torch.no_grad()
    def _calculate_importance_maps(self, images: FloatTensor, small_patch_size: int, **kwargs) -> None:
        assert small_patch_size == self.small_patch_size
        image_size = images.shape[2]

        high_res_size = int(image_size * self.high_res_fraction)
        high_res_images = images if (
            high_res_size == image_size) else resize(images, [high_res_size])

        multiscale_batch = [high_res_images]
        for blur_factor in self.blur_factors:
            downsample_size = int(image_size / blur_factor)
            blurry_images = resize(
                resize(images, [downsample_size]), [high_res_size])
            multiscale_batch.append(blurry_images)

        features = [self._extract_features(x) for x in multiscale_batch]

        high_res_features = features[0]
        importance_maps = []
        for scale in range(1, len(self.blur_factors) + 1):
            low_res_features = features[scale]
            if self.error_method == "mse":
                squared_error = (low_res_features - high_res_features) ** 2
                error = squared_error.mean(dim=1)
            elif self.error_method == "cosine_distance":
                error = - \
                    cosine_similarity(low_res_features,
                                      high_res_features, dim=1)
            else:
                raise ValueError(self.error_method)
            importance_maps.append(error)

        if high_res_size != image_size:
            importance_maps = torch.stack(importance_maps, dim=1)
            importance_maps_size = image_size // self.downscale_factor
            importance_maps = resize(
                importance_maps, [importance_maps_size])

        self.importance_maps = importance_maps


def _extract_features_shufflenet(net: nn.Module, x: FloatTensor, downscale_factor: int) -> FloatTensor:
    """ downscale factor x32 or x16 """
    assert downscale_factor in (16, 32)
    x = net.conv1(x)
    x = net.maxpool(x)
    x = net.stage2(x)
    x = net.stage3(x)
    if downscale_factor == 32:
        x = net.stage4(x)
        x = net.conv5(x)
    return x


def resize(*args, **kwargs) -> FloatTensor:
    return torchvision.transforms.functional.resize(*args, **kwargs, antialias=None)
