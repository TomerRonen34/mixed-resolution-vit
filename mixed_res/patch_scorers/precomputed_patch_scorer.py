from torch import FloatTensor

from mixed_res.patch_scorers.patch_scorer import PatchScorer


class PrecomputedPatchScorer(PatchScorer):
    def _calculate_importance_maps(self,
                                   images: FloatTensor,
                                   patch_size: int,
                                   importance_maps: FloatTensor,
                                   **unused_kwargs) -> None:
        self._validate_shapes(images, patch_size, importance_maps)
        self.importance_maps = importance_maps

    def _validate_shapes(self,
                         images: FloatTensor,
                         patch_size: int,
                         importance_maps: FloatTensor) -> None:
        batch_size = images.shape[0]
        image_size = images.shape[-1]
        grid_size = image_size // patch_size
        assert importance_maps.shape[0] == batch_size
        assert importance_maps.shape[-1] == importance_maps.shape[-2]
        assert importance_maps.shape[-1] >= grid_size
