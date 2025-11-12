# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Prithvi models.

This module provides lightweight wrappers to load IBM–NASA Prithvi foundation
models from the Hugging Face Hub using a TorchGeo-friendly API, mirroring the
approach used for the Aurora integration.

Supported families (initial):
  * ibm-nasa-geospatial/Prithvi-EO-2.0-300M (encoder)
  * ibm-nasa-geospatial/Prithvi-EO-2.0-600M (encoder)
  * ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL (segmentation finetune)
  * ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL (segmentation finetune)
  * ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11 (task-specific)
  * ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars (task-specific)

You can extend the enum below with more checkpoints as they are released.

Note:
- We use `transformers` AutoModel loaders with `trust_remote_code=True` to allow
  repository-specific model classes. If the repo exposes a custom API, extend the
  loader logic accordingly.
- Prithvi operates on multispectral, sometimes multi-temporal chips. We keep the
  default transform as identity here; projects can inject custom transforms.
"""

from __future__ import annotations

from typing import Any, cast

import torch.nn as nn
from torchvision.models._api import Weights, WeightsEnum

from ..datasets.utils import lazy_import


# -----------------------------------------------------------------------------
# Transforms and metadata
# -----------------------------------------------------------------------------

_prithvi_transforms = nn.Identity()  # Placeholder; supply normalization upstream

_prithvi_meta_base = {
    "dataset": "HLS",
    "model": None,
    "resolution": 30,  # meters per pixel (HLS effective grid)
    "bands": ("B2", "B3", "B4", "B5", "B6", "B7"),  # HLS/S2-like set used in paper
    "patch_size": 16,  # MAE/ViT tubelet spatial size (typical)
    "architecture": "Masked Autoencoder Transformer (ViT backbone)",
    "encoder": "ViT",
    "hf_repo": None,
    "filename": None,
    "publication": "https://arxiv.org/abs/2310.18660",
    "repo": "https://github.com/NASA-IMPACT/hls-foundation-os",
    "license": "Apache-2.0",
    "task": "encoder",  # "encoder" or "segmentation"
}


class PrithviEO2_Weights(WeightsEnum):  # type: ignore[misc]
    """Prithvi EO 2.0 weights.

    If you use these models, please cite:
    * https://arxiv.org/abs/2310.18660 (Foundation Models for Generalist Geospatial AI)

    .. versionadded:: 0.8
    """

    # Encoder-only feature extractors
    PRITHVI_EO2_300M = Weights(
        url="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",  # use HF repo id directly
        transforms=_prithvi_transforms,
        meta=
        _prithvi_meta_base
        | {
            "hf_repo": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
            "model": "AutoModel",  # resolved via transformers AutoModel
            "task": "encoder",
        },
    )

    PRITHVI_EO2_600M = Weights(
        url="ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
        transforms=_prithvi_transforms,
        meta=
        _prithvi_meta_base
        | {
            "hf_repo": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
            "model": "AutoModel",
            "task": "encoder",
        },
    )

    # Task- or head-tuned variants (TL = task learning/finetuned)
    PRITHVI_EO2_100M_TL = Weights(
        url="ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL",
        transforms=_prithvi_transforms,
        meta=
        _prithvi_meta_base
        | {
            "hf_repo": "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL",
            "model": "AutoModelForSemanticSegmentation",
            "task": "segmentation",
        },
    )

    PRITHVI_EO2_TINY_TL = Weights(
        url="ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL",
        transforms=_prithvi_transforms,
        meta=
        _prithvi_meta_base
        | {
            "hf_repo": "ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL",
            "model": "AutoModelForSemanticSegmentation",
            "task": "segmentation",
        },
    )

    # Task-specific heads
    PRITHVI_EO2_300M_TL_SEN1FLOODS11 = Weights(
        url="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
        transforms=_prithvi_transforms,
        meta=
        _prithvi_meta_base
        | {
            "hf_repo": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
            "model": "AutoModelForSemanticSegmentation",
            "task": "segmentation",
            "dataset": "Sen1Floods11",
        },
    )

    PRITHVI_EO2_300M_TL_BURNSCARS = Weights(
        url="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
        transforms=_prithvi_transforms,
        meta=
        _prithvi_meta_base
        | {
            "hf_repo": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
            "model": "AutoModelForSemanticSegmentation",
            "task": "segmentation",
            "dataset": "MTBS Burn Scars",
        },
    )


# -----------------------------------------------------------------------------
# Loader function
# -----------------------------------------------------------------------------

def prithvi_eo2(
    weights: PrithviEO2_Weights | None = None, *args: Any, **kwargs: Any
) -> nn.Module:
    """Load a Prithvi EO 2.0 model from Hugging Face.

    If you use this model in your research, please cite:
    * https://arxiv.org/abs/2310.18660

    This loader requires the following additional library to be installed:
    * `transformers <https://pypi.org/project/transformers/>`_ to load models from HF.

    .. versionadded:: 0.8

    Args:
        weights: Pre-trained model weights to use (select from :class:`PrithviEO2_Weights`).
        *args: Additional positional args forwarded to the underlying model constructor.
        **kwargs: Additional keyword args forwarded to the underlying model constructor.

    Returns:
        A PyTorch ``nn.Module`` wrapping the Prithvi model.
    """

    transformers = lazy_import("transformers")

    if weights is None:
        # Default to the smallest encoder as a sensible base
        repo_id = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
        model = transformers.AutoModel.from_pretrained(
            repo_id, trust_remote_code=True, *args, **kwargs
        )
        return cast(nn.Module, model)

    repo_id: str = weights.meta["hf_repo"]  # type: ignore[assignment]
    task: str = weights.meta["task"]  # "encoder" or "segmentation"
    model_kind: str = weights.meta["model"]  # AutoModel or AutoModelForSemanticSegmentation

    if task == "encoder" or model_kind == "AutoModel":
        model = transformers.AutoModel.from_pretrained(
            repo_id, trust_remote_code=True, *args, **kwargs
        )
    elif task == "segmentation" or model_kind == "AutoModelForSemanticSegmentation":
        model = transformers.AutoModelForSemanticSegmentation.from_pretrained(
            repo_id, trust_remote_code=True, *args, **kwargs
        )
    else:
        # Fallback: try generic AutoModel
        model = transformers.AutoModel.from_pretrained(
            repo_id, trust_remote_code=True, *args, **kwargs
        )

    return cast(nn.Module, model)


__all__ = [
    "PrithviEO2_Weights",
    "prithvi_eo2",
]
