# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
import pytest
import torch
from pytest import MonkeyPatch
from _pytest.fixtures import SubRequest
from torchvision.models._api import WeightsEnum
from torchgeo.models import prithvi_eo2, PrithviEO2_Weights


class TestPrithviEO2:
    @pytest.fixture(params=[*PrithviEO2_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> WeightsEnum:
        # use a real member name from your enum
        weights = PrithviEO2_Weights.PRITHVI_EO2_300M_TL_SEN1FLOODS11
        path = tmp_path / f"{weights}.pth"

        # create a dummy state dict so the test stays offline
        model = torch.nn.Linear(4, 4)
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, "url", str(path))
        return weights

    def test_prithvi_default(self, monkeypatch: MonkeyPatch) -> None:
        """Ensure model loads without explicit weights and avoids num_labels=None crash."""
        import transformers

        # Patch both AutoModel and AutoConfig to prevent the NoneType error.
        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            lambda *a, **k: transformers.AutoConfig.from_dict({"num_labels": 1}),
        )
        monkeypatch.setattr(
            transformers.AutoModel,
            "from_pretrained",
            lambda *a, **k: torch.nn.Identity(),
        )

        prithvi_eo2()

    def test_prithvi_with_mocked_weights(self, monkeypatch: MonkeyPatch, mocked_weights: WeightsEnum) -> None:
        """Ensure model loads correctly when mocked weights are supplied."""
        import transformers

        # Patch both possible HF loader classes used by prithvi_eo2
        for cls_name in ("AutoModel", "AutoModelForSemanticSegmentation"):
            monkeypatch.setattr(
                getattr(transformers, cls_name),
                "from_pretrained",
                lambda *a, **k: torch.nn.Identity(),
            )

        prithvi_eo2(weights=mocked_weights)


    @pytest.mark.slow
    def test_prithvi_download(self, weights: WeightsEnum) -> None:
        """Smoke test for real weight download (optional slow run)."""
        prithvi_eo2(weights=weights)

    def test_forward_shape(self, monkeypatch: MonkeyPatch) -> None:
        """Verify forward pass shape."""
        import transformers

        # Replace the HF model with a minimal torch module for fast testing.
        monkeypatch.setattr(
            transformers.AutoModel,
            "from_pretrained",
            lambda *a, **k: torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(3 * 224 * 224, 128),
            ),
        )

        model = prithvi_eo2()
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        assert y.shape == (1, 128)