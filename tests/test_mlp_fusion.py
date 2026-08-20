"""
T1D-UOM — MLP Fusion architecture tests.
"""

from __future__ import annotations

import pytest
import torch

from src.models.mlp_fusion import (
    LATENT_NAMES,
    MLPFusion,
    validate_mlp_fusion,
    validate_unified_patient_state,
)


def _make_latents(
    batch_size: int,
    hidden_dim: int,
) -> dict[str, torch.Tensor]:
    return {
        "zG": torch.randn(batch_size, hidden_dim),
        "zI": torch.randn(batch_size, hidden_dim),
        "zN": torch.randn(batch_size, hidden_dim),
        "zA": torch.randn(batch_size, hidden_dim),
        "zS": torch.randn(batch_size, hidden_dim),
    }


def test_exactly_five_latent_inputs() -> None:
    assert LATENT_NAMES == (
        "zG",
        "zI",
        "zN",
        "zA",
        "zS",
    )


def test_mlp_fusion_structure() -> None:
    model = MLPFusion(
        hidden_dim=32,
        fusion_dim=64,
    )

    validate_mlp_fusion(model)


def test_unified_patient_state_shape() -> None:
    batch_size = 4
    hidden_dim = 32
    fusion_dim = 64

    model = MLPFusion(
        hidden_dim=hidden_dim,
        fusion_dim=fusion_dim,
    )

    latents = _make_latents(
        batch_size=batch_size,
        hidden_dim=hidden_dim,
    )

    state = model(**latents)

    validate_unified_patient_state(
        state,
        batch_size=batch_size,
        fusion_dim=fusion_dim,
    )

    assert state.shape == (
        batch_size,
        fusion_dim,
    )


def test_batch_size_must_match() -> None:
    model = MLPFusion(
        hidden_dim=16,
        fusion_dim=32,
    )

    latents = _make_latents(
        batch_size=3,
        hidden_dim=16,
    )

    latents["zS"] = torch.randn(4, 16)

    with pytest.raises(ValueError, match="same batch size"):
        model(**latents)


def test_hidden_dimension_must_match() -> None:
    model = MLPFusion(
        hidden_dim=16,
        fusion_dim=32,
    )

    latents = _make_latents(
        batch_size=3,
        hidden_dim=16,
    )

    latents["zA"] = torch.randn(3, 8)

    with pytest.raises(ValueError, match="hidden dimension"):
        model(**latents)


def test_non_finite_latent_is_rejected() -> None:
    model = MLPFusion(
        hidden_dim=16,
        fusion_dim=32,
    )

    latents = _make_latents(
        batch_size=2,
        hidden_dim=16,
    )

    latents["zN"][0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN or infinite"):
        model(**latents)


def test_non_floating_latent_is_rejected() -> None:
    model = MLPFusion(
        hidden_dim=16,
        fusion_dim=32,
    )

    latents = _make_latents(
        batch_size=2,
        hidden_dim=16,
    )

    latents["zI"] = torch.ones(
        2,
        16,
        dtype=torch.long,
    )

    with pytest.raises(TypeError, match="floating-point"):
        model(**latents)


def test_invalid_latent_set_is_rejected() -> None:
    model = MLPFusion(
        hidden_dim=16,
        fusion_dim=32,
    )

    latents = _make_latents(
        batch_size=2,
        hidden_dim=16,
    )

    latents.pop("zS")
    latents["zX"] = torch.randn(2, 16)

    with pytest.raises(ValueError, match="exactly the five latent"):
        model(**latents)


def test_different_batch_sizes_work_independently() -> None:
    model = MLPFusion(
        hidden_dim=16,
        fusion_dim=32,
    )

    for batch_size in (1, 2, 5, 8):
        latents = _make_latents(
            batch_size=batch_size,
            hidden_dim=16,
        )

        state = model(**latents)

        assert state.shape == (
            batch_size,
            32,
        )