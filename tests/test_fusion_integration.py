"""
T1D-UOM — Five-GRU → MLP Fusion integration tests.
"""

from __future__ import annotations

import torch

from src.models.five_gru import (
    ACTIVITY_DIM,
    GLUCOSE_DIM,
    INSULIN_DIM,
    NUTRITION_DIM,
    SLEEP_DIM,
    FiveGRU,
)
from src.models.mlp_fusion import (
    MLPFusion,
    validate_unified_patient_state,
)


def test_five_gru_to_mlp_fusion_pipeline() -> None:
    batch_size = 4
    sequence_length = 12
    hidden_dim = 32
    fusion_dim = 64

    five_gru = FiveGRU(
        hidden_dim=hidden_dim,
    )

    fusion = MLPFusion(
        hidden_dim=hidden_dim,
        fusion_dim=fusion_dim,
    )

    inputs = {
        "glucose": torch.randn(
            batch_size,
            sequence_length,
            GLUCOSE_DIM,
        ),
        "insulin": torch.randn(
            batch_size,
            sequence_length,
            INSULIN_DIM,
        ),
        "nutrition": torch.randn(
            batch_size,
            sequence_length,
            NUTRITION_DIM,
        ),
        "activity": torch.randn(
            batch_size,
            sequence_length,
            ACTIVITY_DIM,
        ),
        "sleep": torch.randn(
            batch_size,
            sequence_length,
            SLEEP_DIM,
        ),
    }

    latents = five_gru(**inputs)

    assert set(latents.keys()) == {
        "zG",
        "zI",
        "zN",
        "zA",
        "zS",
    }

    for latent in latents.values():
        assert latent.shape == (
            batch_size,
            hidden_dim,
        )

    unified_state = fusion(**latents)

    validate_unified_patient_state(
        unified_state,
        batch_size=batch_size,
        fusion_dim=fusion_dim,
    )

    assert unified_state.shape == (
        batch_size,
        fusion_dim,
    )