from __future__ import annotations

import pytest
import torch

from src.models.five_gru import FiveGRU
from src.models.five_gru_pipeline import (
    FiveGRUInputBatch,
    FiveGRUStatePipeline,
)
from src.models.mlp_fusion import MLPFusion
from src.models.patient_state import UnifiedPatientState


def _pipeline() -> FiveGRUStatePipeline:
    five_gru = FiveGRU(
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    )

    fusion = MLPFusion(
        hidden_dim=16,
        fusion_dim=32,
        dropout=0.0,
    )

    return FiveGRUStatePipeline(
        five_gru=five_gru,
        fusion=fusion,
    )


def _inputs(
    *,
    batch_size: int = 2,
    sequence_length: int = 5,
) -> FiveGRUInputBatch:
    return FiveGRUInputBatch(
        glucose=torch.randn(
            batch_size,
            sequence_length,
            1,
        ),
        insulin=torch.randn(
            batch_size,
            sequence_length,
            2,
        ),
        nutrition=torch.randn(
            batch_size,
            sequence_length,
            24,
        ),
        activity=torch.randn(
            batch_size,
            sequence_length,
            17,
        ),
        sleep=torch.randn(
            batch_size,
            sequence_length,
            6,
        ),
    )


def test_pipeline_returns_unified_patient_state() -> None:
    torch.manual_seed(42)

    model = _pipeline()

    state = model(
        _inputs()
    )

    assert isinstance(
        state,
        UnifiedPatientState,
    )

    assert state.state.shape == (
        2,
        32,
    )

    assert torch.isfinite(
        state.state
    ).all()


def test_pipeline_preserves_batch_dimension() -> None:
    model = _pipeline()

    state = model(
        _inputs(
            batch_size=4,
            sequence_length=7,
        )
    )

    assert state.state.shape == (
        4,
        32,
    )


def test_pipeline_accepts_different_padded_sequence_lengths() -> None:
    torch.manual_seed(42)

    model = _pipeline()

    inputs = FiveGRUInputBatch(
        glucose=torch.randn(2, 8, 1),
        insulin=torch.randn(2, 5, 2),
        nutrition=torch.randn(2, 7, 24),
        activity=torch.randn(2, 10, 17),
        sleep=torch.randn(2, 6, 6),
    )

    state = model(inputs)

    assert state.state.shape == (
        2,
        32,
    )

    assert torch.isfinite(
        state.state
    ).all()


def test_pipeline_accepts_independent_modality_lengths() -> None:
    torch.manual_seed(42)

    model = _pipeline()

    inputs = FiveGRUInputBatch(
        glucose=torch.randn(2, 8, 1),
        insulin=torch.randn(2, 6, 2),
        nutrition=torch.randn(2, 7, 24),
        activity=torch.randn(2, 10, 17),
        sleep=torch.randn(2, 5, 6),
        lengths={
            "glucose": torch.tensor([8, 5]),
            "insulin": torch.tensor([6, 4]),
            "nutrition": torch.tensor([7, 3]),
            "activity": torch.tensor([10, 8]),
            "sleep": torch.tensor([5, 2]),
        },
    )

    state = model(inputs)

    assert state.state.shape == (
        2,
        32,
    )


def test_pipeline_rejects_wrong_nutrition_dimension() -> None:
    model = _pipeline()

    inputs = _inputs()

    inputs = FiveGRUInputBatch(
        glucose=inputs.glucose,
        insulin=inputs.insulin,
        nutrition=torch.randn(2, 5, 23),
        activity=inputs.activity,
        sleep=inputs.sleep,
    )

    with pytest.raises(
        ValueError,
        match="nutrition expected 24",
    ):
        model(inputs)


def test_pipeline_rejects_wrong_activity_dimension() -> None:
    model = _pipeline()

    inputs = _inputs()

    inputs = FiveGRUInputBatch(
        glucose=inputs.glucose,
        insulin=inputs.insulin,
        nutrition=inputs.nutrition,
        activity=torch.randn(2, 5, 16),
        sleep=inputs.sleep,
    )

    with pytest.raises(
        ValueError,
        match="activity expected 17",
    ):
        model(inputs)


def test_pipeline_rejects_batch_mismatch() -> None:
    model = _pipeline()

    inputs = _inputs()

    inputs = FiveGRUInputBatch(
        glucose=inputs.glucose,
        insulin=torch.randn(3, 5, 2),
        nutrition=inputs.nutrition,
        activity=inputs.activity,
        sleep=inputs.sleep,
    )

    with pytest.raises(
        ValueError,
        match="same batch size",
    ):
        model(inputs)


def test_pipeline_rejects_non_tensor_lengths() -> None:
    model = _pipeline()

    inputs = FiveGRUInputBatch(
        glucose=torch.randn(2, 5, 1),
        insulin=torch.randn(2, 5, 2),
        nutrition=torch.randn(2, 5, 24),
        activity=torch.randn(2, 5, 17),
        sleep=torch.randn(2, 5, 6),
        lengths={
            "glucose": [5, 4],  # type: ignore[dict-item]
        },
    )

    with pytest.raises(
        TypeError,
        match="must be a torch.Tensor",
    ):
        model(inputs)


def test_pipeline_rejects_wrong_length_batch_dimension() -> None:
    model = _pipeline()

    inputs = FiveGRUInputBatch(
        glucose=torch.randn(2, 5, 1),
        insulin=torch.randn(2, 5, 2),
        nutrition=torch.randn(2, 5, 24),
        activity=torch.randn(2, 5, 17),
        sleep=torch.randn(2, 5, 6),
        lengths={
            "glucose": torch.tensor([5, 4, 3]),
        },
    )

    with pytest.raises(
        ValueError,
        match="batch dimension",
    ):
        model(inputs)


def test_pipeline_rejects_zero_length() -> None:
    model = _pipeline()

    inputs = FiveGRUInputBatch(
        glucose=torch.randn(2, 5, 1),
        insulin=torch.randn(2, 5, 2),
        nutrition=torch.randn(2, 5, 24),
        activity=torch.randn(2, 5, 17),
        sleep=torch.randn(2, 5, 6),
        lengths={
            "glucose": torch.tensor([5, 0]),
        },
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        model(inputs)


def test_pipeline_rejects_length_exceeding_padding() -> None:
    model = _pipeline()

    inputs = FiveGRUInputBatch(
        glucose=torch.randn(2, 5, 1),
        insulin=torch.randn(2, 5, 2),
        nutrition=torch.randn(2, 5, 24),
        activity=torch.randn(2, 5, 17),
        sleep=torch.randn(2, 5, 6),
        lengths={
            "glucose": torch.tensor([5, 6]),
        },
    )

    with pytest.raises(
        ValueError,
        match="greater than",
    ):
        model(inputs)


def test_pipeline_rejects_unexpected_length_key() -> None:
    model = _pipeline()

    inputs = FiveGRUInputBatch(
        glucose=torch.randn(2, 5, 1),
        insulin=torch.randn(2, 5, 2),
        nutrition=torch.randn(2, 5, 24),
        activity=torch.randn(2, 5, 17),
        sleep=torch.randn(2, 5, 6),
        lengths={
            "glucose": torch.tensor([5, 4]),
            "unknown": torch.tensor([5, 4]),
        },
    )

    with pytest.raises(
        ValueError,
        match="Unexpected sequence-length keys",
    ):
        model(inputs)


def test_pipeline_is_deterministic_in_eval_mode() -> None:
    torch.manual_seed(42)

    model = _pipeline()
    model.eval()

    inputs = _inputs()

    with torch.no_grad():
        first = model(inputs).state
        second = model(inputs).state

    assert torch.equal(
        first,
        second,
    )


def test_pipeline_requires_five_gru() -> None:
    fusion = MLPFusion(
        hidden_dim=16,
        fusion_dim=32,
    )

    with pytest.raises(
        TypeError,
        match="FiveGRU",
    ):
        FiveGRUStatePipeline(
            five_gru=torch.nn.Linear(
                1,
                1,
            ),  # type: ignore[arg-type]
            fusion=fusion,
        )


def test_pipeline_requires_matching_hidden_dimension() -> None:
    five_gru = FiveGRU(
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    )

    fusion = MLPFusion(
        hidden_dim=32,
        fusion_dim=32,
    )

    with pytest.raises(
        ValueError,
        match="hidden_dim",
    ):
        FiveGRUStatePipeline(
            five_gru=five_gru,
            fusion=fusion,
        )