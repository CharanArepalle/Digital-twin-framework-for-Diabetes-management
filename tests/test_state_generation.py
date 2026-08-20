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
from src.pipeline.state_generation import (
    GeneratedPatientState,
    StateGenerationResult,
    generate_patient_states,
)


def _pipeline() -> FiveGRUStatePipeline:
    return FiveGRUStatePipeline(
        five_gru=FiveGRU(
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
        ),
        fusion=MLPFusion(
            hidden_dim=16,
            fusion_dim=32,
            dropout=0.0,
        ),
    )


def _inputs(
    *,
    batch_size: int = 3,
    sequence_length: int = 6,
) -> FiveGRUInputBatch:
    return FiveGRUInputBatch(
        glucose=torch.randn(batch_size, sequence_length, 1),
        insulin=torch.randn(batch_size, sequence_length, 2),
        nutrition=torch.randn(batch_size, sequence_length, 24),
        activity=torch.randn(batch_size, sequence_length, 17),
        sleep=torch.randn(batch_size, sequence_length, 6),
    )


def test_generate_patient_states_returns_expected_result() -> None:
    torch.manual_seed(42)

    result = generate_patient_states(
        _pipeline(),
        _inputs(),
    )

    assert isinstance(result, StateGenerationResult)
    assert result.batch_size == 3
    assert result.state_dim == 32
    assert result.count == 3


def test_generated_states_have_correct_structure() -> None:
    result = generate_patient_states(
        _pipeline(),
        _inputs(batch_size=4),
    )

    assert len(result.states) == 4

    for index, item in enumerate(result.states):
        assert isinstance(item, GeneratedPatientState)
        assert item.index == index
        assert item.state.shape == (32,)
        assert torch.isfinite(item.state).all()


def test_stacked_states_have_expected_shape() -> None:
    result = generate_patient_states(
        _pipeline(),
        _inputs(batch_size=5),
    )

    assert result.stacked().shape == (5, 32)


def test_generated_states_match_pipeline_output() -> None:
    torch.manual_seed(42)

    pipeline = _pipeline()
    pipeline.eval()
    inputs = _inputs()

    with torch.no_grad():
        expected = pipeline(inputs).state

    actual = generate_patient_states(
        pipeline,
        inputs,
    ).stacked()

    assert torch.allclose(
        actual,
        expected,
    )


def test_independent_modality_lengths_are_supported() -> None:
    torch.manual_seed(42)

    pipeline = _pipeline()

    inputs = FiveGRUInputBatch(
        glucose=torch.randn(3, 8, 1),
        insulin=torch.randn(3, 5, 2),
        nutrition=torch.randn(3, 7, 24),
        activity=torch.randn(3, 10, 17),
        sleep=torch.randn(3, 6, 6),
        lengths={
            "glucose": torch.tensor([8, 6, 4]),
            "insulin": torch.tensor([5, 4, 3]),
            "nutrition": torch.tensor([7, 5, 2]),
            "activity": torch.tensor([10, 8, 6]),
            "sleep": torch.tensor([6, 4, 2]),
        },
    )

    result = generate_patient_states(
        pipeline,
        inputs,
    )

    assert result.batch_size == 3
    assert result.state_dim == 32
    assert result.stacked().shape == (3, 32)


def test_different_padded_lengths_without_lengths_are_supported() -> None:
    pipeline = _pipeline()

    inputs = FiveGRUInputBatch(
        glucose=torch.randn(2, 8, 1),
        insulin=torch.randn(2, 5, 2),
        nutrition=torch.randn(2, 7, 24),
        activity=torch.randn(2, 10, 17),
        sleep=torch.randn(2, 6, 6),
    )

    result = generate_patient_states(
        pipeline,
        inputs,
    )

    assert result.stacked().shape == (2, 32)


def test_generation_is_deterministic_in_eval_mode() -> None:
    torch.manual_seed(42)

    pipeline = _pipeline()
    pipeline.eval()
    inputs = _inputs()

    first = generate_patient_states(
        pipeline,
        inputs,
    ).stacked()

    second = generate_patient_states(
        pipeline,
        inputs,
    ).stacked()

    assert torch.equal(first, second)


def test_generation_does_not_require_gradients() -> None:
    result = generate_patient_states(
        _pipeline(),
        _inputs(),
    )

    for item in result.states:
        assert item.state.requires_grad is False


def test_generation_rejects_wrong_pipeline_type() -> None:
    with pytest.raises(
        TypeError,
        match="FiveGRUStatePipeline",
    ):
        generate_patient_states(
            torch.nn.Linear(1, 1),  # type: ignore[arg-type]
            _inputs(),
        )


def test_generation_rejects_wrong_input_type() -> None:
    with pytest.raises(
        TypeError,
        match="FiveGRUInputBatch",
    ):
        generate_patient_states(
            _pipeline(),
            torch.randn(2, 5, 1),  # type: ignore[arg-type]
        )


def test_generation_rejects_batch_size_mismatch() -> None:
    inputs = _inputs()

    invalid = FiveGRUInputBatch(
        glucose=inputs.glucose,
        insulin=torch.randn(4, 6, 2),
        nutrition=inputs.nutrition,
        activity=inputs.activity,
        sleep=inputs.sleep,
    )

    with pytest.raises(
        ValueError,
        match="same batch size",
    ):
        generate_patient_states(
            _pipeline(),
            invalid,
        )


def test_generation_rejects_invalid_length_batch() -> None:
    inputs = FiveGRUInputBatch(
        glucose=torch.randn(2, 6, 1),
        insulin=torch.randn(2, 5, 2),
        nutrition=torch.randn(2, 7, 24),
        activity=torch.randn(2, 8, 17),
        sleep=torch.randn(2, 5, 6),
        lengths={
            "glucose": torch.tensor([6, 5, 4]),
        },
    )

    with pytest.raises(
        ValueError,
        match="batch dimension",
    ):
        generate_patient_states(
            _pipeline(),
            inputs,
        )


def test_generation_rejects_length_exceeding_padding() -> None:
    inputs = FiveGRUInputBatch(
        glucose=torch.randn(2, 6, 1),
        insulin=torch.randn(2, 5, 2),
        nutrition=torch.randn(2, 7, 24),
        activity=torch.randn(2, 8, 17),
        sleep=torch.randn(2, 5, 6),
        lengths={
            "glucose": torch.tensor([6, 7]),
        },
    )

    with pytest.raises(
        ValueError,
        match="greater than",
    ):
        generate_patient_states(
            _pipeline(),
            inputs,
        )


def test_generation_rejects_empty_batch() -> None:
    inputs = FiveGRUInputBatch(
        glucose=torch.empty(0, 6, 1),
        insulin=torch.empty(0, 5, 2),
        nutrition=torch.empty(0, 7, 24),
        activity=torch.empty(0, 8, 17),
        sleep=torch.empty(0, 5, 6),
    )

    with pytest.raises(
        ValueError,
        match="at least one sample",
    ):
        generate_patient_states(
            _pipeline(),
            inputs,
        )


def test_state_generation_does_not_create_timestamps() -> None:
    result = generate_patient_states(
        _pipeline(),
        _inputs(),
    )

    assert not hasattr(
        result.states[0],
        "timestamp",
    )