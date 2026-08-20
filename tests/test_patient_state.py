"""
T1D-UOM — Unified Patient State contract tests.
"""

from __future__ import annotations

import pytest
import torch

from src.models.patient_state import (
    UnifiedPatientState,
    create_unified_patient_state,
    validate_unified_patient_state_contract,
)


def test_unified_patient_state_basic_contract() -> None:
    state = torch.randn(4, 64)

    patient_state = create_unified_patient_state(state)

    assert patient_state.batch_size == 4
    assert patient_state.state_dim == 64
    assert patient_state.state.shape == (4, 64)


def test_unified_patient_state_validation() -> None:
    state = torch.randn(3, 64)

    patient_state = UnifiedPatientState(state)

    validate_unified_patient_state_contract(
        patient_state,
        expected_batch_size=3,
        expected_state_dim=64,
    )


def test_invalid_rank_is_rejected() -> None:
    state = torch.randn(64)

    with pytest.raises(ValueError, match="shape"):
        UnifiedPatientState(state)


def test_integer_state_is_rejected() -> None:
    state = torch.ones(
        2,
        64,
        dtype=torch.long,
    )

    with pytest.raises(TypeError, match="floating-point"):
        UnifiedPatientState(state)


def test_non_finite_state_is_rejected() -> None:
    state = torch.randn(2, 64)
    state[0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN or infinite"):
        UnifiedPatientState(state)


def test_batch_size_contract() -> None:
    patient_state = UnifiedPatientState(
        torch.randn(5, 64)
    )

    with pytest.raises(
        ValueError,
        match="batch-size mismatch",
    ):
        validate_unified_patient_state_contract(
            patient_state,
            expected_batch_size=4,
        )


def test_state_dimension_contract() -> None:
    patient_state = UnifiedPatientState(
        torch.randn(5, 64)
    )

    with pytest.raises(
        ValueError,
        match="dimension mismatch",
    ):
        validate_unified_patient_state_contract(
            patient_state,
            expected_state_dim=32,
        )


def test_detached_state_preserves_contract() -> None:
    state = torch.randn(
        3,
        64,
        requires_grad=True,
    )

    patient_state = UnifiedPatientState(state)
    detached = patient_state.detached()

    assert detached.state.shape == (3, 64)
    assert detached.state.requires_grad is False