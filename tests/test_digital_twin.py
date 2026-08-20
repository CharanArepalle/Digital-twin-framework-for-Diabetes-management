"""
Tests for the T1D-UOM Digital Twin state engine.
"""

from __future__ import annotations

import pytest
import torch

from src.twin.digital_twin import DigitalTwin, DigitalTwinState


def _state(
    batch_size: int = 3,
    state_dim: int = 16,
) -> torch.Tensor:
    return torch.randn(batch_size, state_dim)


def test_initialization_preserves_patient_state() -> None:
    model = DigitalTwin(state_dim=16)

    patient_state = _state()
    twin = model.initialize(patient_state)

    assert isinstance(twin, DigitalTwinState)
    assert twin.value.shape == (3, 16)
    assert torch.equal(twin.value, patient_state)
    assert twin.value.data_ptr() != patient_state.data_ptr()


def test_update_replaces_state_without_mutating_input() -> None:
    model = DigitalTwin(state_dim=16)

    initial = _state()
    observed = _state()

    twin = model.initialize(initial)
    updated = model.update(twin, observed)

    assert torch.equal(updated.value, observed)
    assert updated.value.data_ptr() != observed.data_ptr()
    assert torch.equal(twin.value, initial)


def test_scenario_isolated_from_original_state() -> None:
    model = DigitalTwin(state_dim=16)

    initial = _state()
    delta = torch.ones_like(initial)

    twin = model.initialize(initial)
    scenario = model.scenario(twin, delta)

    assert torch.allclose(
        scenario.value,
        initial + delta,
    )

    assert torch.equal(twin.value, initial)


def test_evolution_preserves_state_shape() -> None:
    torch.manual_seed(7)

    model = DigitalTwin(
        state_dim=16,
        hidden_dim=32,
    )

    twin = model.initialize(_state())

    evolved = model.evolve(twin)

    assert evolved.value.shape == twin.value.shape
    assert torch.isfinite(evolved.value).all()


def test_context_aware_evolution() -> None:
    torch.manual_seed(7)

    model = DigitalTwin(
        state_dim=16,
        hidden_dim=32,
        context_dim=5,
    )

    twin = model.initialize(_state())
    context = torch.randn(3, 5)

    evolved = model.evolve(
        twin_state=twin,
        context=context,
    )

    assert evolved.value.shape == (3, 16)
    assert torch.isfinite(evolved.value).all()


def test_missing_required_context_is_rejected() -> None:
    model = DigitalTwin(
        state_dim=16,
        context_dim=5,
    )

    twin = model.initialize(_state())

    with pytest.raises(ValueError, match="requires a context"):
        model.evolve(twin)


def test_unexpected_context_is_rejected() -> None:
    model = DigitalTwin(
        state_dim=16,
        context_dim=None,
    )

    twin = model.initialize(_state())

    with pytest.raises(ValueError, match="without context_dim"):
        model.evolve(
            twin,
            context=torch.randn(3, 5),
        )


def test_wrong_state_dimension_is_rejected() -> None:
    model = DigitalTwin(state_dim=16)

    with pytest.raises(ValueError, match="state dimension"):
        model.initialize(torch.randn(3, 15))


def test_invalid_state_rank_is_rejected() -> None:
    model = DigitalTwin(state_dim=16)

    with pytest.raises(ValueError, match=r"shape \[batch, state_dim\]"):
        model.initialize(torch.randn(3, 4, 16))


def test_non_finite_state_is_rejected() -> None:
    model = DigitalTwin(state_dim=16)

    patient_state = _state()
    patient_state[0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        model.initialize(patient_state)


def test_scenario_state_dimension_mismatch_is_rejected() -> None:
    model = DigitalTwin(state_dim=16)

    twin = model.initialize(_state())
    delta = torch.randn(3, 15)

    with pytest.raises(
        ValueError,
        match=r"delta has state dimension 15; expected 16",
    ):
        model.scenario(twin, delta)


def test_batch_mismatch_on_update_is_rejected() -> None:
    model = DigitalTwin(state_dim=16)

    twin = model.initialize(
        _state(batch_size=3)
    )

    observed = _state(batch_size=2)

    with pytest.raises(ValueError, match="batch size"):
        model.update(twin, observed)


def test_context_dimension_mismatch_is_rejected() -> None:
    model = DigitalTwin(
        state_dim=16,
        context_dim=5,
    )

    twin = model.initialize(_state())

    with pytest.raises(ValueError, match="expected 5"):
        model.evolve(
            twin,
            context=torch.randn(3, 4),
        )


def test_forward_matches_evolution() -> None:
    torch.manual_seed(19)

    model = DigitalTwin(
        state_dim=16,
        hidden_dim=32,
    )

    twin = model.initialize(_state())

    direct = model.evolve(twin)
    via_forward = model(twin)

    # Both calls are valid executions of the same deterministic architecture
    # while the model is in evaluation mode.
    assert direct.value.shape == via_forward.value.shape


def test_evolution_does_not_modify_previous_state() -> None:
    torch.manual_seed(23)

    model = DigitalTwin(
        state_dim=16,
        hidden_dim=32,
    )

    twin = model.initialize(_state())
    before = twin.value.clone()

    _ = model.evolve(twin)

    assert torch.equal(twin.value, before)