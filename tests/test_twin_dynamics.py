"""
Tests for T1D-UOM Digital Twin Dynamics.
"""

from __future__ import annotations

import pytest
import torch

from src.twin.dynamics import TwinDynamics


def _state(
    batch_size: int = 4,
    state_dim: int = 16,
) -> torch.Tensor:
    return torch.randn(batch_size, state_dim)


def test_forward_preserves_state_shape() -> None:
    model = TwinDynamics(
        state_dim=16,
        hidden_dim=32,
    )

    state = _state()

    output = model(state)

    assert output.shape == state.shape
    assert torch.isfinite(output).all()


def test_forward_with_context() -> None:
    model = TwinDynamics(
        state_dim=16,
        hidden_dim=32,
        context_dim=5,
    )

    state = _state()
    context = torch.randn(4, 5)

    output = model(
        state,
        context=context,
    )

    assert output.shape == state.shape
    assert torch.isfinite(output).all()


def test_transition_is_residual() -> None:
    torch.manual_seed(11)

    model = TwinDynamics(
        state_dim=16,
        hidden_dim=32,
    )

    state = _state()

    with torch.no_grad():
        output = model(state)
        delta = output - state

    assert delta.shape == state.shape
    assert torch.isfinite(delta).all()


def test_loss_is_scalar_and_finite() -> None:
    model = TwinDynamics(
        state_dim=16,
        hidden_dim=32,
    )

    current = _state()
    next_state = _state()

    loss = model.loss(
        current_state=current,
        next_state=next_state,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_loss_supports_context() -> None:
    model = TwinDynamics(
        state_dim=16,
        hidden_dim=32,
        context_dim=5,
    )

    current = _state()
    next_state = _state()
    context = torch.randn(4, 5)

    loss = model.loss(
        current_state=current,
        next_state=next_state,
        context=context,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_wrong_state_dimension_is_rejected() -> None:
    model = TwinDynamics(
        state_dim=16,
    )

    with pytest.raises(ValueError, match="state dimension"):
        model(torch.randn(4, 15))


def test_wrong_context_dimension_is_rejected() -> None:
    model = TwinDynamics(
        state_dim=16,
        context_dim=5,
    )

    with pytest.raises(ValueError, match="expected 5"):
        model(
            _state(),
            context=torch.randn(4, 4),
        )


def test_missing_required_context_is_rejected() -> None:
    model = TwinDynamics(
        state_dim=16,
        context_dim=5,
    )

    with pytest.raises(ValueError, match="requires a context"):
        model(_state())


def test_unexpected_context_is_rejected() -> None:
    model = TwinDynamics(
        state_dim=16,
        context_dim=None,
    )

    with pytest.raises(ValueError, match="without context_dim"):
        model(
            _state(),
            context=torch.randn(4, 5),
        )


def test_batch_mismatch_is_rejected() -> None:
    model = TwinDynamics(
        state_dim=16,
        context_dim=5,
    )

    with pytest.raises(ValueError, match="batch size"):
        model(
            _state(batch_size=4),
            context=torch.randn(3, 5),
        )


def test_loss_rejects_mismatched_target_shape() -> None:
    model = TwinDynamics(
        state_dim=16,
    )

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        model.loss(
            current_state=_state(batch_size=4),
            next_state=_state(batch_size=3),
        )


def test_non_finite_state_is_rejected() -> None:
    model = TwinDynamics(
        state_dim=16,
    )

    state = _state()
    state[0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        model(state)


def test_non_finite_target_is_rejected() -> None:
    model = TwinDynamics(
        state_dim=16,
    )

    current = _state()
    target = _state()
    target[0, 0] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        model.loss(
            current_state=current,
            next_state=target,
        )


def test_parameters_receive_gradients() -> None:
    model = TwinDynamics(
        state_dim=16,
        hidden_dim=32,
    )

    current = _state()
    target = _state()

    loss = model.loss(
        current_state=current,
        next_state=target,
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert gradients
    assert all(
        gradient is not None
        for gradient in gradients
    )