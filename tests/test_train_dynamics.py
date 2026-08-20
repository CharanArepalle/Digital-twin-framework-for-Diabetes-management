"""
Tests for the T1D-UOM TwinDynamics training stage.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.twin.dynamics import TwinDynamics
from src.twin.train_dynamics import (
    DynamicsTrainingConfig,
    DynamicsTrainingResult,
    train_dynamics,
)
from src.twin.transitions import (
    TwinTransition,
    TwinTransitionDataset,
)


def _transition_dataset(
    *,
    start: float,
    count: int,
    state_dim: int = 4,
) -> TwinTransitionDataset:
    transitions = []

    for index in range(count):
        current_value = start + float(index)
        next_value = current_value + 0.5

        current = torch.full(
            (state_dim,),
            current_value,
            dtype=torch.float32,
        )

        next_state = torch.full(
            (state_dim,),
            next_value,
            dtype=torch.float32,
        )

        transitions.append(
            TwinTransition(
                current_state=current,
                next_state=next_state,
                delta_t=60.0,
                participant_id="UoM2301",
            )
        )

    return TwinTransitionDataset(
        transitions
    )


def test_training_config_defaults_are_valid() -> None:
    config = DynamicsTrainingConfig()

    assert config.epochs == 50
    assert config.batch_size == 32
    assert config.learning_rate == pytest.approx(1e-3)
    assert config.weight_decay == pytest.approx(0.0)
    assert config.seed == 42
    assert config.device == "cpu"
    assert config.num_workers == 0
    assert config.checkpoint_path is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("epochs", 0),
        ("batch_size", 0),
        ("learning_rate", 0.0),
        ("weight_decay", -1.0),
        ("num_workers", -1),
    ],
)
def test_training_config_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    kwargs = {
        field: value,
    }

    with pytest.raises(
        (TypeError, ValueError),
    ):
        DynamicsTrainingConfig(
            **kwargs,
        )


def test_training_requires_twin_dynamics() -> None:
    train_dataset = _transition_dataset(
        start=1.0,
        count=4,
    )

    validation_dataset = _transition_dataset(
        start=10.0,
        count=2,
    )

    with pytest.raises(
        TypeError,
        match="TwinDynamics",
    ):
        train_dynamics(
            model=torch.nn.Linear(4, 4),  # type: ignore[arg-type]
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            config=DynamicsTrainingConfig(
                epochs=1,
            ),
        )


def test_training_rejects_empty_training_dataset() -> None:
    model = TwinDynamics(
        state_dim=4,
        hidden_dim=8,
    )

    empty_dataset = TwinTransitionDataset(
        ()
    )

    validation_dataset = _transition_dataset(
        start=10.0,
        count=2,
    )

    with pytest.raises(
        ValueError,
        match="train_dataset",
    ):
        train_dynamics(
            model=model,
            train_dataset=empty_dataset,
            validation_dataset=validation_dataset,
            config=DynamicsTrainingConfig(
                epochs=1,
            ),
        )


def test_training_rejects_empty_validation_dataset() -> None:
    model = TwinDynamics(
        state_dim=4,
        hidden_dim=8,
    )

    train_dataset = _transition_dataset(
        start=1.0,
        count=4,
    )

    empty_dataset = TwinTransitionDataset(
        ()
    )

    with pytest.raises(
        ValueError,
        match="validation_dataset",
    ):
        train_dynamics(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=empty_dataset,
            config=DynamicsTrainingConfig(
                epochs=1,
            ),
        )


def test_training_rejects_state_dimension_mismatch() -> None:
    model = TwinDynamics(
        state_dim=4,
        hidden_dim=8,
    )

    train_dataset = _transition_dataset(
        start=1.0,
        count=4,
        state_dim=5,
    )

    validation_dataset = _transition_dataset(
        start=10.0,
        count=2,
        state_dim=4,
    )

    with pytest.raises(
        ValueError,
        match="state dimension",
    ):
        train_dynamics(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            config=DynamicsTrainingConfig(
                epochs=1,
            ),
        )


def test_training_rejects_context_dimension() -> None:
    model = TwinDynamics(
        state_dim=4,
        hidden_dim=8,
        context_dim=2,
    )

    train_dataset = _transition_dataset(
        start=1.0,
        count=4,
    )

    validation_dataset = _transition_dataset(
        start=10.0,
        count=2,
    )

    with pytest.raises(
        ValueError,
        match="context_dim",
    ):
        train_dynamics(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            config=DynamicsTrainingConfig(
                epochs=1,
            ),
        )


def test_training_returns_complete_result() -> None:
    torch.manual_seed(7)

    model = TwinDynamics(
        state_dim=4,
        hidden_dim=16,
    )

    train_dataset = _transition_dataset(
        start=1.0,
        count=12,
    )

    validation_dataset = _transition_dataset(
        start=20.0,
        count=4,
    )

    result = train_dynamics(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        config=DynamicsTrainingConfig(
            epochs=3,
            batch_size=4,
            learning_rate=1e-2,
            seed=42,
        ),
    )

    assert isinstance(
        result,
        DynamicsTrainingResult,
    )

    assert result.best_epoch >= 1
    assert result.best_epoch <= 3

    assert result.train_samples == 12
    assert result.validation_samples == 4
    assert result.state_dim == 4

    assert torch.isfinite(
        torch.tensor(result.best_train_loss)
    )

    assert torch.isfinite(
        torch.tensor(result.best_validation_loss)
    )

    assert torch.isfinite(
        torch.tensor(result.final_train_loss)
    )

    assert torch.isfinite(
        torch.tensor(result.final_validation_loss)
    )

    assert len(result.history) == 3


def test_training_creates_best_checkpoint(
    tmp_path: Path,
) -> None:
    model = TwinDynamics(
        state_dim=4,
        hidden_dim=8,
    )

    train_dataset = _transition_dataset(
        start=1.0,
        count=8,
    )

    validation_dataset = _transition_dataset(
        start=10.0,
        count=4,
    )

    checkpoint = (
        tmp_path
        / "twin_dynamics.pt"
    )

    result = train_dynamics(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        config=DynamicsTrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=1e-2,
            seed=42,
            checkpoint_path=str(checkpoint),
        ),
    )

    assert result.checkpoint_path == str(
        checkpoint
    )

    assert checkpoint.exists()
    assert checkpoint.is_file()

    saved = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    assert saved["format"] == (
        "t1d_uom_twin_dynamics_checkpoint_v1"
    )

    assert saved["architecture"]["component"] == (
        "TwinDynamics"
    )

    assert saved["architecture"]["state_dim"] == 4
    assert saved["architecture"]["hidden_dim"] == 8
    assert saved["architecture"]["context_dim"] is None

    assert "model_state_dict" in saved
    assert "optimizer_state_dict" in saved
    assert "training" in saved


def test_training_does_not_modify_transition_dataset() -> None:
    model = TwinDynamics(
        state_dim=4,
        hidden_dim=8,
    )

    train_dataset = _transition_dataset(
        start=1.0,
        count=6,
    )

    validation_dataset = _transition_dataset(
        start=10.0,
        count=3,
    )

    before_current = (
        train_dataset[0]["current_state"]
    )

    before_next = (
        train_dataset[0]["next_state"]
    )

    train_dynamics(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        config=DynamicsTrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-2,
            seed=42,
        ),
    )

    after_current = (
        train_dataset[0]["current_state"]
    )

    after_next = (
        train_dataset[0]["next_state"]
    )

    assert torch.equal(
        before_current,
        after_current,
    )

    assert torch.equal(
        before_next,
        after_next,
    )


def test_training_produces_finite_model_output() -> None:
    model = TwinDynamics(
        state_dim=4,
        hidden_dim=8,
    )

    train_dataset = _transition_dataset(
        start=1.0,
        count=8,
    )

    validation_dataset = _transition_dataset(
        start=10.0,
        count=4,
    )

    train_dynamics(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        config=DynamicsTrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=1e-2,
            seed=42,
        ),
    )

    model.eval()

    current_state = (
        train_dataset[0]["current_state"]
        .unsqueeze(0)
    )

    with torch.no_grad():
        prediction = model(
            current_state
        )

    assert prediction.shape == (
        1,
        4,
    )

    assert torch.isfinite(
        prediction
    ).all()