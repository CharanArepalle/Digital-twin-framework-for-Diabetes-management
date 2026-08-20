from __future__ import annotations

import pytest
import torch

from src.twin.dynamics import TwinDynamics
from src.twin.train_dynamics import DynamicsTrainingConfig
from src.twin.train_real_dynamics import (
    DynamicsSplit,
    split_by_participant,
    train_real_dynamics,
)
from src.twin.transitions import (
    TwinTransition,
    TwinTransitionDataset,
)


def _dataset() -> TwinTransitionDataset:
    transitions = []

    participants = (
        "UoM2301",
        "UoM2302",
        "UoM2303",
        "UoM2304",
    )

    for participant_index, participant in enumerate(
        participants
    ):
        for step in range(3):
            current = torch.full(
                (4,),
                float(participant_index + step),
            )

            next_state = current + 0.1

            transitions.append(
                TwinTransition(
                    current_state=current,
                    next_state=next_state,
                    delta_t=60.0,
                    participant_id=participant,
                )
            )

    return TwinTransitionDataset(
        transitions
    )


def test_split_is_participant_disjoint() -> None:
    split = split_by_participant(
        _dataset(),
        validation_participants=[
            "UoM2303",
            "UoM2304",
        ],
    )

    assert isinstance(
        split,
        DynamicsSplit,
    )

    train_ids = {
        transition.participant_id
        for transition in split.train.transitions
    }

    validation_ids = {
        transition.participant_id
        for transition in split.validation.transitions
    }

    assert train_ids == {
        "UoM2301",
        "UoM2302",
    }

    assert validation_ids == {
        "UoM2303",
        "UoM2304",
    }

    assert train_ids.isdisjoint(
        validation_ids
    )


def test_split_preserves_transition_state_dimension() -> None:
    split = split_by_participant(
        _dataset(),
        validation_participants=[
            "UoM2304",
        ],
    )

    assert split.train.state_dim == 4
    assert split.validation.state_dim == 4


def test_unknown_validation_participant_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="not present",
    ):
        split_by_participant(
            _dataset(),
            validation_participants=[
                "UNKNOWN",
            ],
        )


def test_empty_validation_participants_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        split_by_participant(
            _dataset(),
            validation_participants=[],
        )


def test_split_cannot_consume_entire_dataset() -> None:
    with pytest.raises(
        ValueError,
        match="empty training",
    ):
        split_by_participant(
            _dataset(),
            validation_participants=[
                "UoM2301",
                "UoM2302",
                "UoM2303",
                "UoM2304",
            ],
        )


def test_real_training_uses_existing_dynamics() -> None:
    model = TwinDynamics(
        state_dim=4,
        hidden_dim=8,
    )

    result = train_real_dynamics(
        dataset=_dataset(),
        validation_participants=[
            "UoM2304",
        ],
        model=model,
        config=DynamicsTrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=1e-2,
            seed=42,
        ),
    )

    assert result.state_dim == 4
    assert result.train_samples == 9
    assert result.validation_samples == 3

    assert result.best_epoch >= 1
    assert result.best_epoch <= 2

    assert torch.isfinite(
        torch.tensor(result.best_validation_loss)
    )


def test_real_training_rejects_model_dimension_mismatch() -> None:
    model = TwinDynamics(
        state_dim=5,
        hidden_dim=8,
    )

    with pytest.raises(
        ValueError,
        match="state dimension",
    ):
        train_real_dynamics(
            dataset=_dataset(),
            validation_participants=[
                "UoM2304",
            ],
            model=model,
            config=DynamicsTrainingConfig(
                epochs=1,
            ),
        )