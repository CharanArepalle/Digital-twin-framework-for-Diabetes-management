"""
T1D-UOM real TwinDynamics training/evaluation boundary.

Locked architecture:

Unified Patient State
        |
        v
   Digital Twin
    state layer
        |
        v
   TwinDynamics
 transition model
        |
        v
  Simulated State
      /       \
 Prediction   What-if

This module trains the existing TwinDynamics on an already-created
TwinTransitionDataset.

It does not:
    - modify raw data;
    - create features;
    - construct Unified Patient States;
    - create transitions;
    - change TwinDynamics architecture;
    - implement Prediction;
    - implement What-if;
    - implement UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .dynamics import TwinDynamics
from .train_dynamics import (
    DynamicsTrainingConfig,
    DynamicsTrainingResult,
    train_dynamics,
)
from .transitions import TwinTransitionDataset


__all__ = [
    "DynamicsSplit",
    "split_by_participant",
    "train_real_dynamics",
]


@dataclass(frozen=True)
class DynamicsSplit:
    """
    Participant-disjoint transition split.

    Participant IDs are kept disjoint between train and validation
    to prevent participant leakage.
    """

    train: TwinTransitionDataset
    validation: TwinTransitionDataset


def _participant_ids(
    dataset: TwinTransitionDataset,
) -> tuple[str, ...]:
    return tuple(
        transition.participant_id
        for transition in dataset.transitions
    )


def split_by_participant(
    dataset: TwinTransitionDataset,
    *,
    validation_participants: Sequence[str],
) -> DynamicsSplit:
    """
    Split transitions by participant.

    No transition is divided internally.

    The validation participants must not occur in the training set.
    """

    if not isinstance(
        dataset,
        TwinTransitionDataset,
    ):
        raise TypeError(
            "dataset must be a TwinTransitionDataset."
        )

    if len(dataset) == 0:
        raise ValueError(
            "dataset must contain at least one transition."
        )

    validation_set = {
        participant
        for participant in validation_participants
    }

    if not validation_set:
        raise ValueError(
            "validation_participants must contain at least one "
            "participant."
        )

    available = set(
        _participant_ids(dataset)
    )

    unknown = validation_set - available

    if unknown:
        raise ValueError(
            "Validation participants are not present in the dataset: "
            f"{sorted(unknown)}."
        )

    train_transitions = []
    validation_transitions = []

    for transition in dataset.transitions:
        if transition.participant_id in validation_set:
            validation_transitions.append(
                transition
            )
        else:
            train_transitions.append(
                transition
            )

    if not train_transitions:
        raise ValueError(
            "Participant split produced an empty training dataset."
        )

    if not validation_transitions:
        raise ValueError(
            "Participant split produced an empty validation dataset."
        )

    return DynamicsSplit(
        train=TwinTransitionDataset(
            train_transitions
        ),
        validation=TwinTransitionDataset(
            validation_transitions
        ),
    )


def train_real_dynamics(
    *,
    dataset: TwinTransitionDataset,
    validation_participants: Sequence[str],
    model: TwinDynamics,
    config: DynamicsTrainingConfig | None = None,
) -> DynamicsTrainingResult:
    """
    Train TwinDynamics using a participant-disjoint validation split.

    The caller must provide an already validated TwinTransitionDataset.
    """

    if not isinstance(
        model,
        TwinDynamics,
    ):
        raise TypeError(
            "model must be a TwinDynamics instance."
        )

    split = split_by_participant(
        dataset,
        validation_participants=validation_participants,
    )

    if split.train.state_dim != model.state_dim:
        raise ValueError(
            "Training state dimension does not match TwinDynamics."
        )

    if split.validation.state_dim != model.state_dim:
        raise ValueError(
            "Validation state dimension does not match TwinDynamics."
        )

    return train_dynamics(
        model=model,
        train_dataset=split.train,
        validation_dataset=split.validation,
        config=config,
    )