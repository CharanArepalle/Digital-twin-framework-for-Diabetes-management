"""
T1D-UOM Digital Twin Temporal Transition Contract.

Architecture position:

    Five GRUs
        ↓
    MLP Fusion
        ↓
    Unified Patient State
        ↓
    Temporal Transition Pairs
        ↓
    TwinDynamics
        ↓
    Simulated State
        ↓
    Prediction / What-if

This module defines the temporal transition-data boundary consumed by
TwinDynamics.

It deliberately does NOT:

    - access raw dataset files
    - modify dataset files
    - perform resampling
    - perform interpolation
    - perform imputation
    - perform normalization
    - perform feature engineering
    - encode categorical values
    - create model windows
    - train a model
    - implement Prediction
    - implement What-if
    - implement the Interactive UI

A transition is strictly:

    current_state at time t
            +
    next_state at time t + Δt
            ↓
       one transition pair

Transitions are never constructed across participant boundaries.
Timestamps must be strictly increasing within each participant.

The state tensors are expected to already represent the Unified Patient
State produced by the upstream MLP Fusion stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset


__all__ = [
    "TwinTransition",
    "TwinTransitionDataset",
    "build_transition_dataset",
]


@dataclass(frozen=True)
class TwinTransition:
    """
    One validated Digital Twin state transition.

    Parameters
    ----------
    current_state:
        Unified Patient State at time t.
        Shape: [state_dim].

    next_state:
        Observed Unified Patient State at time t + Δt.
        Shape: [state_dim].

    delta_t:
        Optional elapsed time between the two observations.
        Shape: scalar tensor or Python float.

        This value is metadata by default. It is NOT automatically passed
        to TwinDynamics as context.

    participant_id:
        Optional participant identifier associated with the transition.

    current_time:
        Optional original timestamp representation for the current state.

    next_time:
        Optional original timestamp representation for the next state.
    """

    current_state: Tensor
    next_state: Tensor
    delta_t: Optional[float] = None
    participant_id: Optional[str] = None
    current_time: Optional[object] = None
    next_time: Optional[object] = None

    def __post_init__(self) -> None:
        _validate_state_vector(
            self.current_state,
            "current_state",
        )

        _validate_state_vector(
            self.next_state,
            "next_state",
        )

        if self.current_state.shape != self.next_state.shape:
            raise ValueError(
                "current_state and next_state must have exactly "
                "the same shape."
            )

        if self.delta_t is not None:
            if not isinstance(
                self.delta_t,
                (int, float),
            ):
                raise TypeError(
                    "delta_t must be None, int, or float."
                )

            if not torch.isfinite(
                torch.tensor(
                    float(self.delta_t),
                    dtype=torch.float64,
                )
            ):
                raise ValueError(
                    "delta_t must be finite."
                )

            if float(self.delta_t) <= 0.0:
                raise ValueError(
                    "delta_t must be strictly positive."
                )

        if self.participant_id is not None:
            if not isinstance(
                self.participant_id,
                str,
            ):
                raise TypeError(
                    "participant_id must be None or a string."
                )

            if not self.participant_id.strip():
                raise ValueError(
                    "participant_id must not be empty."
                )


@dataclass(frozen=True)
class _TransitionRecord:
    """
    Internal immutable record used while constructing transitions.
    """

    participant_id: str
    timestamp: object
    state: Tensor


class TwinTransitionDataset(Dataset[dict[str, Tensor]]):
    """
    Read-only PyTorch Dataset containing validated transition pairs.

    Each item contains:

        current_state
        next_state

    Optional metadata are exposed separately through:

        delta_t
        participant_id
        current_time
        next_time

    No transformation is performed by this Dataset.
    """

    def __init__(
        self,
        transitions: Sequence[TwinTransition],
    ) -> None:
        if not isinstance(
            transitions,
            Sequence,
        ):
            raise TypeError(
                "transitions must be a sequence of TwinTransition objects."
            )

        self._transitions = tuple(transitions)

        for index, transition in enumerate(
            self._transitions
        ):
            if not isinstance(
                transition,
                TwinTransition,
            ):
                raise TypeError(
                    "transitions["
                    f"{index}"
                    "] must be a TwinTransition."
                )

        if self._transitions:
            state_dim = (
                self._transitions[0]
                .current_state
                .shape[0]
            )

            for index, transition in enumerate(
                self._transitions
            ):
                if (
                    transition.current_state.shape[0]
                    != state_dim
                ):
                    raise ValueError(
                        "All transitions must have the same "
                        f"state dimension. Transition {index} "
                        f"has dimension "
                        f"{transition.current_state.shape[0]}, "
                        f"expected {state_dim}."
                    )

    def __len__(self) -> int:
        return len(self._transitions)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:
        transition = self._transitions[index]

        item: dict[str, Tensor] = {
            "current_state": transition.current_state.clone(),
            "next_state": transition.next_state.clone(),
        }

        if transition.delta_t is not None:
            item["delta_t"] = torch.tensor(
                transition.delta_t,
                dtype=transition.current_state.dtype,
            )

        return item

    @property
    def transitions(self) -> tuple[TwinTransition, ...]:
        """
        Return the immutable transition collection.

        The transition objects themselves are immutable.
        """
        return self._transitions

    @property
    def state_dim(self) -> Optional[int]:
        """Return the common state dimension."""
        if not self._transitions:
            return None

        return int(
            self._transitions[0]
            .current_state
            .shape[0]
        )


def _validate_state_vector(
    state: Tensor,
    name: str,
) -> None:
    """
    Validate one Unified Patient State vector.

    A transition pair contains one patient state, therefore the required
    shape here is [state_dim], not [batch, state_dim].
    """

    if not isinstance(
        state,
        Tensor,
    ):
        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    if state.ndim != 1:
        raise ValueError(
            f"{name} must have shape [state_dim]; "
            f"received {tuple(state.shape)}."
        )

    if state.shape[0] <= 0:
        raise ValueError(
            f"{name} must have a positive state dimension."
        )

    if not torch.is_floating_point(state):
        raise TypeError(
            f"{name} must use a floating-point dtype."
        )

    if not torch.isfinite(state).all():
        raise ValueError(
            f"{name} must contain only finite values."
        )


def _timestamp_to_seconds(
    timestamp: object,
) -> float:
    """
    Convert an already-resolved timestamp object to seconds.

    Supported values:

        - int
        - float
        - objects implementing total_seconds()

    This function does not parse raw CSV timestamp strings.
    """

    if isinstance(
        timestamp,
        (int, float),
    ):
        value = float(timestamp)

    elif hasattr(
        timestamp,
        "timestamp",
    ):
        value = float(
            timestamp.timestamp()
        )

    else:
        raise TypeError(
    "Timestamp values must be already-resolved objects "
    "or numeric seconds. Raw timestamp strings are not "
    "parsed by this module."
)

    if not torch.isfinite(
        torch.tensor(
            value,
            dtype=torch.float64,
        )
    ):
        raise ValueError(
            "Timestamp value must be finite."
        )

    return value


def build_transition_dataset(
    states: Tensor,
    timestamps: Sequence[object],
    participant_ids: Sequence[str],
) -> TwinTransitionDataset:
    """
    Build adjacent temporal transition pairs.

    Parameters
    ----------
    states:
        Unified Patient State matrix.

        Shape:
            [num_observations, state_dim]

        The rows must already be ordered according to the supplied
        timestamps within each participant.

    timestamps:
        One already-resolved timestamp for each state row.

        Raw timestamp strings are intentionally rejected.

    participant_ids:
        Participant identifier for each state row.

    Returns
    -------
    TwinTransitionDataset
        Validated adjacent transitions.

    Construction rule
    -----------------
    For rows i and i+1:

        if participant[i] == participant[i+1]
        and timestamp[i+1] > timestamp[i]:

            create:
                current_state = states[i]
                next_state    = states[i+1]

        otherwise:

            do not create a transition.

    No transition is ever created across participant boundaries.

    Important
    ---------
    This function does not sort the input. It does not resample it.
    It does not interpolate it. It does not impute it.

    The caller is responsible for supplying the observations in the
    intended temporal order.
    """

    if not isinstance(
        states,
        Tensor,
    ):
        raise TypeError(
            "states must be a torch.Tensor."
        )

    if states.ndim != 2:
        raise ValueError(
            "states must have shape "
            "[num_observations, state_dim]."
        )

    if states.shape[0] == 0:
        raise ValueError(
            "states must contain at least one observation."
        )

    if states.shape[1] <= 0:
        raise ValueError(
            "states must have a positive state dimension."
        )

    if not torch.is_floating_point(states):
        raise TypeError(
            "states must use a floating-point dtype."
        )

    if not torch.isfinite(states).all():
        raise ValueError(
            "states must contain only finite values."
        )

    if len(timestamps) != states.shape[0]:
        raise ValueError(
            "timestamps length must equal the number of state rows."
        )

    if len(participant_ids) != states.shape[0]:
        raise ValueError(
            "participant_ids length must equal the number of state rows."
        )

    if states.shape[0] < 2:
        return TwinTransitionDataset(())

    resolved_timestamps = [
        _timestamp_to_seconds(timestamp)
        for timestamp in timestamps
    ]

    normalized_participants: list[str] = []

    for index, participant_id in enumerate(
        participant_ids
    ):
        if not isinstance(
            participant_id,
            str,
        ):
            raise TypeError(
                "participant_ids["
                f"{index}"
                "] must be a string."
            )

        if not participant_id.strip():
            raise ValueError(
                "participant_ids["
                f"{index}"
                "] must not be empty."
            )

        normalized_participants.append(
            participant_id
        )

    transitions: list[TwinTransition] = []

    for index in range(
        states.shape[0] - 1
    ):
        current_participant = (
            normalized_participants[index]
        )
        next_participant = (
            normalized_participants[index + 1]
        )

        # Never cross participant boundaries.
        if (
            current_participant
            != next_participant
        ):
            continue

        current_time = resolved_timestamps[index]
        next_time = resolved_timestamps[index + 1]

        # Equal or backward timestamps do not define a valid forward
        # temporal transition.
        if next_time <= current_time:
            continue

        delta_t = next_time - current_time

        transitions.append(
            TwinTransition(
                current_state=states[index].clone(),
                next_state=states[index + 1].clone(),
                delta_t=delta_t,
                participant_id=current_participant,
                current_time=timestamps[index],
                next_time=timestamps[index + 1],
            )
        )

    return TwinTransitionDataset(
        transitions
    )