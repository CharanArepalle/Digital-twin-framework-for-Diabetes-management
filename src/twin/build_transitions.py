"""
T1D-UOM Unified Patient State -> TwinTransitionDataset bridge.

Locked architecture
-------------------

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
        v         v
   Prediction   What-if

This module is ONLY the project-level bridge between generated
Unified Patient States and the existing authoritative
build_transition_dataset() contract.

It does NOT:

    - read raw CSV files;
    - modify datasets;
    - sort observations;
    - resample;
    - interpolate;
    - impute;
    - normalize;
    - engineer features;
    - construct timestamps;
    - parse raw timestamp strings;
    - train TwinDynamics;
    - implement Prediction;
    - implement What-if;
    - implement the UI.

The authoritative temporal transition logic remains in:

    src.twin.transitions.build_transition_dataset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .transitions import (
    TwinTransitionDataset,
    build_transition_dataset,
)


__all__ = [
    "StateRecord",
    "build_transitions",
]


@dataclass(frozen=True)
class StateRecord:
    """
    One already-generated Unified Patient State observation.

    Parameters
    ----------
    state:
        Unified Patient State vector.

        Shape:
            [state_dim]

    timestamp:
        Already-resolved timestamp object.

        Accepted by the authoritative transition layer:
            - numeric seconds;
            - datetime-like objects exposing timestamp().

        Raw timestamp strings are intentionally not accepted.

    participant_id:
        Frozen participant identifier.
    """

    state: Tensor
    timestamp: object
    participant_id: str


def _validate_record(
    record: StateRecord,
    index: int,
) -> None:
    """
    Validate one state record before delegation.
    """

    if not isinstance(
        record,
        StateRecord,
    ):
        raise TypeError(
            f"records[{index}] must be a StateRecord."
        )

    state = record.state

    if not isinstance(
        state,
        Tensor,
    ):
        raise TypeError(
            f"records[{index}].state must be a torch.Tensor."
        )

    if state.ndim != 1:
        raise ValueError(
            f"records[{index}].state must have shape "
            "[state_dim]; received {tuple(state.shape)}."
        )

    if state.shape[0] <= 0:
        raise ValueError(
            f"records[{index}].state must have a positive "
            "state dimension."
        )

    if not torch.is_floating_point(state):
        raise TypeError(
            f"records[{index}].state must use a floating-point dtype."
        )

    if not torch.isfinite(state).all():
        raise ValueError(
            f"records[{index}].state must contain only finite values."
        )

    if not isinstance(
        record.participant_id,
        str,
    ):
        raise TypeError(
            f"records[{index}].participant_id must be a string."
        )

    if not record.participant_id.strip():
        raise ValueError(
            f"records[{index}].participant_id must not be empty."
        )


def build_transitions(
    records: Sequence[StateRecord],
) -> TwinTransitionDataset:
    """
    Build the project's TwinTransitionDataset from state records.

    IMPORTANT
    ---------
    Records must already be in the intended temporal order.

    This function deliberately does NOT sort them.

    The authoritative transition rules are delegated to
    build_transition_dataset().
    """

    if not isinstance(
        records,
        Sequence,
    ):
        raise TypeError(
            "records must be a sequence of StateRecord objects."
        )

    for index, record in enumerate(records):
        _validate_record(
            record,
            index,
        )

    if not records:
        return TwinTransitionDataset(())

    state_dim = records[0].state.shape[0]

    for index, record in enumerate(records):
        if record.state.shape[0] != state_dim:
            raise ValueError(
                "All state records must have the same state dimension. "
                f"records[{index}] has dimension "
                f"{record.state.shape[0]}, expected {state_dim}."
            )

    states = torch.stack(
        tuple(
            record.state.clone()
            for record in records
        ),
        dim=0,
    )

    timestamps = tuple(
        record.timestamp
        for record in records
    )

    participant_ids = tuple(
        record.participant_id
        for record in records
    )

    return build_transition_dataset(
        states=states,
        timestamps=timestamps,
        participant_ids=participant_ids,
    )