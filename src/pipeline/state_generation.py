"""
T1D-UOM Unified Patient State Generation.

Locked architecture:

    Five modality inputs
            |
            v
        Five GRUs
            |
            v
       zG zI zN zA zS
            |
            v
        MLP Fusion
            |
            v
    Unified Patient State
            |
            v
       Digital Twin
            |
            v
       TwinDynamics
            |
            v
     Simulated State
        /       \
 Prediction    What-if

This module is an integration boundary only.

It does not:
    - read or modify raw CSV files;
    - resample;
    - interpolate;
    - impute;
    - normalize;
    - create features;
    - create timestamps;
    - create transitions;
    - train TwinDynamics;
    - implement Prediction;
    - implement What-if;
    - implement UI.

The five modality branches may have independent padded sequence
lengths. Their valid lengths are passed through to FiveGRU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from src.models.five_gru_pipeline import (
    FiveGRUInputBatch,
    FiveGRUStatePipeline,
)
from src.models.patient_state import UnifiedPatientState


__all__ = [
    "GeneratedPatientState",
    "StateGenerationResult",
    "generate_patient_states",
]


@dataclass(frozen=True)
class GeneratedPatientState:
    """One generated Unified Patient State."""

    index: int
    state: Tensor


@dataclass(frozen=True)
class StateGenerationResult:
    """Immutable result of one state-generation operation."""

    states: tuple[GeneratedPatientState, ...]
    state_dim: int
    batch_size: int

    @property
    def count(self) -> int:
        return len(self.states)

    def stacked(self) -> Tensor:
        """Return generated states as [batch, state_dim]."""

        if not self.states:
            raise RuntimeError(
                "Cannot stack an empty StateGenerationResult."
            )

        return torch.stack(
            tuple(item.state for item in self.states),
            dim=0,
        )


def _validate_input_batch(
    inputs: FiveGRUInputBatch,
) -> int:
    """Validate batch-level invariants."""

    if not isinstance(
        inputs,
        FiveGRUInputBatch,
    ):
        raise TypeError(
            "inputs must be a FiveGRUInputBatch."
        )

    batch_sizes = {
        inputs.glucose.shape[0],
        inputs.insulin.shape[0],
        inputs.nutrition.shape[0],
        inputs.activity.shape[0],
        inputs.sleep.shape[0],
    }

    if len(batch_sizes) != 1:
        raise ValueError(
            "All five modality inputs must have the same "
            "batch size."
        )

    batch_size = inputs.glucose.shape[0]

    if batch_size <= 0:
        raise ValueError(
            "Input batch must contain at least one sample."
        )

    return batch_size


def _validate_lengths(
    lengths: Mapping[str, Tensor] | None,
    *,
    inputs: FiveGRUInputBatch,
    batch_size: int,
) -> None:
    """
    Validate optional independent modality lengths.

    The authoritative sequence handling remains inside
    FiveGRUStatePipeline/FiveGRU. This function only checks
    the state-generation boundary.
    """

    if lengths is None:
        return

    expected_names = {
        "glucose",
        "insulin",
        "nutrition",
        "activity",
        "sleep",
    }

    unexpected = set(lengths.keys()) - expected_names

    if unexpected:
        raise ValueError(
            "Unexpected sequence-length keys: "
            f"{sorted(unexpected)}."
        )

    tensors = {
        "glucose": inputs.glucose,
        "insulin": inputs.insulin,
        "nutrition": inputs.nutrition,
        "activity": inputs.activity,
        "sleep": inputs.sleep,
    }

    for name, value in lengths.items():
        if not isinstance(
            value,
            Tensor,
        ):
            raise TypeError(
                f"lengths['{name}'] must be a torch.Tensor."
            )

        if value.ndim != 1:
            raise ValueError(
                f"lengths['{name}'] must have shape [batch]."
            )

        if value.shape[0] != batch_size:
            raise ValueError(
                f"lengths['{name}'] batch dimension does "
                "not match the inputs."
            )

        if not torch.is_floating_point(value) and (
            value.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            )
        ):
            raise TypeError(
                f"lengths['{name}'] must contain integer "
                "sequence lengths."
            )

        value_long = value.to(
            dtype=torch.long,
            device=tensors[name].device,
        )

        if torch.any(value_long <= 0):
            raise ValueError(
                f"lengths['{name}'] must contain only "
                "positive values."
            )

        if torch.any(
            value_long > tensors[name].shape[1]
        ):
            raise ValueError(
                f"lengths['{name}'] contains a value "
                "greater than the supplied padded sequence length."
            )


def _validate_generated_state(
    state: UnifiedPatientState,
    *,
    expected_batch_size: int,
    expected_state_dim: int,
) -> None:
    """Validate the state returned by the existing model path."""

    if not isinstance(
        state,
        UnifiedPatientState,
    ):
        raise TypeError(
            "FiveGRUStatePipeline must return "
            "UnifiedPatientState."
        )

    value = state.state

    if not isinstance(
        value,
        Tensor,
    ):
        raise TypeError(
            "UnifiedPatientState.state must be a torch.Tensor."
        )

    if value.ndim != 2:
        raise ValueError(
            "UnifiedPatientState.state must have shape "
            "[batch, state_dim]. "
            f"Received {tuple(value.shape)}."
        )

    if value.shape[0] != expected_batch_size:
        raise ValueError(
            "UnifiedPatientState batch size does not match "
            "the input batch size."
        )

    if value.shape[1] != expected_state_dim:
        raise ValueError(
            "UnifiedPatientState state dimension does not "
            "match the configured pipeline state dimension."
        )

    if not torch.is_floating_point(value):
        raise TypeError(
            "UnifiedPatientState.state must be floating point."
        )

    if not torch.isfinite(value).all():
        raise ValueError(
            "UnifiedPatientState.state contains non-finite values."
        )


def generate_patient_states(
    pipeline: FiveGRUStatePipeline,
    inputs: FiveGRUInputBatch,
) -> StateGenerationResult:
    """
    Generate Unified Patient States from model-ready modality inputs.

    The five modality sequences may have different padded lengths.

    No timestamp or temporal ordering is created here.
    """

    if not isinstance(
        pipeline,
        FiveGRUStatePipeline,
    ):
        raise TypeError(
            "pipeline must be a FiveGRUStatePipeline."
        )

    batch_size = _validate_input_batch(
        inputs
    )

    _validate_lengths(
        inputs.lengths,
        inputs=inputs,
        batch_size=batch_size,
    )

    with torch.no_grad():
        patient_state = pipeline(
            inputs
        )

    _validate_generated_state(
        patient_state,
        expected_batch_size=batch_size,
        expected_state_dim=pipeline.state_dim,
    )

    state_tensor = patient_state.state.detach()

    states = tuple(
        GeneratedPatientState(
            index=index,
            state=state_tensor[index].clone(),
        )
        for index in range(batch_size)
    )

    return StateGenerationResult(
        states=states,
        state_dim=pipeline.state_dim,
        batch_size=batch_size,
    )