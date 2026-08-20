"""
T1D-UOM — Unified Patient State.

Architecture position:

    zG ─┐
    zI ─┤
    zN ─┼──> MLP Fusion
    zA ─┤
    zS ─┘
            ↓
    Unified Patient State
            ↓
       Digital Twin

This module defines and validates the output contract of MLP Fusion.

It does not implement:
    - Digital Twin dynamics
    - Prediction
    - What-if simulation
    - Interactive UI
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class UnifiedPatientState:
    """
    Immutable representation of the unified patient state.

    state:
        Tensor with shape [batch, state_dim].

    The state is the direct representation produced by
    the MLP Fusion stage.
    """

    state: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.state, Tensor):
            raise TypeError(
                "UnifiedPatientState.state must be a torch.Tensor"
            )

        if self.state.ndim != 2:
            raise ValueError(
                "UnifiedPatientState.state must have shape "
                "[batch, state_dim]; "
                f"received {tuple(self.state.shape)}"
            )

        if not torch.is_floating_point(self.state):
            raise TypeError(
                "UnifiedPatientState.state must be floating-point; "
                f"received {self.state.dtype}"
            )

        if not torch.isfinite(self.state).all():
            raise ValueError(
                "UnifiedPatientState.state contains "
                "NaN or infinite values"
            )

    @property
    def batch_size(self) -> int:
        """Return the number of patient samples in the batch."""
        return int(self.state.shape[0])

    @property
    def state_dim(self) -> int:
        """Return the dimensionality of the unified state."""
        return int(self.state.shape[1])

    def detached(self) -> "UnifiedPatientState":
        """
        Return a detached copy of the state.

        This does not alter the original state.
        """
        return UnifiedPatientState(
            state=self.state.detach()
        )

    def to(self, *args, **kwargs) -> "UnifiedPatientState":
        """
        Move the state tensor to another device/dtype.
        """
        return UnifiedPatientState(
            state=self.state.to(*args, **kwargs)
        )


def create_unified_patient_state(
    state: Tensor,
) -> UnifiedPatientState:
    """
    Construct a validated Unified Patient State.
    """
    return UnifiedPatientState(state=state)


def validate_unified_patient_state_contract(
    patient_state: UnifiedPatientState,
    *,
    expected_batch_size: int | None = None,
    expected_state_dim: int | None = None,
) -> None:
    """
    Validate the Unified Patient State contract.

    Optional expected dimensions allow downstream components
    to enforce their interface requirements.
    """

    if not isinstance(
        patient_state,
        UnifiedPatientState,
    ):
        raise TypeError(
            "Expected UnifiedPatientState, got "
            f"{type(patient_state).__name__}"
        )

    if expected_batch_size is not None:
        if patient_state.batch_size != expected_batch_size:
            raise ValueError(
                "Unified Patient State batch-size mismatch: "
                f"expected {expected_batch_size}, "
                f"got {patient_state.batch_size}"
            )

    if expected_state_dim is not None:
        if patient_state.state_dim != expected_state_dim:
            raise ValueError(
                "Unified Patient State dimension mismatch: "
                f"expected {expected_state_dim}, "
                f"got {patient_state.state_dim}"
            )