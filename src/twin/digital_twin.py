"""
T1D-UOM — Patient-Specific Digital Twin.

Architecture position:

    Unified Patient State
            |
            v
       Digital Twin
            |
       Twin State
            |
       Twin Dynamics
            |
            v
     Simulated State
        /       \
 Prediction    What-if

Responsibilities
----------------
This module owns the Digital Twin state-management/orchestration layer.

It is responsible for:
    - initializing a Digital Twin from the Unified Patient State
    - maintaining an immutable Digital Twin state representation
    - applying observed patient-state updates
    - constructing isolated hypothetical scenario states
    - delegating learned state evolution to TwinDynamics

The learned transition model itself lives exclusively in:
    twin/dynamics.py

This module does NOT:
    - access dataset files
    - modify dataset files
    - perform preprocessing
    - perform resampling
    - perform interpolation
    - perform imputation
    - perform normalization
    - perform feature engineering
    - perform categorical encoding
    - create windows
    - create targets
    - implement Prediction
    - implement What-if decision logic
    - implement the Interactive UI

Important
---------
TwinDynamics is a learned transition model. Its outputs must be trained
and validated before they are interpreted as meaningful physiological or
clinical predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .dynamics import TwinDynamics


__all__ = [
    "DigitalTwinState",
    "DigitalTwin",
]


@dataclass(frozen=True)
class DigitalTwinState:
    """
    Immutable representation of the current Digital Twin state.

    Parameters
    ----------
    value:
        Tensor with shape [batch, state_dim].
    """

    value: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.value, Tensor):
            raise TypeError(
                "DigitalTwinState.value must be a torch.Tensor."
            )

        if self.value.ndim != 2:
            raise ValueError(
                "DigitalTwinState.value must have shape "
                "[batch, state_dim]."
            )

        if not torch.is_floating_point(self.value):
            raise TypeError(
                "DigitalTwinState.value must use a floating-point dtype."
            )

        if not torch.isfinite(self.value).all():
            raise ValueError(
                "DigitalTwinState.value must contain only finite values."
            )

    @property
    def batch_size(self) -> int:
        """Return the number of patient states in the batch."""
        return int(self.value.shape[0])

    @property
    def state_dim(self) -> int:
        """Return the dimensionality of the patient state."""
        return int(self.value.shape[1])


class DigitalTwin(nn.Module):
    """
    Patient-specific Digital Twin state engine.

    The Digital Twin owns the state-management boundary while delegating
    learned state evolution to the single authoritative TwinDynamics
    component.

    Supported operations
    --------------------
    1. initialize
       Unified Patient State -> DigitalTwinState

    2. update
       observed Unified Patient State -> replacement DigitalTwinState

    3. scenario
       current DigitalTwinState + delta -> isolated hypothetical state

    4. evolve
       current DigitalTwinState -> next simulated DigitalTwinState

    Architecture
    ------------
        Unified Patient State
                |
                v
           Digital Twin
                |
           TwinDynamics
                |
                v
         Simulated State
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(state_dim, int) or state_dim <= 0:
            raise ValueError(
                "state_dim must be a positive integer."
            )

        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be a positive integer."
            )

        if context_dim is not None:
            if not isinstance(context_dim, int) or context_dim <= 0:
                raise ValueError(
                    "context_dim must be None or a positive integer."
                )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0.0 <= dropout < 1.0."
            )

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        # TwinDynamics is the single authoritative learned
        # state-transition implementation.
        self.dynamics = TwinDynamics(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            dropout=dropout,
        )

    def _validate_state(
        self,
        state: Tensor,
        name: str,
    ) -> None:
        """Validate a tensor against the Digital Twin state contract."""

        if not isinstance(state, Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if state.ndim != 2:
            raise ValueError(
                f"{name} must have shape [batch, state_dim]."
            )

        if state.shape[1] != self.state_dim:
            raise ValueError(
                f"{name} has state dimension {state.shape[1]}; "
                f"expected {self.state_dim}."
            )

        if not torch.is_floating_point(state):
            raise TypeError(
                f"{name} must use a floating-point dtype."
            )

        if not torch.isfinite(state).all():
            raise ValueError(
                f"{name} must contain only finite values."
            )

    def _validate_context(
        self,
        context: Optional[Tensor],
        batch_size: int,
    ) -> None:
        """
        Validate optional context against the TwinDynamics contract.
        """

        if self.context_dim is None:
            if context is not None:
                raise ValueError(
                    "This Digital Twin was created without context_dim; "
                    "context must therefore be None."
                )
            return

        if context is None:
            raise ValueError(
                "This Digital Twin requires a context tensor."
            )

        if not isinstance(context, Tensor):
            raise TypeError(
                "context must be a torch.Tensor."
            )

        if context.ndim != 2:
            raise ValueError(
                "context must have shape [batch, context_dim]."
            )

        if context.shape[0] != batch_size:
            raise ValueError(
                "context batch size must match the twin-state batch size."
            )

        if context.shape[1] != self.context_dim:
            raise ValueError(
                f"context has dimension {context.shape[1]}; "
                f"expected {self.context_dim}."
            )

        if not torch.is_floating_point(context):
            raise TypeError(
                "context must use a floating-point dtype."
            )

        if not torch.isfinite(context).all():
            raise ValueError(
                "context must contain only finite values."
            )

    def initialize(
        self,
        patient_state: Tensor,
    ) -> DigitalTwinState:
        """
        Initialize a Digital Twin from the Unified Patient State.

        The supplied tensor is cloned so that the Digital Twin does not
        mutate the caller's tensor.
        """

        self._validate_state(
            patient_state,
            "patient_state",
        )

        return DigitalTwinState(
            value=patient_state.clone()
        )

    def update(
        self,
        twin_state: DigitalTwinState,
        patient_state: Tensor,
    ) -> DigitalTwinState:
        """
        Replace the current twin state with a newly observed patient state.

        This is an observation update, not a prediction or simulation.
        """

        if not isinstance(twin_state, DigitalTwinState):
            raise TypeError(
                "twin_state must be a DigitalTwinState."
            )

        self._validate_state(
            twin_state.value,
            "twin_state.value",
        )

        self._validate_state(
            patient_state,
            "patient_state",
        )

        if twin_state.batch_size != patient_state.shape[0]:
            raise ValueError(
                "patient_state batch size must match "
                "twin_state batch size."
            )

        return DigitalTwinState(
            value=patient_state.clone()
        )

    def scenario(
        self,
        twin_state: DigitalTwinState,
        delta: Tensor,
    ) -> DigitalTwinState:
        """
        Construct an isolated hypothetical scenario state.

        Mathematical operation:

            scenario_state = twin_state + delta

        The original Digital Twin state is never modified.

        This method provides the state-construction primitive required by
        the later What-if branch. It does not itself implement What-if
        decision logic or UI behavior.
        """

        if not isinstance(twin_state, DigitalTwinState):
            raise TypeError(
                "twin_state must be a DigitalTwinState."
            )

        self._validate_state(
            twin_state.value,
            "twin_state.value",
        )

        self._validate_state(
            delta,
            "delta",
        )

        if delta.shape != twin_state.value.shape:
            raise ValueError(
                "delta must have exactly the same shape "
                "as twin_state.value."
            )

        return DigitalTwinState(
            value=twin_state.value + delta
        )

    def evolve(
        self,
        twin_state: DigitalTwinState,
        context: Optional[Tensor] = None,
    ) -> DigitalTwinState:
        """
        Evolve the Digital Twin by one modeled time step.

        State transition:

            current_state
                  |
                  v
             TwinDynamics
                  |
                  v
             next_state

        TwinDynamics is the only learned transition mechanism used by
        the Digital Twin.

        The method does not claim that the current transition parameters
        are physiologically valid. Training and validation are required
        before interpreting simulated states scientifically.
        """

        if not isinstance(twin_state, DigitalTwinState):
            raise TypeError(
                "twin_state must be a DigitalTwinState."
            )

        current = twin_state.value

        self._validate_state(
            current,
            "twin_state.value",
        )

        self._validate_context(
            context,
            current.shape[0],
        )

        next_state = self.dynamics(
            state=current,
            context=context,
        )

        if not torch.isfinite(next_state).all():
            raise ValueError(
                "Digital Twin evolution produced non-finite values."
            )

        return DigitalTwinState(
            value=next_state
        )

    def forward(
        self,
        twin_state: DigitalTwinState,
        context: Optional[Tensor] = None,
    ) -> DigitalTwinState:
        """
        PyTorch forward interface for one Digital Twin evolution step.
        """

        return self.evolve(
            twin_state=twin_state,
            context=context,
        )