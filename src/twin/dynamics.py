"""
T1D-UOM Digital Twin Dynamics.

This module implements the learned state-transition component inside the
Digital Twin.

Locked architecture:

    Unified Patient State
            |
            v
       Twin State
            |
       Twin Dynamics
            |
            v
      Simulated State
            |
       +----+----+
       |         |
       v         v
  Prediction   What-if

Scope:
    - Learns a transition from current twin state to next twin state.
    - Optionally accepts explicitly supplied context.
    - Does not access dataset files.
    - Does not modify source data.
    - Does not perform preprocessing.
    - Does not perform imputation.
    - Does not perform normalization.
    - Does not implement Prediction.
    - Does not implement What-if.
    - Does not implement UI.

The transition target is the next observed Unified Patient State.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


__all__ = ["TwinDynamics"]


class TwinDynamics(nn.Module):
    """
    Learned Digital Twin state-transition model.

    The model estimates:

        state(t + Δt) = state(t) + Δstate

    where Δstate is learned from the current state and, when configured,
    explicitly supplied context.

    Parameters
    ----------
    state_dim:
        Dimension of the Unified Patient State.

    hidden_dim:
        Hidden dimension of the transition network.

    context_dim:
        Optional dimensionality of temporal/context information.

    dropout:
        Dropout probability used during training.
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
            raise ValueError("state_dim must be a positive integer.")

        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer.")

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

        input_dim = state_dim + (context_dim or 0)

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, state_dim),
        )

    def _validate_state(
        self,
        state: Tensor,
        name: str,
    ) -> None:
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
        if self.context_dim is None:
            if context is not None:
                raise ValueError(
                    "This TwinDynamics instance was created without "
                    "context_dim; context must therefore be None."
                )
            return

        if context is None:
            raise ValueError(
                "This TwinDynamics instance requires a context tensor."
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
                "context batch size must match state batch size."
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

    def forward(
        self,
        state: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict the next Digital Twin state.

        Parameters
        ----------
        state:
            Current Unified Patient State / Twin State with shape
            [batch, state_dim].

        context:
            Optional contextual information with shape
            [batch, context_dim].

        Returns
        -------
        Tensor
            Predicted next state with shape [batch, state_dim].
        """
        self._validate_state(state, "state")
        self._validate_context(context, state.shape[0])

        if self.context_dim is None:
            model_input = state
        else:
            assert context is not None
            model_input = torch.cat(
                (state, context),
                dim=1,
            )

        delta = self.network(model_input)
        next_state = state + delta

        if not torch.isfinite(next_state).all():
            raise ValueError(
                "TwinDynamics produced non-finite values."
            )

        return next_state

    def loss(
        self,
        current_state: Tensor,
        next_state: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute supervised next-state transition loss.

        The target is the next observed Unified Patient State.

        No target is created or modified inside this method.
        """
        self._validate_state(
            current_state,
            "current_state",
        )
        self._validate_state(
            next_state,
            "next_state",
        )

        if current_state.shape != next_state.shape:
            raise ValueError(
                "current_state and next_state must have identical shapes."
            )

        self._validate_context(
            context,
            current_state.shape[0],
        )

        prediction = self.forward(
            current_state,
            context=context,
        )

        return torch.mean(
            (prediction - next_state) ** 2
        )