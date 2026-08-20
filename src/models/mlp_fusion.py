"""
T1D-UOM — MLP Fusion Network.

Architectural position
----------------------

    zG  ─┐
    zI  ─┤
    zN  ─┼──> MLP Fusion ──> Unified Patient State
    zA  ─┤
    zS  ─┘

This module performs ONLY the MLP Fusion stage.

It does not:
    - modify datasets
    - construct windows
    - create targets
    - impute values
    - normalize values
    - encode categorical variables
    - implement Digital Twin
    - implement prediction
    - implement What-if simulation
    - implement UI
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn


# Frozen latent-input names and order.
LATENT_NAMES: tuple[str, ...] = (
    "zG",
    "zI",
    "zN",
    "zA",
    "zS",
)


class MLPFusion(nn.Module):
    """
    Fuse the five modality-specific GRU latent representations.

    Expected inputs:

        zG: [batch, hidden_dim]
        zI: [batch, hidden_dim]
        zN: [batch, hidden_dim]
        zA: [batch, hidden_dim]
        zS: [batch, hidden_dim]

    Output:

        Unified Patient State:
            [batch, fusion_dim]
    """

    def __init__(
        self,
        hidden_dim: int,
        fusion_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer, got {hidden_dim!r}"
            )

        if not isinstance(fusion_dim, int) or fusion_dim <= 0:
            raise ValueError(
                f"fusion_dim must be a positive integer, got {fusion_dim!r}"
            )

        if not isinstance(dropout, (float, int)):
            raise TypeError(
                f"dropout must be numeric, got {type(dropout).__name__}"
            )

        dropout = float(dropout)

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must satisfy 0.0 <= dropout < 1.0, got {dropout}"
            )

        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        self.dropout = dropout

        input_dim = len(LATENT_NAMES) * hidden_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(self, **latents: Tensor) -> Tensor:
        """
        Fuse exactly the five frozen latent representations.

        The accepted keyword names are exactly:

            zG, zI, zN, zA, zS

        Any missing or unexpected latent name is rejected explicitly.
        """

        _validate_latent_inputs(
            latents,
            hidden_dim=self.hidden_dim,
        )

        # IMPORTANT:
        # Concatenation order is frozen and explicit.
        fused = torch.cat(
            [
                latents["zG"],
                latents["zI"],
                latents["zN"],
                latents["zA"],
                latents["zS"],
            ],
            dim=-1,
        )

        return self.network(fused)


def _validate_latent_inputs(
    latent: Mapping[str, Tensor],
    *,
    hidden_dim: int,
) -> None:
    """
    Validate the five latent tensors before fusion.
    """

    expected = set(LATENT_NAMES)
    received = set(latent.keys())

    if received != expected:
        missing = [
            name
            for name in LATENT_NAMES
            if name not in received
        ]

        unexpected = sorted(
            name
            for name in received
            if name not in expected
        )

        details: list[str] = [
            "MLP Fusion requires exactly the five latent inputs "
            f"{LATENT_NAMES}."
        ]

        if missing:
            details.append(
                f"Missing latent inputs: {missing}."
            )

        if unexpected:
            details.append(
                f"Unexpected latent inputs: {unexpected}."
            )

        raise ValueError(" ".join(details))

    batch_size: int | None = None

    for name in LATENT_NAMES:
        tensor = latent[name]

        if not isinstance(tensor, Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor, "
                f"got {type(tensor).__name__}"
            )

        if tensor.ndim != 2:
            raise ValueError(
                f"{name} must have shape [batch, hidden_dim]; "
                f"received shape {tuple(tensor.shape)}"
            )

        if tensor.shape[-1] != hidden_dim:
            raise ValueError(
                f"{name} has hidden dimension {tensor.shape[-1]}; "
                f"expected {hidden_dim}"
            )

        if not torch.is_floating_point(tensor):
            raise TypeError(
                f"{name} must be floating-point; "
                f"received dtype {tensor.dtype}"
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"{name} contains NaN or infinite values"
            )

        current_batch = tensor.shape[0]

        if batch_size is None:
            batch_size = current_batch
        elif current_batch != batch_size:
            raise ValueError(
                "All latent tensors must have the same batch size; "
                f"expected {batch_size}, "
                f"{name} has {current_batch}"
            )


def validate_mlp_fusion(
    model: MLPFusion,
) -> None:
    """
    Validate the structural MLP Fusion contract.
    """

    if not isinstance(model, MLPFusion):
        raise TypeError(
            f"Expected MLPFusion, got {type(model).__name__}"
        )

    expected_input_dim = len(LATENT_NAMES) * model.hidden_dim

    first_layer = model.network[0]

    if not isinstance(first_layer, nn.Linear):
        raise AssertionError(
            "MLP Fusion first layer must be nn.Linear"
        )

    if first_layer.in_features != expected_input_dim:
        raise AssertionError(
            "MLP Fusion input dimension mismatch: "
            f"expected {expected_input_dim}, "
            f"got {first_layer.in_features}"
        )

    if first_layer.out_features != model.fusion_dim:
        raise AssertionError(
            "MLP Fusion first projection dimension mismatch: "
            f"expected {model.fusion_dim}, "
            f"got {first_layer.out_features}"
        )

    last_layer = model.network[-1]

    if not isinstance(last_layer, nn.Linear):
        raise AssertionError(
            "MLP Fusion final layer must be nn.Linear"
        )

    if last_layer.out_features != model.fusion_dim:
        raise AssertionError(
            "Unified Patient State dimension mismatch: "
            f"expected {model.fusion_dim}, "
            f"got {last_layer.out_features}"
        )


def validate_unified_patient_state(
    state: Tensor,
    *,
    batch_size: int,
    fusion_dim: int,
) -> None:
    """
    Validate the output contract of MLP Fusion.
    """

    if not isinstance(state, Tensor):
        raise TypeError(
            "Unified Patient State must be a torch.Tensor, "
            f"got {type(state).__name__}"
        )

    expected_shape = (
        batch_size,
        fusion_dim,
    )

    if tuple(state.shape) != expected_shape:
        raise AssertionError(
            "Unified Patient State shape mismatch: "
            f"expected {expected_shape}, "
            f"got {tuple(state.shape)}"
        )

    if not torch.is_floating_point(state):
        raise TypeError(
            "Unified Patient State must be floating-point; "
            f"got {state.dtype}"
        )

    if not torch.isfinite(state).all():
        raise ValueError(
            "Unified Patient State contains NaN or infinite values"
        )