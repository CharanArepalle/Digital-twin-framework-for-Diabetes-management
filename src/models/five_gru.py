"""
T1D-UOM — Frozen Five-GRU Architecture

Architecture
------------
Glucose    -> GRU -> zG
Insulin    -> GRU -> zI
Nutrition  -> GRU -> zN
Activity   -> GRU -> zA
Sleep      -> GRU -> zS

This module intentionally implements ONLY the five modality-specific GRUs.

Not implemented here:
- MLP Fusion
- Unified Patient State
- Digital Twin
- Prediction
- What-if
- Interactive UI

Runtime input dimensions
------------------------
Glucose    : 1
Insulin    : 2
Nutrition  : 24
Activity   : 17
Sleep      : 6
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Frozen runtime dimensions
# ---------------------------------------------------------------------------

GLUCOSE_DIM = 1
INSULIN_DIM = 2
NUTRITION_DIM = 24
ACTIVITY_DIM = 17
SLEEP_DIM = 6

BRANCH_NAMES = (
    "glucose",
    "insulin",
    "nutrition",
    "activity",
    "sleep",
)

LATENT_NAMES = (
    "zG",
    "zI",
    "zN",
    "zA",
    "zS",
)


class ModalityGRU(nn.Module):
    """
    One independent GRU branch.

    Input:
        [batch, sequence_length, input_dim]

    Output:
        [batch, hidden_dim]

    If lengths are supplied, the output corresponding to the last valid
    timestep of each sequence is returned.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        # PyTorch ignores GRU dropout when num_layers == 1.
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "GRU input must have shape "
                "[batch, sequence_length, feature_dim]. "
                f"Received shape={tuple(x.shape)}."
            )

        output, hidden = self.gru(x)

        if lengths is None:
            return hidden[-1]

        if lengths.ndim != 1:
            raise ValueError(
                "lengths must have shape [batch]. "
                f"Received shape={tuple(lengths.shape)}."
            )

        if lengths.shape[0] != x.shape[0]:
            raise ValueError(
                "lengths batch dimension does not match x."
            )

        lengths = lengths.to(device=x.device, dtype=torch.long)

        if torch.any(lengths <= 0):
            raise ValueError("All sequence lengths must be > 0.")

        if torch.any(lengths > x.shape[1]):
            raise ValueError(
                "A sequence length exceeds the supplied sequence length."
            )

        last_index = lengths - 1

        batch_index = torch.arange(
            x.shape[0],
            device=x.device,
        )

        return output[batch_index, last_index]


class FiveGRU(nn.Module):
    """
    Frozen T1D-UOM five-branch architecture.

    Exactly five GRUs are instantiated:

        glucose  -> GRU -> zG
        insulin  -> GRU -> zI
        nutrition -> GRU -> zN
        activity -> GRU -> zA
        sleep -> GRU -> zS

    No fusion or downstream components are included.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.glucose_gru = ModalityGRU(
            input_dim=GLUCOSE_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.insulin_gru = ModalityGRU(
            input_dim=INSULIN_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.nutrition_gru = ModalityGRU(
            input_dim=NUTRITION_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.activity_gru = ModalityGRU(
            input_dim=ACTIVITY_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.sleep_gru = ModalityGRU(
            input_dim=SLEEP_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        glucose: Tensor,
        insulin: Tensor,
        nutrition: Tensor,
        activity: Tensor,
        sleep: Tensor,
        lengths: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        lengths = lengths or {}

        return {
            "zG": self.glucose_gru(
                glucose,
                lengths.get("glucose"),
            ),
            "zI": self.insulin_gru(
                insulin,
                lengths.get("insulin"),
            ),
            "zN": self.nutrition_gru(
                nutrition,
                lengths.get("nutrition"),
            ),
            "zA": self.activity_gru(
                activity,
                lengths.get("activity"),
            ),
            "zS": self.sleep_gru(
                sleep,
                lengths.get("sleep"),
            ),
        }


def validate_five_gru(model: FiveGRU) -> None:
    """
    Hard architecture guard.

    This prevents accidental introduction/removal of modality GRUs.
    """

    expected = {
        "glucose_gru",
        "insulin_gru",
        "nutrition_gru",
        "activity_gru",
        "sleep_gru",
    }

    actual = {
        name
        for name, module in model.named_children()
        if isinstance(module, ModalityGRU)
    }

    if actual != expected:
        raise AssertionError(
            "Frozen five-GRU architecture violation.\n"
            f"Expected: {sorted(expected)}\n"
            f"Actual:   {sorted(actual)}"
        )

    gru_count = sum(
        1
        for module in model.modules()
        if isinstance(module, nn.GRU)
    )

    if gru_count != 5:
        raise AssertionError(
            f"Expected exactly 5 nn.GRU modules; found {gru_count}."
        )


def validate_runtime_dimensions(model: FiveGRU) -> None:
    """
    Verify the five frozen input dimensions.
    """

    expected = {
        "glucose_gru": GLUCOSE_DIM,
        "insulin_gru": INSULIN_DIM,
        "nutrition_gru": NUTRITION_DIM,
        "activity_gru": ACTIVITY_DIM,
        "sleep_gru": SLEEP_DIM,
    }

    for name, expected_dim in expected.items():
        branch = getattr(model, name)

        actual_dim = branch.gru.input_size

        if actual_dim != expected_dim:
            raise AssertionError(
                f"{name}: expected input_dim={expected_dim}, "
                f"found {actual_dim}."
            )


def validate_output_contract(
    outputs: Dict[str, Tensor],
    batch_size: int,
    hidden_dim: int,
) -> None:
    expected_keys = set(LATENT_NAMES)

    if set(outputs) != expected_keys:
        raise AssertionError(
            f"Expected latent outputs {sorted(expected_keys)}, "
            f"found {sorted(outputs)}."
        )

    expected_shape = (batch_size, hidden_dim)

    for name in LATENT_NAMES:
        if tuple(outputs[name].shape) != expected_shape:
            raise AssertionError(
                f"{name}: expected shape {expected_shape}, "
                f"found {tuple(outputs[name].shape)}."
            )


__all__ = [
    "GLUCOSE_DIM",
    "INSULIN_DIM",
    "NUTRITION_DIM",
    "ACTIVITY_DIM",
    "SLEEP_DIM",
    "BRANCH_NAMES",
    "LATENT_NAMES",
    "ModalityGRU",
    "FiveGRU",
    "validate_five_gru",
    "validate_runtime_dimensions",
    "validate_output_contract",
]