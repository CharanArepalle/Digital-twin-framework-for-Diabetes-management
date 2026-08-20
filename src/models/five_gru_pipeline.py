"""
T1D-UOM Five-GRU -> MLP Fusion -> Unified Patient State pipeline.

Locked architecture:

    Input representations
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

This module is an integration layer only.

Important:
    The five GRU branches may have different padded sequence lengths.
    The existing FiveGRU length contract selects the valid final
    timestep independently for each modality.

This module does NOT:
    - modify source data;
    - create another GRU;
    - modify FiveGRU;
    - modify MLPFusion;
    - implement Digital Twin;
    - implement TwinDynamics;
    - implement Prediction;
    - implement What-if;
    - implement UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from .five_gru import (
    ACTIVITY_DIM,
    GLUCOSE_DIM,
    INSULIN_DIM,
    NUTRITION_DIM,
    SLEEP_DIM,
    FiveGRU,
)
from .mlp_fusion import MLPFusion
from .patient_state import UnifiedPatientState


__all__ = [
    "FiveGRUInputBatch",
    "FiveGRUStatePipeline",
]


@dataclass(frozen=True)
class FiveGRUInputBatch:
    """
    Model-ready inputs for the five GRU branches.

    Each tensor has shape:

        [batch, padded_sequence_length, feature_dim]

    lengths is optional.

    When supplied, it contains the valid sequence length for each
    sample and modality:

        {
            "glucose": Tensor[batch],
            "insulin": Tensor[batch],
            "nutrition": Tensor[batch],
            "activity": Tensor[batch],
            "sleep": Tensor[batch],
        }

    Different modalities are explicitly allowed to have different
    padded sequence lengths.
    """

    glucose: Tensor
    insulin: Tensor
    nutrition: Tensor
    activity: Tensor
    sleep: Tensor
    lengths: Mapping[str, Tensor] | None = None


class FiveGRUStatePipeline(nn.Module):
    """
    Compose the locked:

        FiveGRU -> MLPFusion -> UnifiedPatientState

    No additional neural architecture is introduced.
    """

    def __init__(
        self,
        *,
        five_gru: FiveGRU,
        fusion: MLPFusion,
    ) -> None:
        super().__init__()

        if not isinstance(
            five_gru,
            FiveGRU,
        ):
            raise TypeError(
                "five_gru must be a FiveGRU instance."
            )

        if not isinstance(
            fusion,
            MLPFusion,
        ):
            raise TypeError(
                "fusion must be an MLPFusion instance."
            )

        # FiveGRU intentionally does not expose a top-level
        # hidden_dim attribute. Read the authoritative hidden
        # dimensions from the five underlying nn.GRU modules.
        hidden_dims = {
            five_gru.glucose_gru.gru.hidden_size,
            five_gru.insulin_gru.gru.hidden_size,
            five_gru.nutrition_gru.gru.hidden_size,
            five_gru.activity_gru.gru.hidden_size,
            five_gru.sleep_gru.gru.hidden_size,
        }

        if len(hidden_dims) != 1:
            raise ValueError(
                "All five FiveGRU branches must use the same "
                "hidden dimension."
            )

        five_gru_hidden_dim = next(
            iter(hidden_dims)
        )

        if fusion.hidden_dim != five_gru_hidden_dim:
            raise ValueError(
                "MLPFusion hidden_dim must match FiveGRU "
                f"hidden dimension; expected "
                f"{five_gru_hidden_dim}, received "
                f"{fusion.hidden_dim}."
            )

        self.five_gru = five_gru
        self.fusion = fusion

        self.hidden_dim = five_gru_hidden_dim
        self.state_dim = fusion.fusion_dim

    @staticmethod
    def _validate_branch(
        x: Tensor,
        *,
        expected_dim: int,
        name: str,
    ) -> None:
        if not isinstance(
            x,
            Tensor,
        ):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if x.ndim != 3:
            raise ValueError(
                f"{name} must have shape "
                "[batch, sequence_length, features]. "
                f"Received {tuple(x.shape)}."
            )

        if x.shape[-1] != expected_dim:
            raise ValueError(
                f"{name} expected {expected_dim} features; "
                f"received {x.shape[-1]}."
            )

        if not torch.is_floating_point(x):
            raise TypeError(
                f"{name} must be floating point."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                f"{name} contains non-finite values."
            )

        if x.shape[1] <= 0:
            raise ValueError(
                f"{name} sequence length must be positive."
            )

    @staticmethod
    def _validate_lengths(
        lengths: Mapping[str, Tensor] | None,
        *,
        batch_size: int,
        inputs: FiveGRUInputBatch,
    ) -> dict[str, Tensor] | None:
        """
        Validate and normalize the optional per-modality lengths.

        The actual FiveGRU implementation also validates these values.
        This wrapper validates them early so failures are explicit.
        """

        if lengths is None:
            return None

        expected_names = {
            "glucose",
            "insulin",
            "nutrition",
            "activity",
            "sleep",
        }

        received_names = set(lengths.keys())

        unexpected = received_names - expected_names

        if unexpected:
            raise ValueError(
                "Unexpected sequence-length keys: "
                f"{sorted(unexpected)}."
            )

        normalized: dict[str, Tensor] = {}

        tensors = {
            "glucose": inputs.glucose,
            "insulin": inputs.insulin,
            "nutrition": inputs.nutrition,
            "activity": inputs.activity,
            "sleep": inputs.sleep,
        }

        for name in expected_names:
            if name not in lengths:
                continue

            value = lengths[name]

            if not isinstance(
                value,
                Tensor,
            ):
                raise TypeError(
                    f"lengths['{name}'] must be a torch.Tensor."
                )

            if value.ndim != 1:
                raise ValueError(
                    f"lengths['{name}'] must have shape "
                    f"[batch]."
                )

            if value.shape[0] != batch_size:
                raise ValueError(
                    f"lengths['{name}'] batch dimension does "
                    f"not match the inputs."
                )

            if value.numel() == 0:
                raise ValueError(
                    f"lengths['{name}'] must not be empty."
                )

            value_long = value.to(
                device=tensors[name].device,
                dtype=torch.long,
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
                    "greater than the supplied sequence length."
                )

            normalized[name] = value_long

        return normalized

    def forward(
        self,
        inputs: FiveGRUInputBatch,
    ) -> UnifiedPatientState:
        """
        Run:

            FiveGRU -> MLPFusion -> UnifiedPatientState

        with optional independent modality sequence lengths.
        """

        if not isinstance(
            inputs,
            FiveGRUInputBatch,
        ):
            raise TypeError(
                "inputs must be a FiveGRUInputBatch."
            )

        self._validate_branch(
            inputs.glucose,
            expected_dim=GLUCOSE_DIM,
            name="glucose",
        )

        self._validate_branch(
            inputs.insulin,
            expected_dim=INSULIN_DIM,
            name="insulin",
        )

        self._validate_branch(
            inputs.nutrition,
            expected_dim=NUTRITION_DIM,
            name="nutrition",
        )

        self._validate_branch(
            inputs.activity,
            expected_dim=ACTIVITY_DIM,
            name="activity",
        )

        self._validate_branch(
            inputs.sleep,
            expected_dim=SLEEP_DIM,
            name="sleep",
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
                "All five modality branches must have the same "
                "batch size."
            )

        batch_size = inputs.glucose.shape[0]

        lengths = self._validate_lengths(
            inputs.lengths,
            batch_size=batch_size,
            inputs=inputs,
        )

        # Existing FiveGRU already owns the independent sequence
        # length mechanism. We simply pass it through.
        latents = self.five_gru(
            glucose=inputs.glucose,
            insulin=inputs.insulin,
            nutrition=inputs.nutrition,
            activity=inputs.activity,
            sleep=inputs.sleep,
            lengths=lengths,
        )

        expected_latent_names = {
            "zG",
            "zI",
            "zN",
            "zA",
            "zS",
        }

        if set(latents.keys()) != expected_latent_names:
            raise RuntimeError(
                "FiveGRU returned an unexpected latent contract. "
                f"Expected {sorted(expected_latent_names)}, "
                f"received {sorted(latents.keys())}."
            )

        fused = self.fusion(
            zG=latents["zG"],
            zI=latents["zI"],
            zN=latents["zN"],
            zA=latents["zA"],
            zS=latents["zS"],
        )

        return UnifiedPatientState(
            fused
        )