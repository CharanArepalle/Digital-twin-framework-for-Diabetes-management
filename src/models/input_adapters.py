"""
T1D-UOM explicit modality input adapters.

Locked architecture:

    Frozen sequence inputs
            |
            v
    Explicit input adapters
            |
            v
        Five GRUs
            |
            v
       zG,zI,zN,zA,zS
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
        /        \
   Prediction   What-if

This module is ONLY the explicit representation boundary between the
frozen physical/semantic sequence inputs and numeric tensors consumed
by the existing FiveGRU model.

It does not modify source CSV files.

It does not resample, interpolate, impute, normalize, or delete data.

It does not create another GRU branch.

It does not implement MLP Fusion, the Digital Twin, Prediction,
What-if, or the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


__all__ = [
    "ModalityAdapter",
    "NutritionAdapter",
    "ActivityAdapter",
    "InsulinAdapter",
    "IdentityNumericAdapter",
]


def _validate_numeric_tensor(
    x: Tensor,
    *,
    expected_features: int,
    name: str,
) -> None:
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")

    if x.ndim != 3:
        raise ValueError(
            f"{name} must have shape "
            "[batch, sequence_length, features]. "
            f"Received {tuple(x.shape)}."
        )

    if x.shape[-1] != expected_features:
        raise ValueError(
            f"{name} expected {expected_features} features; "
            f"received {x.shape[-1]}."
        )

    if not torch.is_floating_point(x):
        raise TypeError(
            f"{name} must be a floating-point tensor."
        )

    if not torch.isfinite(x).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )


class ModalityAdapter(nn.Module):
    """
    Base interface for an explicit modality-to-GRU representation.

    Every adapter produces:

        [batch, sequence_length, output_dim]
    """

    input_dim: int
    output_dim: int

    def _validate_output(
        self,
        output: Tensor,
        *,
        name: str,
    ) -> Tensor:
        if output.ndim != 3:
            raise ValueError(
                f"{name} output must be three-dimensional."
            )

        if output.shape[-1] != self.output_dim:
            raise ValueError(
                f"{name} output expected {self.output_dim} features; "
                f"received {output.shape[-1]}."
            )

        if not torch.isfinite(output).all():
            raise ValueError(
                f"{name} output contains non-finite values."
            )

        return output


class IdentityNumericAdapter(ModalityAdapter):
    """
    Adapter for already numeric modalities.

    Used for:

        Glucose: 1 -> 1
        Sleep:   6 -> 6
    """

    def __init__(
        self,
        input_dim: int,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive."
            )

        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(self, x: Tensor) -> Tensor:
        _validate_numeric_tensor(
            x,
            expected_features=self.input_dim,
            name="numeric modality input",
        )

        return x


class NutritionAdapter(ModalityAdapter):
    """
    Nutrition representation.

    Physical/runtime fields:

        carbs_g
        prot_g
        fat_g
        fibre_g
        meal_type
        meal_tag

    The categorical fields are represented through learned embeddings.

    Output dimension:

        4 numeric
        + meal_type_embedding_dim
        + meal_tag_embedding_dim

    The default dimensions deliberately match the existing FiveGRU
    runtime contract:

        4 + 10 + 10 = 24
    """

    input_dim = 6

    def __init__(
        self,
        *,
        meal_type_cardinality: int,
        meal_tag_cardinality: int,
        meal_type_embedding_dim: int = 10,
        meal_tag_embedding_dim: int = 10,
    ) -> None:
        super().__init__()

        if meal_type_cardinality <= 0:
            raise ValueError(
                "meal_type_cardinality must be positive."
            )

        if meal_tag_cardinality <= 0:
            raise ValueError(
                "meal_tag_cardinality must be positive."
            )

        if meal_type_embedding_dim <= 0:
            raise ValueError(
                "meal_type_embedding_dim must be positive."
            )

        if meal_tag_embedding_dim <= 0:
            raise ValueError(
                "meal_tag_embedding_dim must be positive."
            )

        self.meal_type_cardinality = meal_type_cardinality
        self.meal_tag_cardinality = meal_tag_cardinality
        self.meal_type_embedding_dim = meal_type_embedding_dim
        self.meal_tag_embedding_dim = meal_tag_embedding_dim

        self.meal_type_embedding = nn.Embedding(
            meal_type_cardinality,
            meal_type_embedding_dim,
        )

        self.meal_tag_embedding = nn.Embedding(
            meal_tag_cardinality,
            meal_tag_embedding_dim,
        )

        self.output_dim = (
            4
            + meal_type_embedding_dim
            + meal_tag_embedding_dim
        )

    def forward(
        self,
        numeric: Tensor,
        meal_type_ids: Tensor,
        meal_tag_ids: Tensor,
    ) -> Tensor:
        _validate_numeric_tensor(
            numeric,
            expected_features=4,
            name="nutrition numeric input",
        )

        if meal_type_ids.shape != numeric.shape[:2]:
            raise ValueError(
                "meal_type_ids must have shape "
                "[batch, sequence_length]."
            )

        if meal_tag_ids.shape != numeric.shape[:2]:
            raise ValueError(
                "meal_tag_ids must have shape "
                "[batch, sequence_length]."
            )

        if meal_type_ids.dtype != torch.long:
            raise TypeError(
                "meal_type_ids must use torch.long."
            )

        if meal_tag_ids.dtype != torch.long:
            raise TypeError(
                "meal_tag_ids must use torch.long."
            )

        if meal_type_ids.min() < 0:
            raise ValueError(
                "meal_type_ids contains a negative category index."
            )

        if meal_type_ids.max() >= self.meal_type_cardinality:
            raise ValueError(
                "meal_type_ids contains an out-of-range category index."
            )

        if meal_tag_ids.min() < 0:
            raise ValueError(
                "meal_tag_ids contains a negative category index."
            )

        if meal_tag_ids.max() >= self.meal_tag_cardinality:
            raise ValueError(
                "meal_tag_ids contains an out-of-range category index."
            )

        meal_type_emb = self.meal_type_embedding(
            meal_type_ids
        )

        meal_tag_emb = self.meal_tag_embedding(
            meal_tag_ids
        )

        output = torch.cat(
            (
                numeric,
                meal_type_emb,
                meal_tag_emb,
            ),
            dim=-1,
        )

        return self._validate_output(
            output,
            name="nutrition",
        )


class ActivityAdapter(ModalityAdapter):
    """
    Activity representation.

    Physical/runtime fields:

        activity_type
        active_Kcal
        step_count
        distance_m
        duration_s
        active_time_s
        start_time_s
        start_time_offset_s
        met
        intensity
        motion_intensity_mean
        motion_intensity_max

    Output:

        10 numeric
        + activity_type embedding (4)
        + intensity embedding (3)

        = 17 features

    This matches the existing FiveGRU runtime input dimension.
    """

    input_dim = 12

    def __init__(
        self,
        *,
        activity_type_cardinality: int,
        intensity_cardinality: int,
        activity_type_embedding_dim: int = 4,
        intensity_embedding_dim: int = 3,
    ) -> None:
        super().__init__()

        if activity_type_cardinality <= 0:
            raise ValueError(
                "activity_type_cardinality must be positive."
            )

        if intensity_cardinality <= 0:
            raise ValueError(
                "intensity_cardinality must be positive."
            )

        if activity_type_embedding_dim <= 0:
            raise ValueError(
                "activity_type_embedding_dim must be positive."
            )

        if intensity_embedding_dim <= 0:
            raise ValueError(
                "intensity_embedding_dim must be positive."
            )

        self.activity_type_cardinality = (
            activity_type_cardinality
        )
        self.intensity_cardinality = (
            intensity_cardinality
        )

        self.activity_type_embedding = nn.Embedding(
            activity_type_cardinality,
            activity_type_embedding_dim,
        )

        self.intensity_embedding = nn.Embedding(
            intensity_cardinality,
            intensity_embedding_dim,
        )

        self.output_dim = (
            10
            + activity_type_embedding_dim
            + intensity_embedding_dim
        )

    def forward(
        self,
        numeric: Tensor,
        activity_type_ids: Tensor,
        intensity_ids: Tensor,
    ) -> Tensor:
        _validate_numeric_tensor(
            numeric,
            expected_features=10,
            name="activity numeric input",
        )

        if activity_type_ids.shape != numeric.shape[:2]:
            raise ValueError(
                "activity_type_ids must have shape "
                "[batch, sequence_length]."
            )

        if intensity_ids.shape != numeric.shape[:2]:
            raise ValueError(
                "intensity_ids must have shape "
                "[batch, sequence_length]."
            )

        if activity_type_ids.dtype != torch.long:
            raise TypeError(
                "activity_type_ids must use torch.long."
            )

        if intensity_ids.dtype != torch.long:
            raise TypeError(
                "intensity_ids must use torch.long."
            )

        if activity_type_ids.min() < 0:
            raise ValueError(
                "activity_type_ids contains a negative category index."
            )

        if activity_type_ids.max() >= self.activity_type_cardinality:
            raise ValueError(
                "activity_type_ids contains an out-of-range category index."
            )

        if intensity_ids.min() < 0:
            raise ValueError(
                "intensity_ids contains a negative category index."
            )

        if intensity_ids.max() >= self.intensity_cardinality:
            raise ValueError(
                "intensity_ids contains an out-of-range category index."
            )

        activity_type_emb = (
            self.activity_type_embedding(
                activity_type_ids
            )
        )

        intensity_emb = (
            self.intensity_embedding(
                intensity_ids
            )
        )

        output = torch.cat(
            (
                numeric,
                activity_type_emb,
                intensity_emb,
            ),
            dim=-1,
        )

        return self._validate_output(
            output,
            name="activity",
        )


class InsulinAdapter(ModalityAdapter):
    """
    Single unified Insulin representation.

    Frozen contract:

        dose
        event_type

    Mapping:

        basal -> 0
        bolus -> 1

    No second insulin GRU is introduced.
    """

    input_dim = 2
    output_dim = 2

    BASAL_EVENT_TYPE = 0
    BOLUS_EVENT_TYPE = 1

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        dose: Tensor,
        event_type: Tensor,
    ) -> Tensor:
        if dose.ndim != 3:
            raise ValueError(
                "dose must have shape "
                "[batch, sequence_length, 1]."
            )

        if dose.shape[-1] != 1:
            raise ValueError(
                "dose must have exactly one feature."
            )

        if not torch.is_floating_point(dose):
            raise TypeError(
                "dose must be floating point."
            )

        if not torch.isfinite(dose).all():
            raise ValueError(
                "dose contains non-finite values."
            )

        if event_type.shape != dose.shape[:2]:
            raise ValueError(
                "event_type must have shape "
                "[batch, sequence_length]."
            )

        if event_type.dtype != torch.float32:
            event_type = event_type.to(
                dtype=torch.float32
            )

        if not torch.isfinite(event_type).all():
            raise ValueError(
                "event_type contains non-finite values."
            )

        if not torch.all(
            (event_type == 0.0)
            | (event_type == 1.0)
        ):
            raise ValueError(
                "event_type must contain only 0 (basal) "
                "or 1 (bolus)."
            )

        return torch.cat(
            (
                dose,
                event_type.unsqueeze(-1),
            ),
            dim=-1,
        )