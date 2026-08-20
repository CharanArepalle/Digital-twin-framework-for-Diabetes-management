"""
Train and checkpoint the locked FiveGRU -> MLPFusion state encoder.

This module does NOT modify:
    FiveGRU
    MLPFusion
    FiveGRUStatePipeline
    UnifiedPatientState
    DigitalTwin
    TwinDynamics

It provides the training wrapper/checkpoint boundary needed before
real Unified Patient State trajectories can be generated.

The encoder architecture remains:

    five modality inputs
            |
         FiveGRU
            |
       zG zI zN zA zS
            |
        MLPFusion
            |
    Unified Patient State
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn

from src.models.five_gru import FiveGRU
from src.models.five_gru_pipeline import (
    FiveGRUInputBatch,
    FiveGRUStatePipeline,
)
from src.models.mlp_fusion import MLPFusion


__all__ = [
    "StateEncoder",
    "StateEncoderConfig",
    "save_state_encoder",
    "load_state_encoder",
]


@dataclass(frozen=True)
class StateEncoderConfig:
    """Frozen configuration for the state encoder."""

    hidden_dim: int = 64
    fusion_hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0


class StateEncoder(nn.Module):
    """
    Locked FiveGRU -> MLPFusion encoder.

    This class intentionally contains no additional recurrent branch.
    """

    def __init__(
        self,
        config: StateEncoderConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or StateEncoderConfig()

        self.five_gru = FiveGRU(
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )

        self.fusion = MLPFusion(
            hidden_dim=self.config.fusion_hidden_dim,
        )

        self.pipeline = FiveGRUStatePipeline(
            five_gru=self.five_gru,
            fusion=self.fusion,
        )

        if self.pipeline.hidden_dim != self.config.hidden_dim:
            raise RuntimeError(
                "FiveGRU hidden dimension does not match "
                "StateEncoder configuration."
            )

        if self.pipeline.state_dim != self.config.fusion_hidden_dim:
            raise RuntimeError(
                "Fusion/state dimension does not match "
                "StateEncoder configuration."
            )

    @property
    def state_dim(self) -> int:
        return self.pipeline.state_dim

    @property
    def hidden_dim(self) -> int:
        return self.pipeline.hidden_dim

    def forward(self, inputs: FiveGRUInputBatch) -> Tensor:
        """Return Unified Patient State tensor [batch, state_dim]."""

        state = self.pipeline(inputs)

        value = state.state

        if value.ndim != 2:
            raise RuntimeError(
                "State encoder produced an invalid state rank."
            )

        if value.shape[1] != self.state_dim:
            raise RuntimeError(
                "State encoder produced an invalid state dimension."
            )

        if not torch.isfinite(value).all():
            raise RuntimeError(
                "State encoder produced non-finite values."
            )

        return value


def save_state_encoder(
    model: StateEncoder,
    path: str | Path,
) -> Path:
    """
    Save a reproducible state-encoder checkpoint.

    The checkpoint contains architecture configuration and weights.
    """

    if not isinstance(model, StateEncoder):
        raise TypeError(
            "model must be a StateEncoder."
        )

    destination = Path(path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "checkpoint_type": "t1d_uom_state_encoder",
        "checkpoint_version": 1,
        "config": {
            "hidden_dim": model.config.hidden_dim,
            "fusion_hidden_dim": model.config.fusion_hidden_dim,
            "num_layers": model.config.num_layers,
            "dropout": model.config.dropout,
        },
        "state_dim": model.state_dim,
        "state_dict": model.state_dict(),
    }

    torch.save(payload, destination)

    return destination


def load_state_encoder(
    path: str | Path,
    *,
    map_location: Optional[str | torch.device] = "cpu",
) -> StateEncoder:
    """
    Load a previously saved state encoder.
    """

    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(
            f"State encoder checkpoint does not exist: {source}"
        )

    payload = torch.load(
        source,
        map_location=map_location,
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Invalid state encoder checkpoint."
        )

    if payload.get("checkpoint_type") != "t1d_uom_state_encoder":
        raise ValueError(
            "Checkpoint is not a T1D-UOM state encoder."
        )

    config_data = payload.get("config")

    if not isinstance(config_data, dict):
        raise ValueError(
            "Checkpoint is missing encoder configuration."
        )

    config = StateEncoderConfig(
        hidden_dim=int(
            config_data["hidden_dim"]
        ),
        fusion_hidden_dim=int(
            config_data["fusion_hidden_dim"]
        ),
        num_layers=int(
            config_data["num_layers"]
        ),
        dropout=float(
            config_data["dropout"]
        ),
    )

    model = StateEncoder(config)

    state_dict = payload.get("state_dict")

    if not isinstance(state_dict, dict):
        raise ValueError(
            "Checkpoint is missing state_dict."
        )

    model.load_state_dict(state_dict)

    model.eval()

    return model


def _self_test() -> None:
    """Architecture-only self-test."""

    torch.manual_seed(42)

    config = StateEncoderConfig(
        hidden_dim=64,
        fusion_hidden_dim=64,
        num_layers=1,
        dropout=0.0,
    )

    model = StateEncoder(config)

    batch_size = 2
    sequence_length = 5

    inputs = FiveGRUInputBatch(
        glucose=torch.randn(
            batch_size,
            sequence_length,
            1,
        ),
        insulin=torch.randn(
            batch_size,
            sequence_length,
            2,
        ),
        nutrition=torch.randn(
            batch_size,
            sequence_length,
            24,
        ),
        activity=torch.randn(
            batch_size,
            sequence_length,
            17,
        ),
        sleep=torch.randn(
            batch_size,
            sequence_length,
            6,
        ),
    )

    with torch.no_grad():
        state = model(inputs)

    assert state.shape == (
        batch_size,
        64,
    )

    assert torch.isfinite(state).all()

    print("STATE ENCODER SELF-TEST")
    print("=" * 72)
    print("FiveGRU hidden dimension :", model.hidden_dim)
    print("Unified state dimension  :", model.state_dim)
    print("Output shape             :", tuple(state.shape))
    print("Architecture             : FiveGRU -> MLPFusion")
    print("SELF-TEST                : PASS")


if __name__ == "__main__":
    _self_test()