"""Tests for the LSTM sequence-model classifier and Transformer stub (Phase 5 U2).

Pins:
* Forward pass returns ``(batch,)`` logits regardless of ``bidirectional``.
* Same seed -> same outputs (CLAUDE.md rule 10 reproducibility).
* Different seeds -> different outputs (model is actually random-initialized,
  not a constant).
* Length-1 sequences do not crash ``outputs[:, -1, :]``.
* The Transformer stub raises with a clear pointer to the plan -- no silent
  fallback to a half-implemented model.
* :class:`SequenceResult` is frozen (matches :class:`BaselineResult`).
* State-dict round trip preserves predictions, so U3 can save/load
  checkpoints without divergence.

All tests run on CPU. No MPS or CUDA assumptions, so the suite is portable
to CI runners that lack a GPU.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from icumodelstream.torch_models import (
    LSTMBaseline,
    SequenceResult,
    TransformerBaseline,
)


def test_forward_shape_and_dtype_happy_path() -> None:
    """Forward pass returns (batch,) float32 logits and is not all zeros."""
    torch.manual_seed(42)
    model = LSTMBaseline(input_dim=26, hidden_dim=64, n_layers=2)
    model.eval()

    x = torch.randn(8, 24, 26)
    logits = model(x)

    assert logits.shape == (8,)
    assert logits.dtype == torch.float32
    # If we accidentally short-circuited the linear head, logits would be a
    # constant; a healthy random init produces a non-degenerate spread.
    assert not torch.allclose(logits, torch.zeros_like(logits))


def test_same_seed_same_outputs() -> None:
    """Two LSTMBaselines built under the same seed produce identical logits."""
    x = torch.randn(4, 12, 10)

    torch.manual_seed(42)
    model_a = LSTMBaseline(input_dim=10, hidden_dim=32, n_layers=2)
    model_a.eval()

    torch.manual_seed(42)
    model_b = LSTMBaseline(input_dim=10, hidden_dim=32, n_layers=2)
    model_b.eval()

    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)

    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_different_seeds_give_different_outputs() -> None:
    """Different seeds yield meaningfully different (non-identical) logits."""
    x = torch.randn(4, 12, 10)

    torch.manual_seed(0)
    model_a = LSTMBaseline(input_dim=10, hidden_dim=32, n_layers=2)
    model_a.eval()

    torch.manual_seed(123)
    model_b = LSTMBaseline(input_dim=10, hidden_dim=32, n_layers=2)
    model_b.eval()

    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)

    # Tight bound on "really different" -- random init shouldn't collide.
    max_diff = (out_a - out_b).abs().max().item()
    assert max_diff > 1e-3, (
        f"Different seeds produced near-identical outputs (max diff {max_diff}); "
        "weight init may not be seeded as expected."
    )


def test_bidirectional_doubles_classifier_input_features() -> None:
    """Bidirectional LSTM head must consume 2 * hidden_dim features."""
    torch.manual_seed(42)
    model = LSTMBaseline(input_dim=10, hidden_dim=32, n_layers=2, bidirectional=True)
    model.eval()

    # Linear head is sized for both directions.
    assert model.classifier.in_features == 64
    assert model.classifier.out_features == 1

    x = torch.randn(3, 8, 10)
    logits = model(x)
    assert logits.shape == (3,)


def test_unidirectional_classifier_input_features() -> None:
    """Unidirectional LSTM head consumes exactly hidden_dim features."""
    torch.manual_seed(42)
    model = LSTMBaseline(input_dim=10, hidden_dim=32, n_layers=2)
    assert model.classifier.in_features == 32
    assert model.classifier.out_features == 1


def test_single_timestep_input() -> None:
    """Length-1 sequences must not crash ``outputs[:, -1, :]``."""
    torch.manual_seed(42)
    model = LSTMBaseline(input_dim=10, hidden_dim=16, n_layers=2)
    model.eval()

    x = torch.randn(4, 1, 10)
    logits = model(x)
    assert logits.shape == (4,)


def test_transformer_stub_raises_immediately() -> None:
    """TransformerBaseline must raise at construction, with a pointer to the plan."""
    with pytest.raises(NotImplementedError, match="intentionally deferred"):
        TransformerBaseline(input_dim=10)


def test_sequence_result_is_frozen_dataclass() -> None:
    """SequenceResult holds dummy payloads and is immutable like BaselineResult."""
    result = SequenceResult(
        model_name="lstm",
        y_true=np.array([0, 1, 0, 1]),
        y_pred_proba=np.array([0.1, 0.9, 0.2, 0.8]),
        metrics={
            "auroc": 1.0,
            "auprc": 1.0,
            "brier_score": 0.025,
            "prevalence": 0.5,
            "calibration_intercept": 0.0,
            "calibration_slope": 1.0,
        },
        calibration_table=pl.DataFrame(
            {
                "bin": [0, 9],
                "mean_pred": [0.15, 0.85],
                "mean_actual": [0.0, 1.0],
                "count": [2, 2],
            }
        ),
        epochs_trained=5,
        early_stopped_at_epoch=None,
    )

    assert result.model_name == "lstm"
    assert result.epochs_trained == 5
    assert result.early_stopped_at_epoch is None

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.model_name = "x"  # type: ignore[misc]


def test_state_dict_round_trip(tmp_path: Path) -> None:
    """Saving and reloading state_dict preserves forward outputs to <1e-6."""
    torch.manual_seed(42)
    model_a = LSTMBaseline(input_dim=10, hidden_dim=16, n_layers=2)
    model_a.eval()

    x = torch.randn(4, 8, 10)
    with torch.no_grad():
        out_a = model_a(x)

    ckpt_path = tmp_path / "lstm_baseline.pt"
    torch.save(model_a.state_dict(), ckpt_path)

    # Re-create a *differently* seeded model so any failure to load weights
    # would show up as a divergent forward pass.
    torch.manual_seed(999)
    model_b = LSTMBaseline(input_dim=10, hidden_dim=16, n_layers=2)
    model_b.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model_b.eval()

    with torch.no_grad():
        out_b = model_b(x)

    assert torch.allclose(out_a, out_b, atol=1e-6)
