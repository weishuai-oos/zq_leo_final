"""Keep evidence of real pulses when a smoke environment autoresets."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_check_config(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(
        "_impact_smoke_checks", Path(__file__).resolve().parents[1] / "scripts/check_config.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_report_serializes_metadata_without_changing_runtime_state(monkeypatch):
    module = load_check_config(monkeypatch)
    rng_state = torch.Generator().get_state()
    state = {
        "impact": {"generator_state": rng_state},
        "finetune": {"stage_origin_iteration": 140000, "completed_updates": 0},
        "impact_v2": {"source_v2": {"completed_updates": 40000}},
    }
    adapter = SimpleNamespace(state_dict=lambda: state)

    report = module.checkpoint_metadata_for_report(adapter)

    assert json.loads(json.dumps(report)) == {
        "finetune": state["finetune"], "impact_v2": state["impact_v2"],
    }
    assert adapter.state_dict()["impact"]["generator_state"] is rng_state
    assert torch.equal(rng_state, torch.Generator().get_state())


def test_impulse_evidence_survives_partial_and_repeated_autoresets(monkeypatch):
    module = load_check_config(monkeypatch)
    impulse = torch.zeros(2)

    def reset(indices):
        impulse[indices] = 0

    raw = SimpleNamespace(impact_runtime=SimpleNamespace(_stats={"impulse_ns": impulse}), _reset_idx=reset)
    with module.capture_impact_impulse(SimpleNamespace(unwrapped=raw)) as evidence:
        impulse[:] = torch.tensor([10.0, 20.0])
        raw._reset_idx([0])
        impulse += torch.tensor([3.0, 4.0])
        raw._reset_idx([0, 1])
        impulse[:] = torch.tensor([5.0, 6.0])
    assert evidence["impulse_ns"] == 48.0
    assert raw._reset_idx is reset


def test_impulse_capture_restores_reset_on_error(monkeypatch):
    module = load_check_config(monkeypatch)
    def reset(indices):
        pass
    raw = SimpleNamespace(impact_runtime=SimpleNamespace(_stats={"impulse_ns": torch.zeros(2)}), _reset_idx=reset)
    with pytest.raises(RuntimeError, match="fixture failure"):
        with module.capture_impact_impulse(SimpleNamespace(unwrapped=raw)):
            raise RuntimeError("fixture failure")
    assert raw._reset_idx is reset
