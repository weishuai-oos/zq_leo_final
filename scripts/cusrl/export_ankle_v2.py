#!/usr/bin/env python3
"""Export the final AnkleCombinedV2 actor without starting Isaac Sim.

This intentionally exports one checkpoint at a time.  The generated ONNX graph
contains the CusRL observation RMS and recurrent actor; physical environment
configuration is not constructed or re-computed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile


TASK = "T800_base-AnkleCombinedV2"
EXPECTED_ITERATION = 140000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(value) -> list[int]:
    return [int(dim) for dim in value.shape]


def _validate_checkpoint(checkpoint: dict, expected_iteration: int) -> dict:
    if checkpoint.get("iteration") != expected_iteration:
        raise ValueError(
            f"expected checkpoint iteration {expected_iteration}, got {checkpoint.get('iteration')!r}"
        )
    environment = checkpoint.get("environment", {})
    metadata = environment.get("ankle_finetune")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint is missing environment.ankle_finetune metadata")
    if metadata.get("version") != 1 or metadata.get("task") != TASK:
        raise ValueError(f"unsupported ankle metadata: version={metadata.get('version')!r}, task={metadata.get('task')!r}")
    origin = metadata.get("origin_iteration")
    updates = metadata.get("completed_updates")
    if type(origin) is not int or type(updates) is not int or origin < 0 or updates < 0:
        raise ValueError("checkpoint is missing valid origin_iteration/completed_updates")
    if origin + updates != expected_iteration:
        raise ValueError(f"checkpoint metadata iteration mismatch: {origin}+{updates} != {expected_iteration}")
    actor_state = checkpoint.get("agent", {}).get("actor")
    rms_state = checkpoint.get("agent", {}).get("hook", {}).get("observation_normalization", {}).get("observation_rms")
    if not isinstance(actor_state, dict) or not actor_state:
        raise ValueError("checkpoint is missing agent.actor state")
    if not isinstance(rms_state, dict) or not rms_state:
        raise ValueError("checkpoint is missing observation normalization RMS state")
    return metadata


def _export(checkpoint_path: Path, output_dir: Path, expected_iteration: int) -> None:
    import torch
    import cusrl
    from cusrl.module import GraphBuilder, RunningMeanStd

    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = _validate_checkpoint(checkpoint, expected_iteration)

    actor = cusrl.Actor.Factory(
        backbone_factory=cusrl.Lstm.Factory(hidden_size=256, num_layers=2),
        distribution_factory=cusrl.NormalDist.Factory(),
    )(84, 25).cpu().eval()
    actor.load_state_dict(checkpoint["agent"]["actor"], strict=True)

    rms = RunningMeanStd(84).cpu().eval()
    rms.load_state_dict(
        checkpoint["agent"]["hook"]["observation_normalization"]["observation_rms"],
        strict=True,
    )
    if not all(torch.isfinite(value).all() for value in (*actor.parameters(), rms.mean, rms.std)):
        raise ValueError("actor/RMS contains non-finite values")
    if not torch.all(rms.std > 0):
        raise ValueError("RMS standard deviation must be positive")

    graph = GraphBuilder(graph_name="actor")
    graph.add_node(
        rms,
        module_name="observation_rms",
        input_names={"input": "observation"},
        output_names="observation",
        expose_outputs=False,
    )
    graph.add_node(
        actor,
        module_name="actor",
        input_names={"observation": "observation", "memory": "memory_in"},
        output_names=["action", "memory_out"],
        extra_kwargs={"forward_type": "act_deterministic"},
        info={"observation_dim": 84, "action_dim": 25, "is_recurrent": True},
        expose_outputs=True,
    )

    observation = torch.zeros(1, 1, 84)
    # Keep regular tensors for torch.onnx.export; inference tensors cannot be
    # saved by the LSTM exporter even when the graph itself is eval-only.
    with torch.no_grad():
        _, memory = actor(observation)
        actor.reset_memory(memory)
        inputs = {"observation": observation, "memory_in": memory}
        action, memory_out = actor(observation, memory=memory, forward_type="act_deterministic")
    if _shape(action) != [1, 1, 25] or len(memory_out) != 2 or any(_shape(item) != [2, 1, 256] for item in memory_out):
        raise ValueError(f"unexpected actor IO shapes: action={_shape(action)}, memory={[ _shape(item) for item in memory_out ]}")

    graph.export_onnx(
        inputs,
        output_dir,
        optimize=False,
        dynamo=False,
        verbose=False,
        opset_version=17,
    )

    import onnx

    exported = onnx.load(str(output_dir / "actor.onnx"))
    def io_shapes(values):
        return {value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim] for value in values}

    manifest = {
        "version": 1,
        "task": TASK,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_iteration": expected_iteration,
        "ankle_finetune": metadata,
        "software": {name: importlib.metadata.version(name) for name in ("torch", "cusrl", "onnx")},
        "files": {name: _sha256(output_dir / name) for name in ("actor.onnx", "actor.yml")},
        "export": {
            "device": "cpu",
            "isaac_sim_started": False,
            "environment_config_recomputed": False,
            "observation_dim": 84,
            "action_dim": 25,
            "observation_rms_clamp": 10.0,
            "input_shapes": io_shapes(exported.graph.input),
            "output_shapes": io_shapes(exported.graph.output),
            "opset_version": 17,
        },
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-iteration", type=int, default=EXPECTED_ITERATION)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty; refusing to overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        _export(checkpoint_path, temporary, args.expected_iteration)
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Export completed: {output_dir}")


if __name__ == "__main__":
    main()
