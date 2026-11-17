"""Checkpoint compatibility checks that CusRL's permissive loader does not enforce."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast


def _compare_state(saved: Any, current: Any, path: str, *, values: bool) -> None:
    """Compare nested state without allowing CusRL's warn-and-continue fallback."""
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        if set(saved) != set(current):
            missing = sorted(set(current) - set(saved), key=str)
            extra = sorted(set(saved) - set(current), key=str)
            raise ValueError(f"Checkpoint {path} keys differ: missing={missing}, unexpected={extra}.")
        for key in current:
            _compare_state(saved[key], current[key], f"{path}.{key}", values=values)
    elif isinstance(saved, (list, tuple)) and isinstance(current, (list, tuple)):
        if len(saved) != len(current):
            raise ValueError(f"Checkpoint {path} length differs: {len(saved)} != {len(current)}.")
        for index, (old, new) in enumerate(zip(saved, current, strict=True)):
            _compare_state(old, new, f"{path}[{index}]", values=values)
    elif hasattr(saved, "shape") and hasattr(current, "shape"):
        old_tensor, new_tensor = cast(Any, saved), cast(Any, current)
        if tuple(old_tensor.shape) != tuple(new_tensor.shape) or old_tensor.dtype != new_tensor.dtype:
            raise ValueError(f"Checkpoint {path} tensor shape/dtype differs: {old_tensor.shape} != {new_tensor.shape}.")
        if values:
            import torch

            if not torch.equal(old_tensor.detach().cpu(), new_tensor.detach().cpu()):
                raise ValueError(f"Checkpoint {path} was not restored exactly.")
    elif type(saved) is not type(current):
        raise ValueError(f"Checkpoint {path} type differs: {type(saved).__name__} != {type(current).__name__}.")
    elif values and saved != current:
        raise ValueError(f"Checkpoint {path} was not restored exactly: {saved!r} != {current!r}.")


def validate_agent_checkpoint_compatibility(saved: Mapping[str, Any], agent: Any) -> None:
    """Require the same modules, hooks and ordered optimizer parameters before loading.

    Adam moments are absent in a freshly built optimizer, so validate them
    against its actual ordered parameters instead of the empty initial state.
    """
    import math

    current = agent.state_dict()
    if set(saved) != set(current):
        raise ValueError(f"Checkpoint agent modules differ: {sorted(saved)} != {sorted(current)}.")
    optimizer_names = set(agent.OPTIMIZERS)
    for name in current:
        if name not in optimizer_names:
            _compare_state(saved[name], current[name], f"agent.{name}", values=False)
            continue
        optimizer = getattr(agent, name)
        old_optimizer = saved[name]
        if not isinstance(old_optimizer, Mapping) or set(old_optimizer) != {"state", "param_groups"}:
            raise ValueError(f"Checkpoint {name} must contain state and param_groups.")
        old_groups = old_optimizer["param_groups"]
        if len(old_groups) != len(optimizer.param_groups):
            raise ValueError(f"Checkpoint {name} parameter group count differs.")
        expected_ids = set()
        for group_index, (old, new) in enumerate(zip(old_groups, optimizer.param_groups, strict=True)):
            # CusRL stores optimizer names before DDP/torch.compile wrappers
            # are stripped from module state_dict. CPU evaluation and a
            # different world size may therefore add/remove only these known
            # wrapper components. Preserve the actual parameter order.
            def canonical_names(names):
                if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
                    raise ValueError(f"Checkpoint {name} group {group_index} has invalid parameter names.")
                result = [".".join(part for part in item.split(".") if part not in {"module", "_orig_mod"})
                          for item in names]
                if len(set(result)) != len(result):
                    raise ValueError(f"Checkpoint {name} group {group_index} parameter names are ambiguous.")
                return result

            if canonical_names(old.get("param_names")) != canonical_names(new.get("param_names")):
                raise ValueError(f"Checkpoint {name} group {group_index} parameter names/order differ.")
            if len(old["params"]) != len(new["params"]) or set(old) != set(new):
                raise ValueError(f"Checkpoint {name} group {group_index} structure differs.")
            if not math.isfinite(float(old["lr"])) or float(old["lr"]) <= 0:
                raise ValueError(f"Checkpoint {name} group {group_index} learning rate is invalid.")
            for parameter_id, parameter in zip(old["params"], new["params"], strict=True):
                if parameter_id in expected_ids:
                    raise ValueError(f"Checkpoint {name} contains duplicate parameter IDs.")
                expected_ids.add(parameter_id)
                moment = old_optimizer["state"].get(parameter_id)
                required = {"step", "exp_avg", "exp_avg_sq"}
                if old.get("amsgrad"):
                    required.add("max_exp_avg_sq")
                if not isinstance(moment, Mapping) or set(moment) != required:
                    raise ValueError(f"Checkpoint {name} parameter {parameter_id} has incomplete Adam state.")
                for key in required - {"step"}:
                    tensor = moment[key]
                    if tuple(getattr(tensor, "shape", ())) != tuple(parameter.shape):
                        raise ValueError(f"Checkpoint {name} parameter {parameter_id} {key} shape differs.")
                    if tensor.dtype != parameter.dtype:
                        raise ValueError(f"Checkpoint {name} parameter {parameter_id} {key} dtype differs.")
                step = moment["step"]
                if getattr(step, "numel", lambda: 0)() != 1 or not math.isfinite(float(step)) or float(step) <= 0:
                    raise ValueError(f"Checkpoint {name} parameter {parameter_id} has invalid optimizer step.")
        if set(old_optimizer["state"]) != expected_ids:
            raise ValueError(f"Checkpoint {name} optimizer states do not match all parameters.")


def verify_agent_checkpoint_restored(saved: Mapping[str, Any], agent: Any) -> None:
    """Verify actual restored values, including moments and RMS extra state."""
    _compare_state(saved, agent.state_dict(), "agent", values=True)


def load_agent_checkpoint_strict(saved: Mapping[str, Any], agent: Any) -> None:
    """Load a complete agent or fail before any rollout starts."""
    validate_agent_checkpoint_compatibility(saved, agent)
    agent.load_state_dict(saved)
    verify_agent_checkpoint_restored(saved, agent)


def get_amp_checkpoint_dim(checkpoint: Mapping[str, Any]) -> int | None:
    """Return the AMP discriminator input dimension stored in a CusRL checkpoint."""
    agent_state = checkpoint.get("agent")
    if not isinstance(agent_state, Mapping):
        return None
    hook_state = agent_state.get("hook")
    if not isinstance(hook_state, Mapping):
        return None

    dimensions: set[int] = set()
    for state in hook_state.values():
        if not isinstance(state, Mapping):
            continue
        discriminator = state.get("discriminator")
        transition_rms = state.get("transition_rms")
        if not isinstance(discriminator, Mapping) or not isinstance(transition_rms, Mapping):
            continue

        first_weight = discriminator.get("layers.0.weight")
        mean = transition_rms.get("mean")
        if first_weight is not None and hasattr(first_weight, "shape") and len(first_weight.shape) == 2:
            dimensions.add(int(first_weight.shape[1]))
        if mean is not None and hasattr(mean, "shape") and len(mean.shape) == 1:
            dimensions.add(int(mean.shape[0]))

    if len(dimensions) > 1:
        raise ValueError(f"Checkpoint contains inconsistent AMP dimensions: {sorted(dimensions)}.")
    return next(iter(dimensions), None)


def validate_amp_checkpoint_compatibility(
    checkpoint: Mapping[str, Any],
    expected_dim: int | None,
    *,
    checkpoint_path: str,
    task_name: str,
) -> None:
    """Fail before Play when the task and checkpoint use different AMP structures."""
    checkpoint_dim = get_amp_checkpoint_dim(checkpoint)
    if checkpoint_dim == expected_dim:
        return

    if checkpoint_dim is None:
        detail = "the checkpoint has no AMP discriminator state"
    elif expected_dim is None:
        detail = f"the checkpoint expects AMP dimension {checkpoint_dim}, but the task has no AMP observation"
    else:
        detail = f"the checkpoint expects AMP dimension {checkpoint_dim}, but the task builds dimension {expected_dim}"

    raise ValueError(
        "AMP checkpoint/task mismatch: "
        f"{detail}. task={task_name!r}, checkpoint={checkpoint_path!r}. "
        "Use the matching KeepBase/MaskBase task and the same command-conditioned code revision used for training."
    )
