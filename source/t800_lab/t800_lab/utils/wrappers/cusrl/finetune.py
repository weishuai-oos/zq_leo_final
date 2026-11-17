"""Learning-rate caps shared by continuation stages, without adding PPO hooks."""

import math


def cap_learning_rate(agent, caps: list[float]) -> None:
    scheduler = next((hook for hook in agent.hook if hook.name == "adaptive_lr_schedule"), None)
    if scheduler is None or not hasattr(scheduler, "_lr_scale"):
        raise ValueError("Continuation requires the original adaptive_lr_schedule hook.")
    ratios = []
    for group, cap in zip(agent.optimizer.param_groups, caps, strict=True):
        old_lr = float(group["lr"])
        if not math.isfinite(old_lr) or old_lr <= 0 or not math.isfinite(cap) or cap <= 0:
            raise ValueError("Continuation has an invalid optimizer learning rate/cap.")
        ratio = min(old_lr, cap) / old_lr
        group["lr"] = min(old_lr, cap)
        if scheduler.scale_all_params or any(name.startswith("actor.") for name in group["param_names"]):
            ratios.append(ratio)
    if not ratios or any(not math.isclose(ratio, ratios[0], rel_tol=1e-12) for ratio in ratios):
        raise ValueError("Continuation requires a consistent LR scale for scheduler-controlled groups.")
    scheduler._lr_scale *= ratios[0]
