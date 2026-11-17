"""List stage-one and impact registrations without launching Isaac Sim."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "t800_lab"))
import gymnasium as gym  # noqa: E402
import t800_lab.tasks  # noqa: E402,F401

for task in ("T800_base", "T800_base-Play", "T800_base-Impact", "T800_base-Impact-Play"):
    print(f"{task}: {gym.spec(task).kwargs['env_cfg_entry_point']}")
