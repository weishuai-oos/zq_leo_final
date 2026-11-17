"""CPU contract tests for the T800 dynamics-randomization helpers."""

from types import SimpleNamespace
from pathlib import Path
import sys

import torch
import pytest

SOURCE_ROOT = str(Path(__file__).resolve().parents[1] / "source/t800_lab")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from t800_lab.tasks.t800_dynamics_dr import mdp  # noqa: E402 -- load the local source tree


def test_offset_sampling_keeps_head_slots_zero_and_is_paired():
    offsets = mdp.sample_joint_offsets(8, "cpu", generator=torch.Generator().manual_seed(4))
    assert offsets.shape == (8, 25)
    assert torch.equal(offsets[:, 23:], torch.zeros(8, 2))
    assert offsets[:, :23].abs().max() <= 0.01
    physical = torch.zeros(8, 4)
    mapping = torch.tensor([4, 1, 23, 0])
    result = mdp.add_physical_target_offsets(physical, mapping, offsets)
    assert torch.allclose(result[:, 2], torch.zeros(8))
    assert torch.allclose(result[:, 0], offsets[:, 4])
    assert torch.allclose(result[:, 1], offsets[:, 1])
    assert torch.allclose(result[:, 3], offsets[:, 0])


def test_observation_compensation_does_not_touch_head_or_velocity():
    offsets = torch.zeros(2, 25)
    offsets[:, :23] = 0.01
    q = torch.ones(2, 25)
    compensated = mdp.subtract_joint_offsets(q, offsets)
    assert torch.allclose(compensated[:, :23], torch.full((2, 23), 0.99))
    assert torch.allclose(compensated[:, 23:], torch.ones(2, 2))


def test_amp_offset_projection_uses_amp_joint_order_and_not_first_21_slots():
    offsets = torch.zeros(1, 25)
    offsets[0, 0] = 0.01
    offsets[0, 4] = 0.02  # ankle pitch is intentionally absent from AMP
    values = torch.ones(1, 21)
    compensated = mdp.subtract_amp_joint_offsets(values, offsets)
    assert torch.allclose(compensated[0, 0], torch.tensor(0.99))
    assert torch.allclose(compensated[0, 4], torch.tensor(1.0))
    assert torch.allclose(compensated[0, -2:], torch.ones(2))


class _PhysX:
    def __init__(self):
        self.masses = torch.tensor([[2.0, 3.0], [2.0, 3.0]])
        self.inertias = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]] * 2)
        self.coms = torch.zeros(2, 2, 3)

    def get_masses(self):
        return self.masses

    def set_masses(self, values, env_ids):
        self.masses[env_ids] = values[env_ids]

    def get_inertias(self):
        return self.inertias

    def set_inertias(self, values, env_ids):
        self.inertias[env_ids] = values[env_ids]

    def get_coms(self):
        return self.coms

    def set_coms(self, values, env_ids):
        self.coms[env_ids] = values[env_ids]


def test_mass_inertia_com_are_written_from_nominal_without_accumulation():
    view = _PhysX()
    asset = SimpleNamespace(
        body_names=["LINK_BASE", "LINK_TORSO_YAW"],
        num_bodies=2,
        root_physx_view=view,
        data=SimpleNamespace(
            default_mass=torch.tensor([[2.0, 3.0], [2.0, 3.0]]),
            default_inertia=torch.tensor([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]] * 2),
            default_com_pos_b=torch.zeros(2, 2, 3),
        ),
    )
    env = SimpleNamespace(scene=SimpleNamespace(num_envs=2), extras={},)
    env.scene.__getitem__ = lambda name: asset
    # SimpleNamespace cannot provide special-method lookup; use a tiny scene.
    env.scene = type("Scene", (), {"num_envs": 2, "__getitem__": lambda self, name: asset})()
    cfg = SimpleNamespace(name="robot", body_ids=slice(None))
    torch.manual_seed(7)
    mdp.randomize_dynamics(env, torch.tensor([0, 1]), cfg)
    first = view.masses.clone(), view.coms.clone()
    mdp.randomize_dynamics(env, torch.tensor([0, 1]), cfg)
    # Every call is relative to defaults, not the previous randomization.
    assert torch.all(view.masses > 0)
    assert torch.all((view.masses[:, 0] >= 1.8) & (view.masses[:, 0] <= 2.2))
    assert not torch.equal(first[0], view.masses)
    assert env.extras["log"]["DR/mass_scale_min"] > 0


def dynamics_scene(enabled=True):
    view = _PhysX()
    asset = SimpleNamespace(body_names=["LINK_BASE", "LINK_TORSO_YAW"], num_bodies=2, root_physx_view=view)
    scene = type("Scene", (), {"num_envs": 2, "__getitem__": lambda self, name: asset})()
    env = SimpleNamespace(scene=scene, extras={}, cfg=SimpleNamespace(dr_enabled=enabled))
    return env, view, SimpleNamespace(name="robot", body_ids=slice(None))


def test_torso_mass_includes_both_scale_and_extra_mass_and_inertia_matches():
    env, view, cfg = dynamics_scene()
    original_inertia = view.inertias.clone()
    for _ in range(2):
        mdp.randomize_dynamics(env, None, cfg, mass_scale_range=(1.1, 1.1),
                              torso_mass_delta_range=(2.0, 2.0), torso_com_offset_range=(0.0, 0.0, 0.0))
        torch.testing.assert_close(view.masses, torch.tensor([[2.2, 5.3], [2.2, 5.3]]))
        torch.testing.assert_close(view.inertias[:, 1], original_inertia[:, 1] * (5.3 / 3.0))


@pytest.mark.parametrize(("setter", "label"), [
    ("set_masses", "mass"), ("set_inertias", "inertia"), ("set_coms", "torso COM"),
])
def test_failed_physics_write_is_detected_by_readback(setter, label):
    env, view, cfg = dynamics_scene()
    setattr(view, setter, lambda values, env_ids: None)
    with pytest.raises(RuntimeError, match=label):
        mdp.randomize_dynamics(env, None, cfg, mass_scale_range=(1.1, 1.1))


def test_play_keeps_nominal_physics_and_zero_joint_offsets():
    env, view, cfg = dynamics_scene(enabled=False)
    mass, inertia, com = view.masses.clone(), view.inertias.clone(), view.coms.clone()
    mdp.randomize_dynamics(env, None, cfg)
    assert torch.equal(view.masses, mass) and torch.equal(view.inertias, inertia) and torch.equal(view.coms, com)
    assert torch.equal(mdp.sample_joint_offsets(2, "cpu", enabled=False), torch.zeros(2, 25))
