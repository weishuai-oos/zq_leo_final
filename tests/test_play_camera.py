"""Check startup framing and manual camera ownership without starting Isaac."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from cusrl.template.player import PlayerHook


ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / "scripts/cusrl/play.py").read_text())
nodes = [node for node in tree.body if getattr(node, "name", None) in {
    "_configure_play_viewer", "InitialFreeCameraHook",
}]
namespace = {"PlayerHook": PlayerHook}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "play_camera_helpers", "exec"), namespace)
configure = namespace["_configure_play_viewer"]
CameraHook = namespace["InitialFreeCameraHook"]


def viewer_config():
    return SimpleNamespace(viewer=SimpleNamespace(
        eye=(2.0, 2.0, 0.5), lookat=(0.0, 0.0, 0.0),
        env_index=0, origin_type="asset_root", asset_name="robot", body_name=None,
    ))


@pytest.mark.parametrize("free_camera", [False, True])
def test_selected_environment_and_tracking_mode(free_camera):
    cfg = viewer_config()
    configure(cfg, free_camera, 1)
    assert cfg.viewer.env_index == 1
    assert cfg.viewer.origin_type == ("world" if free_camera else "asset_root")
    assert cfg.viewer.asset_name == (None if free_camera else "robot")


def test_frame_live_robot_once_then_preserve_manual_view_across_steps_and_resets():
    cfg = viewer_config()
    configure(cfg, True, 1)
    data = SimpleNamespace(root_pos_w=torch.zeros(2, 3))
    controller = SimpleNamespace(update_view_location=Mock())
    env = SimpleNamespace(
        cfg=cfg, scene={"robot": SimpleNamespace(data=data)}, viewport_camera_controller=controller,
    )
    hook = CameraHook(env, 1)
    # The Player startup reset may relocate the robot after hook construction.
    data.root_pos_w[1] = torch.tensor([70.0, -12.0, 0.9])
    hook.step(0, {}, {})
    kwargs = controller.update_view_location.call_args.kwargs
    assert kwargs["eye"] == pytest.approx((72.0, -10.0, 1.4))
    assert kwargs["lookat"] == pytest.approx((70.0, -12.0, 0.9))
    # A manually moved camera must survive walking and subsequent episode resets.
    data.root_pos_w[1] += 10.0
    hook.step(1, {}, {})
    hook.reset([1])
    hook.step(2, {}, {})
    controller.update_view_location.assert_called_once()
