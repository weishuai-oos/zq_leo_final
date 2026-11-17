"""CPU geometry regressions retained from the historical Plus terrain tests."""

from __future__ import annotations

import ast

from pathlib import Path

from types import SimpleNamespace

import numpy as np

import pytest

import trimesh

ROOT = Path(__file__).resolve().parents[1]

MESH_TERRAINS = ROOT / "source/t800_lab/t800_lab/terrains/trimesh/mesh_terrains.py"

PERLIN_UTILS = ROOT / "source/t800_lab/t800_lab/utils/perlin.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _load_pure_terrain_functions():
    selected_names = {
        "low_obstacle_direction_positions",
        "_weighted_shape_schedule",
        "_ground_box",
        "_height_surface_mesh",
        "_smooth_random_surface",
        "_surface_origin",
        "_horizontal_obstacle_mesh",
        "structured_low_obstacle_terrain",
        "_plus_surface",
        "_plus_perlin",
        "_plus_obstacle_width",
        "_plus_footprint_axes",
        "_plus_footprint_corners",
        "_plus_footprints_overlap",
        "_plus_candidate_clear",
        "structured_fight_mat_plus_terrain",
    }
    selected_nodes = []
    for node in _tree(MESH_TERRAINS).body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            selected_nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            (
                isinstance(target, ast.Name) and target.id == "CARDINAL_AND_DIAGONAL_DIRECTIONS"
                for target in node.targets
            )
        ):
            selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in selected_names:
            selected_nodes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=selected_nodes, type_ignores=[]))
    namespace = {"np": np, "trimesh": trimesh, "generate_wall": lambda function: function}
    perlin_nodes = [
        node
        for node in _tree(PERLIN_UTILS).body
        if isinstance(node, ast.FunctionDef) and node.name in {"generate_perlin_noise_2d", "generate_fractal_noise_2d"}
    ]
    perlin_module = ast.fix_missing_locations(ast.Module(body=perlin_nodes, type_ignores=[]))
    exec(compile(perlin_module, PERLIN_UTILS, "exec"), namespace)
    exec(compile(module, MESH_TERRAINS, "exec"), namespace)
    return namespace


def _plus_cfg(terrain_kind: str) -> SimpleNamespace:
    return SimpleNamespace(
        size=(8.0, 8.0),
        surface_resolution=0.1,
        seed=43,
        terrain_kind=terrain_kind,
        perlin_noise_range=(0.0, 0.012),
        rough_height_range=(-0.012, 0.03),
        wave_amplitude_range=(0.0, 0.08),
        wave_count=5.0,
        box_count_range=(4, 18),
        box_height_range=(0.015, 0.08),
        box_footprint_range=(0.25, 0.5),
        obstacle_count_range=(0, 20),
        obstacle_exposure_range=(0.03, 0.12),
        obstacle_exposure_mode=0.1,
        obstacle_shape_weights=(0.65, 0.2, 0.15),
        obstacle_length_range=(0.8, 1.2),
        cylinder_radius_range=(0.12, 0.22),
        hex_radius_range=(0.12, 0.22),
        box_width_range=(0.2, 0.38),
        spawn_clearance=1.0,
        inner_ring_radius=1.55,
        second_ring_radius=2.4,
        outer_lane_radius_range=(2.8, 3.25),
    )


def test_plus_terrain_families_increase_severity_without_changing_family():
    functions = _load_pure_terrain_functions()
    generate = functions["structured_fight_mat_plus_terrain"]
    for terrain_kind in ("perlin", "rough", "wave"):
        easy_meshes, _ = generate(0.0, _plus_cfg(terrain_kind))
        hard_meshes, _ = generate(1.0, _plus_cfg(terrain_kind))
        easy_height = np.max(np.abs(easy_meshes[0].vertices[:, 2]))
        hard_height = np.max(np.abs(hard_meshes[0].vertices[:, 2]))
        assert easy_height == pytest.approx(0.0)
        assert hard_height > easy_height
    wave_meshes, _ = generate(1.0, _plus_cfg("wave"))
    assert wave_meshes[0].bounds[:, 2] == pytest.approx((-0.08, 0.08))
    box_counts = [len(generate(difficulty, _plus_cfg("boxes"))[0]) - 1 for difficulty in (0.0, 0.5, 1.0)]
    assert box_counts == [4, 11, 18]
    horizontal_counts = [
        len(generate(difficulty, _plus_cfg("horizontal"))[0]) - 1 for difficulty in np.linspace(0.0, 1.0, 11)
    ]
    assert horizontal_counts == sorted(horizontal_counts)
    assert horizontal_counts[0] == 0
    assert horizontal_counts[-1] == 20


def test_plus_horizontal_obstacles_respect_spawn_boundary_and_overlap_contracts():
    functions = _load_pure_terrain_functions()
    generate = functions["structured_fight_mat_plus_terrain"]
    cfg = _plus_cfg("horizontal")
    meshes, origin = generate(1.0, cfg)
    assert len(meshes) - 1 == 20
    assert origin == pytest.approx((4.0, 4.0, 0.0))
    for obstacle in meshes[1:]:
        assert np.all(obstacle.bounds[:, :2] >= -1e-09)
        assert np.all(obstacle.bounds[:, :2] <= 8.0 + 1e-09)
        assert obstacle.bounds[1, 2] <= cfg.obstacle_exposure_range[1] + 1e-09
    center = np.array((4.0, 4.0))
    first = (np.array((5.55, 4.0)), np.pi / 2.0, 1.0, 0.4)
    overlapping = (np.array((5.55, 4.1)), np.pi / 2.0, 1.0, 0.4)
    clear = (np.array((2.45, 4.0)), np.pi / 2.0, 1.0, 0.4)
    intruding = (np.array((5.0, 4.0)), np.pi / 2.0, 1.0, 0.4)
    outside = (np.array((7.9, 4.0)), 0.0, 1.0, 0.4)
    assert functions["_plus_candidate_clear"](first, center, cfg, [])
    assert not functions["_plus_candidate_clear"](overlapping, center, cfg, [first])
    assert functions["_plus_candidate_clear"](clear, center, cfg, [first])
    assert not functions["_plus_candidate_clear"](intruding, center, cfg, [])
    assert not functions["_plus_candidate_clear"](outside, center, cfg, [])
