from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import trimesh
from t800_lab.utils.perlin import generate_fractal_noise_2d
from .utils import generate_wall

if TYPE_CHECKING:
    from . import mesh_terrains_cfg

CARDINAL_AND_DIAGONAL_DIRECTIONS = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (np.sqrt(0.5), np.sqrt(0.5)),
    (-np.sqrt(0.5), np.sqrt(0.5)),
    (np.sqrt(0.5), -np.sqrt(0.5)),
    (-np.sqrt(0.5), -np.sqrt(0.5)),
)


def _weighted_shape_schedule(obstacle_count: int, shape_weights: np.ndarray, rng: np.random.Generator) -> list[str]:
    """Create a shuffled schedule whose realized mix closely matches the configured weights."""
    expected_counts = shape_weights * obstacle_count
    shape_counts = np.floor(expected_counts).astype(int)
    remainder = obstacle_count - int(shape_counts.sum())
    if remainder:
        fractional_order = np.argsort(-(expected_counts - shape_counts))
        shape_counts[fractional_order[:remainder]] += 1
    shape_names = ("cylinder", "hexagonal_prism", "box")
    schedule = [name for name, count in zip(shape_names, shape_counts, strict=True) for _ in range(count)]
    rng.shuffle(schedule)
    return schedule


def _ground_box(size: tuple[float, float]) -> trimesh.Trimesh:
    """Create a thin, flat collision floor whose top surface is z=0."""
    return trimesh.creation.box(
        extents=(size[0], size[1], 0.02),
        transform=trimesh.transformations.translation_matrix((size[0] / 2.0, size[1] / 2.0, -0.01)),
    )


def _height_surface_mesh(
    size: tuple[float, float], resolution: float, heights: np.ndarray
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """Convert a float height grid into a deterministic triangle surface."""
    if resolution <= 0.0:
        raise ValueError("surface_resolution must be positive")
    x_coordinates = np.linspace(0.0, size[0], heights.shape[0])
    y_coordinates = np.linspace(0.0, size[1], heights.shape[1])
    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates, indexing="ij")
    vertices = np.column_stack((grid_x.ravel(), grid_y.ravel(), heights.ravel()))
    vertex_ids = np.arange(heights.size).reshape(heights.shape)
    lower_left = vertex_ids[:-1, :-1].ravel()
    lower_right = vertex_ids[1:, :-1].ravel()
    upper_left = vertex_ids[:-1, 1:].ravel()
    upper_right = vertex_ids[1:, 1:].ravel()
    faces = np.vstack(
        (
            np.column_stack((lower_left, lower_right, upper_right)),
            np.column_stack((lower_left, upper_right, upper_left)),
        )
    )
    return (
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
        grid_x,
        grid_y,
    )


def _smooth_random_surface(
    size: tuple[float, float],
    resolution: float,
    height_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Generate smooth random roughness without stairs, pits, or sharp seams."""
    sample_count_x = int(np.ceil(size[0] / resolution)) + 1
    sample_count_y = int(np.ceil(size[1] / resolution)) + 1
    coarse_count_x = max(3, int(np.ceil(size[0] / 0.5)) + 1)
    coarse_count_y = max(3, int(np.ceil(size[1] / 0.5)) + 1)
    coarse = rng.uniform(*height_range, size=(coarse_count_x, coarse_count_y))
    coarse_x = np.linspace(0.0, size[0], coarse_count_x)
    coarse_y = np.linspace(0.0, size[1], coarse_count_y)
    fine_x = np.linspace(0.0, size[0], sample_count_x)
    fine_y = np.linspace(0.0, size[1], sample_count_y)
    interpolated_x = np.column_stack(
        [np.interp(fine_x, coarse_x, coarse[:, column]) for column in range(coarse_count_y)]
    )
    heights = np.vstack([np.interp(fine_y, coarse_y, interpolated_x[row]) for row in range(sample_count_x)])
    mesh, grid_x, grid_y = _height_surface_mesh(size, resolution, heights)
    edge_distance = np.minimum.reduce((grid_x, size[0] - grid_x, grid_y, size[1] - grid_y))
    edge_fade = np.clip(edge_distance / 0.4, 0.0, 1.0)
    mesh.vertices[:, 2] *= edge_fade.ravel()
    return mesh, mesh.vertices[:, 2]


def _surface_origin(size: tuple[float, float], vertices: np.ndarray) -> np.ndarray:
    """Use the highest point in the central 2 m square as reset origin."""
    center = np.asarray(size, dtype=float) / 2.0
    central = (np.abs(vertices[:, 0] - center[0]) <= 1.0) & (np.abs(vertices[:, 1] - center[1]) <= 1.0)
    center_height = float(np.max(vertices[central, 2])) if np.any(central) else 0.0
    return np.array([center[0], center[1], center_height])


def _horizontal_obstacle_mesh(
    shape: str,
    length: float,
    width_or_radius: float,
    exposure: float,
    yaw: float,
    position: tuple[float, float],
) -> trimesh.Trimesh:
    """Create a partly buried horizontal link with the requested exposed height."""
    if shape in {"cylinder", "hexagonal_prism"}:
        sections = 32 if shape == "cylinder" else 6
        mesh = trimesh.creation.cylinder(radius=width_or_radius, height=length, sections=sections)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, (0.0, 1.0, 0.0)))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0.0, 0.0, 1.0)))
        center_height = exposure - width_or_radius
    else:
        mesh = trimesh.creation.box(extents=(length, width_or_radius, exposure))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0.0, 0.0, 1.0)))
        center_height = exposure / 2.0
    mesh.apply_translation((position[0], position[1], center_height))
    return mesh


def _plus_surface(cfg, heights: np.ndarray) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Build a surface and fade its boundary to the flat fight mat."""
    mesh, grid_x, grid_y = _height_surface_mesh(cfg.size, cfg.surface_resolution, heights)
    edge_distance = np.minimum.reduce((grid_x, cfg.size[0] - grid_x, grid_y, cfg.size[1] - grid_y))
    mesh.vertices[:, 2] *= np.clip(edge_distance / 0.4, 0.0, 1.0).ravel()
    return mesh, mesh.vertices[:, 2]


def _plus_perlin(cfg, difficulty: float, seed: int) -> np.ndarray:
    """Generate deterministic Perlin noise without changing NumPy's RNG state."""
    sample_x = int(np.ceil(cfg.size[0] / cfg.surface_resolution)) + 1
    sample_y = int(np.ceil(cfg.size[1] / cfg.surface_resolution)) + 1
    amplitude = np.interp(difficulty, (0.0, 1.0), cfg.perlin_noise_range)
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return generate_fractal_noise_2d(
            xSize=cfg.size[0],
            ySize=cfg.size[1],
            xSamples=sample_x,
            ySamples=sample_y,
            frequency=8,
            fractalOctaves=2,
            fractalLacunarity=2.0,
            fractalGain=0.25,
            zScale=amplitude,
            centering=True,
        )
    finally:
        np.random.set_state(state)


def _plus_obstacle_width(shape: str, width_or_radius: float) -> float:
    """Return the horizontal width of a partly buried obstacle."""
    return 2.0 * width_or_radius if shape != "box" else width_or_radius


def _plus_footprint_axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    lateral = np.array([-forward[1], forward[0]])
    return forward, lateral


def _plus_footprint_corners(position: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    """Return the four corners of an oriented obstacle footprint."""
    forward, lateral = _plus_footprint_axes(yaw)
    half_forward = 0.5 * length * forward
    half_lateral = 0.5 * width * lateral
    return np.asarray(
        [
            position + half_forward + half_lateral,
            position + half_forward - half_lateral,
            position - half_forward + half_lateral,
            position - half_forward - half_lateral,
        ]
    )


def _plus_footprints_overlap(
    first: tuple[np.ndarray, float, float, float],
    second: tuple[np.ndarray, float, float, float],
    clearance: float = 0.02,
) -> bool:
    """Conservatively test two oriented rectangular footprints with SAT."""
    first_position, first_yaw, first_length, first_width = first
    second_position, second_yaw, second_length, second_width = second
    first_axes = _plus_footprint_axes(first_yaw)
    second_axes = _plus_footprint_axes(second_yaw)
    center_delta = second_position - first_position

    for axis in (*first_axes, *second_axes):
        first_radius = 0.5 * (
            first_length * abs(np.dot(first_axes[0], axis)) + first_width * abs(np.dot(first_axes[1], axis))
        )
        second_radius = 0.5 * (
            second_length * abs(np.dot(second_axes[0], axis)) + second_width * abs(np.dot(second_axes[1], axis))
        )
        if abs(np.dot(center_delta, axis)) >= first_radius + second_radius + clearance:
            return False
    return True


def _plus_candidate_clear(
    footprint: tuple[np.ndarray, float, float, float],
    center: np.ndarray,
    cfg,
    occupied: list[tuple[np.ndarray, float, float, float]],
) -> bool:
    position, yaw, length, width = footprint
    corners = _plus_footprint_corners(position, yaw, length, width)
    if np.any(corners < 0.0) or np.any(corners > np.asarray(cfg.size)):
        return False

    forward, lateral = _plus_footprint_axes(yaw)
    center_in_local_frame = np.array([np.dot(center - position, forward), np.dot(center - position, lateral)])
    distance_to_footprint = np.linalg.norm(
        np.maximum(
            np.abs(center_in_local_frame) - np.array([0.5 * length, 0.5 * width]),
            0.0,
        )
    )
    if distance_to_footprint < cfg.spawn_clearance:
        return False
    return not any(_plus_footprints_overlap(footprint, other) for other in occupied)


@generate_wall
def structured_fight_mat_plus_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.StructuredFightMatPlusTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate one blind-walking Plus terrain family at a given difficulty."""
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    rng = np.random.default_rng(np.random.SeedSequence((cfg.seed, int(difficulty * 2**32))))
    center = np.asarray(cfg.size, dtype=float) / 2.0
    kind = cfg.terrain_kind

    if kind == "perlin":
        heights = _plus_perlin(cfg, difficulty, int(rng.integers(2**31)))
        surface, _ = _plus_surface(cfg, heights)
        return [surface], _surface_origin(cfg.size, surface.vertices)
    if kind == "rough":
        severity = difficulty
        low, high = cfg.rough_height_range
        surface, _ = _smooth_random_surface(cfg.size, cfg.surface_resolution, (low * severity, high * severity), rng)
        return [surface], _surface_origin(cfg.size, surface.vertices)
    if kind == "wave":
        sample_x = int(np.ceil(cfg.size[0] / cfg.surface_resolution)) + 1
        sample_y = int(np.ceil(cfg.size[1] / cfg.surface_resolution)) + 1
        x = np.linspace(0.0, cfg.size[0], sample_x)
        y = np.linspace(0.0, cfg.size[1], sample_y)
        grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
        amplitude = np.interp(difficulty, (0.0, 1.0), cfg.wave_amplitude_range)
        heights = (
            0.5
            * amplitude
            * (
                np.sin(2.0 * np.pi * cfg.wave_count * grid_x / cfg.size[0])
                + np.sin(2.0 * np.pi * cfg.wave_count * grid_y / cfg.size[1])
            )
        )
        surface, _ = _plus_surface(cfg, heights)
        return [surface], _surface_origin(cfg.size, surface.vertices)
    if kind == "boxes":
        count = int(round(np.interp(difficulty, (0.0, 1.0), cfg.box_count_range)))
        max_height = np.interp(difficulty, (0.0, 1.0), cfg.box_height_range)
        meshes = [_ground_box(cfg.size)]
        occupied: list[tuple[np.ndarray, float, float, float]] = []
        for _ in range(count):
            footprint = rng.uniform(*cfg.box_footprint_range, size=2)
            yaw = rng.uniform(-np.pi, np.pi)
            for _attempt in range(100):
                position = rng.uniform((0.0, 0.0), np.asarray(cfg.size))
                candidate = (position, yaw, float(footprint[0]), float(footprint[1]))
                if _plus_candidate_clear(candidate, center, cfg, occupied):
                    occupied.append(candidate)
                    break
            else:
                continue
            height = rng.uniform(cfg.box_height_range[0], max_height)
            box = trimesh.creation.box(extents=(footprint[0], footprint[1], height))
            box.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0, 0, 1)))
            box.apply_translation((position[0], position[1], height / 2.0))
            meshes.append(box)
        return meshes, np.array([center[0], center[1], 0.0])
    if kind != "horizontal":
        raise ValueError(f"unknown Plus terrain kind: {kind!r}")

    exposure_min, exposure_max = cfg.obstacle_exposure_range
    max_exposure = np.interp(difficulty, (0.0, 1.0), (exposure_min, exposure_max))
    obstacle_count = int(round(np.interp(difficulty, (0.0, 1.0), cfg.obstacle_count_range)))
    shape_weights = np.asarray(cfg.obstacle_shape_weights, dtype=float)
    if shape_weights.shape != (3,) or np.any(shape_weights < 0) or not np.isclose(shape_weights.sum(), 1.0):
        raise ValueError("obstacle_shape_weights must contain three non-negative weights summing to one")
    direction_angles = np.asarray([np.arctan2(y, x) for x, y in CARDINAL_AND_DIAGONAL_DIRECTIONS])
    ring_candidates: list[tuple[np.ndarray, float]] = []
    inner_count = min(8, int(round(8.0 * difficulty)))
    upper_progress = np.clip((difficulty - 0.5) / 0.5, 0.0, 1.0)
    second_count = min(8, int(round(8.0 * upper_progress)))
    outer_count = min(4, int(round(4.0 * upper_progress)))
    for radius, count, offset in (
        (cfg.inner_ring_radius, inner_count, 0.0),
        (cfg.second_ring_radius, second_count, np.pi / 8.0),
    ):
        for angle in direction_angles[:count] + offset:
            position = center + radius * np.array([np.cos(angle), np.sin(angle)])
            ring_candidates.append((position, angle + np.pi / 2.0))

    target_count = min(obstacle_count, len(ring_candidates) + outer_count)
    meshes = [_ground_box(cfg.size)]
    occupied: list[tuple[np.ndarray, float, float, float]] = []
    shape_schedule = _weighted_shape_schedule(target_count, shape_weights, rng)
    placements: list[tuple[np.ndarray, float] | None] = [
        *ring_candidates,
        *([None] * outer_count),
    ][:target_count]
    for placement, shape in zip(placements, shape_schedule, strict=True):
        accepted = None
        for _attempt in range(100):
            length = float(rng.uniform(*cfg.obstacle_length_range))
            if shape == "cylinder":
                width_or_radius = float(rng.uniform(*cfg.cylinder_radius_range))
            elif shape == "hexagonal_prism":
                width_or_radius = float(rng.uniform(*cfg.hex_radius_range))
            else:
                width_or_radius = float(rng.uniform(*cfg.box_width_range))
            width = _plus_obstacle_width(shape, width_or_radius)
            if placement is None:
                angle = rng.uniform(-np.pi, np.pi)
                radius = rng.uniform(*cfg.outer_lane_radius_range)
                position = center + radius * np.array([np.cos(angle), np.sin(angle)])
                yaw = angle + np.pi / 2.0
            else:
                position, yaw = placement
            candidate = (position, yaw, length, width)
            if _plus_candidate_clear(candidate, center, cfg, occupied):
                accepted = (candidate, width_or_radius)
                break
        if accepted is None:
            continue

        candidate, width_or_radius = accepted
        position, yaw, length, _ = candidate
        occupied.append(candidate)
        current_mode = min(
            exposure_min + difficulty * (cfg.obstacle_exposure_mode - exposure_min),
            max_exposure,
        )
        exposure = rng.triangular(max(0.005, 0.5 * max_exposure), current_mode, max_exposure)
        meshes.append(
            _horizontal_obstacle_mesh(
                shape,
                length,
                width_or_radius,
                exposure,
                yaw,
                tuple(position),
            )
        )
    return meshes, np.array([center[0], center[1], 0.0])
