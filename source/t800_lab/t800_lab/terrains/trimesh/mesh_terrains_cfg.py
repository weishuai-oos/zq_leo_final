from typing import List
from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass
from . import mesh_terrains


class WallTerrainCfgMixin:
    wall_prob: List[float] = [0.0, 0.0, 0.0, 0.0]  # Probability of generating walls on [left, right, front, back] sides
    wall_height: float = 5.0  # Height of the walls
    wall_thickness: float = 0.05  # Thickness of the walls


@configclass
class StructuredFightMatPlusTerrainCfg(SubTerrainBaseCfg, WallTerrainCfgMixin):
    """One difficulty-aware, blind-walking fight-mat terrain family.

    The generator selects ``terrain_kind`` for each curriculum column.  Unlike
    :class:`StructuredLowObstacleTerrainCfg`, every kind remains the same from
    row 0 through row 9; only its severity changes with difficulty.
    """

    function = mesh_terrains.structured_fight_mat_plus_terrain

    terrain_kind: str = "perlin"
    surface_resolution: float = 0.10
    perlin_noise_range: tuple[float, float] = (0.0, 0.012)
    rough_height_range: tuple[float, float] = (-0.012, 0.030)
    wave_amplitude_range: tuple[float, float] = (0.0, 0.08)
    wave_count: float = 5.0
    box_count_range: tuple[int, int] = (4, 18)
    box_height_range: tuple[float, float] = (0.015, 0.08)
    box_footprint_range: tuple[float, float] = (0.25, 0.50)

    obstacle_count_range: tuple[int, int] = (0, 20)
    obstacle_exposure_range: tuple[float, float] = (0.03, 0.12)
    obstacle_exposure_mode: float = 0.10
    obstacle_shape_weights: tuple[float, float, float] = (0.65, 0.20, 0.15)
    obstacle_length_range: tuple[float, float] = (0.80, 1.20)
    cylinder_radius_range: tuple[float, float] = (0.12, 0.22)
    hex_radius_range: tuple[float, float] = (0.12, 0.22)
    box_width_range: tuple[float, float] = (0.20, 0.38)
    spawn_clearance: float = 1.0
    inner_ring_radius: float = 1.55
    second_ring_radius: float = 2.40
    outer_lane_radius_range: tuple[float, float] = (2.80, 3.25)
    seed: int = 43
