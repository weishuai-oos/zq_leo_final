"""T800 terrain curriculum: 20% flat and 80% of the historical Plus mix."""

import isaaclab.sim as sim_utils
import isaaclab.terrains as isaac_terrain
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from t800_lab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from t800_lab.terrains.trimesh import StructuredFightMatPlusTerrainCfg


# Fifty columns represent every requested proportion exactly. Each family stays
# in its columns while the ten rows increase that family's difficulty.
T800_TERRAINS_CFG = FiledTerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=50,
    horizontal_scale=0.05,
    vertical_scale=0.002,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    seed=43,
    sub_terrains={
        "fight_mat_flat": isaac_terrain.MeshPlaneTerrainCfg(proportion=0.20),
        "fight_mat_perlin": StructuredFightMatPlusTerrainCfg(
            proportion=0.08, terrain_kind="perlin", perlin_noise_range=(0.0, 0.012)
        ),
        "fight_mat_smooth_rough": StructuredFightMatPlusTerrainCfg(
            proportion=0.12, terrain_kind="rough", rough_height_range=(-0.012, 0.030)
        ),
        "fight_mat_wave": StructuredFightMatPlusTerrainCfg(
            proportion=0.12,
            terrain_kind="wave",
            wave_amplitude_range=(0.0, 0.08),
            wave_count=5.0,
        ),
        "fight_mat_sparse_boxes": StructuredFightMatPlusTerrainCfg(
            proportion=0.08,
            terrain_kind="boxes",
            box_count_range=(4, 18),
            box_height_range=(0.015, 0.08),
        ),
        "fight_mat_horizontal_obstacles": StructuredFightMatPlusTerrainCfg(
            proportion=0.40,
            terrain_kind="horizontal",
            obstacle_count_range=(0, 20),
            obstacle_exposure_range=(0.03, 0.12),
            obstacle_exposure_mode=0.10,
        ),
    },
)


def make_terrain() -> TerrainImporterCfg:
    """Keep the original robot/ground material-combination contract."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=T800_TERRAINS_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
