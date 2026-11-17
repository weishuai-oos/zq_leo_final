"""Keep the useful fallen-opponent obstacle, with predominantly level ground."""

from t800_lab.tasks.t800_base.terrain_cfg import make_terrain as make_base_terrain


def make_terrain():
    terrain = make_base_terrain()
    generator = terrain.terrain_generator
    flat = generator.sub_terrains["fight_mat_flat"].copy()
    horizontal = generator.sub_terrains["fight_mat_horizontal_obstacles"].copy()
    flat.proportion = 0.8
    horizontal.proportion = 0.2
    generator.sub_terrains = {
        "fight_mat_flat": flat,
        "fight_mat_horizontal_obstacles": horizontal,
    }
    generator.num_cols = 10
    return terrain
