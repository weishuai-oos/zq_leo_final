from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _update_terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
) -> tuple[TerrainImporter, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update terrain rows and return transition masks for optional logging."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    levels_before_update = terrain.terrain_levels[env_ids].clone()
    top_level_solve = move_up & (levels_before_update >= terrain.max_terrain_level - 1)
    terrain.update_env_origins(env_ids, move_up, move_down)
    return terrain, move_up, move_down, top_level_solve


def _terrain_columns_by_name(terrain: TerrainImporter) -> dict[str, tuple[int, ...]]:
    """Reproduce IsaacLab's curriculum column allocation for metric grouping."""
    generator_cfg = terrain.cfg.terrain_generator
    sub_terrains = generator_cfg.sub_terrains
    proportions = [float(cfg.proportion) for cfg in sub_terrains.values()]
    total = sum(proportions)
    cumulative = []
    running_total = 0.0
    for proportion in proportions:
        running_total += proportion / total
        cumulative.append(running_total)

    columns: dict[str, list[int]] = {name: [] for name in sub_terrains}
    names = list(sub_terrains)
    for column in range(int(generator_cfg.num_cols)):
        threshold = column / generator_cfg.num_cols + 0.001
        terrain_index = next(index for index, boundary in enumerate(cumulative) if threshold < boundary)
        columns[names[terrain_index]].append(column)
    return {name: tuple(indices) for name, indices in columns.items()}


def _safe_terrain_metric_name(name: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in name).strip("_")


def terrain_levels_vel_with_stats(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Update the terrain curriculum and expose scalar level/type diagnostics.

    The scalar-only dictionary is compatible with IsaacLab's
    :class:`CurriculumManager` logger.  Occupancy metrics describe all current
    environments, while transition rates describe only the environments reset
    during this curriculum update.
    """
    terrain, move_up, move_down, top_level_solve = _update_terrain_levels_vel(env, env_ids, asset_cfg)
    levels = terrain.terrain_levels.float()
    num_levels = int(terrain.max_terrain_level)
    max_level = num_levels - 1
    stay = ~(move_up | move_down)
    reset_count = move_up.numel()

    columns_by_name = _terrain_columns_by_name(terrain)
    column_groups = torch.empty(
        int(terrain.cfg.terrain_generator.num_cols),
        device=terrain.terrain_types.device,
        dtype=torch.long,
    )
    for group_index, column_ids in enumerate(columns_by_name.values()):
        column_groups[list(column_ids)] = group_index

    terrain_groups = column_groups[terrain.terrain_types]
    level_counts = torch.bincount(
        terrain_groups * num_levels + terrain.terrain_levels,
        minlength=len(columns_by_name) * num_levels,
    ).reshape(len(columns_by_name), num_levels)
    level_counts = level_counts.float()
    global_level_counts = level_counts.sum(dim=0)

    reset_groups = column_groups[terrain.terrain_types[env_ids]]
    reset_group_counts = torch.bincount(reset_groups, minlength=len(columns_by_name)).float()
    promote_group_counts = torch.bincount(
        reset_groups,
        weights=move_up.float(),
        minlength=len(columns_by_name),
    )
    demote_group_counts = torch.bincount(
        reset_groups,
        weights=move_down.float(),
        minlength=len(columns_by_name),
    )
    top_solve_group_counts = torch.bincount(
        reset_groups,
        weights=top_level_solve.float(),
        minlength=len(columns_by_name),
    )

    stats = {
        "level/mean": levels.mean(),
        "level/p50": torch.quantile(levels, 0.50),
        "level/p90": torch.quantile(levels, 0.90),
        "level/current_max": levels.max(),
        "level/max_fraction": (levels == max_level).float().mean(),
        "transition/reset_count": levels.new_tensor(float(reset_count)),
        "transition/promote_rate": move_up.float().mean(),
        "transition/demote_rate": move_down.float().mean(),
        "transition/stay_rate": stay.float().mean(),
        "transition/top_level_solve_rate": top_level_solve.float().mean(),
        "transition/top_level_solve_count": top_level_solve.float().sum(),
    }
    for level in range(num_levels):
        stats[f"level/{level:02d}_fraction"] = global_level_counts[level] / levels.numel()

    level_indices = torch.arange(num_levels, device=levels.device, dtype=levels.dtype)
    for group_index, (terrain_name, column_ids) in enumerate(columns_by_name.items()):
        safe_name = _safe_terrain_metric_name(terrain_name)
        type_level_counts = level_counts[group_index]
        type_count = type_level_counts.sum()
        type_prefix = f"type/{safe_name}"
        stats[f"{type_prefix}/column_count"] = levels.new_tensor(float(len(column_ids)))
        stats[f"{type_prefix}/env_count"] = type_count
        stats[f"{type_prefix}/env_fraction"] = type_count / levels.numel()
        safe_type_count = type_count.clamp_min(1.0)
        stats[f"{type_prefix}/level_mean"] = (type_level_counts * level_indices).sum() / safe_type_count
        stats[f"{type_prefix}/max_level_fraction"] = type_level_counts[max_level] / safe_type_count
        for level in range(num_levels):
            stats[f"{type_prefix}/level_{level:02d}_fraction"] = type_level_counts[level] / safe_type_count

        group_reset_count = reset_group_counts[group_index]
        safe_group_reset_count = group_reset_count.clamp_min(1.0)
        stats[f"{type_prefix}/reset_count"] = group_reset_count
        stats[f"{type_prefix}/promote_rate"] = promote_group_counts[group_index] / safe_group_reset_count
        stats[f"{type_prefix}/demote_rate"] = demote_group_counts[group_index] / safe_group_reset_count
        stats[f"{type_prefix}/top_level_solve_rate"] = top_solve_group_counts[group_index] / safe_group_reset_count

    return stats
