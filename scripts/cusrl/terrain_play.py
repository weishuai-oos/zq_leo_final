"""Helpers for selecting an exact generated-terrain level during Play."""

from __future__ import annotations

from typing import NamedTuple


class TerrainLevelState(NamedTuple):
    """Current generated-terrain placement used by Play environments."""

    level: int
    num_levels: int
    terrain_types: tuple[int, ...]


class TerrainSelection(NamedTuple):
    """Validated Play selection before the simulator is constructed."""

    level: int | None
    column: int | None
    terrain_name: str | None


def _generator_cfg(env_cfg):
    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    if terrain_cfg is None:
        raise ValueError("Terrain selection requires a task with a configured terrain.")
    if getattr(terrain_cfg, "terrain_type", None) != "generator":
        raise ValueError("Terrain selection is only supported for generated-terrain tasks.")
    generator_cfg = getattr(terrain_cfg, "terrain_generator", None)
    if generator_cfg is None:
        raise ValueError("Generated terrain does not expose terrain_generator.")
    return terrain_cfg, generator_cfg


def terrain_type_columns(env_cfg) -> dict[str, tuple[int, ...]]:
    """Return the generator's curriculum column allocation by sub-terrain name.

    This mirrors IsaacLab's cumulative-proportion mapping, including its small
    epsilon, so the selection remains correct when proportions are not exact
    multiples of the number of columns.
    """
    _, generator_cfg = _generator_cfg(env_cfg)
    sub_terrains = getattr(generator_cfg, "sub_terrains", None)
    num_cols = getattr(generator_cfg, "num_cols", None)
    if not sub_terrains or num_cols is None:
        raise ValueError("Generated terrain must expose sub_terrains and num_cols.")
    names = list(sub_terrains.keys())
    total = sum(float(getattr(cfg, "proportion", 0.0)) for cfg in sub_terrains.values())
    if total <= 0.0:
        raise ValueError("Generated terrain proportions must sum to a positive value.")
    columns: dict[str, list[int]] = {name: [] for name in names}
    cumulative = []
    running = 0.0
    for name in names:
        running += float(getattr(sub_terrains[name], "proportion", 0.0)) / total
        cumulative.append(running)
    for column in range(int(num_cols)):
        fraction = column / int(num_cols) + 0.001
        try:
            index = next(index for index, limit in enumerate(cumulative) if fraction < limit)
        except StopIteration as exc:
            raise ValueError("Generated terrain proportions do not cover all columns.") from exc
        columns[names[index]].append(column)
    return {name: tuple(values) for name, values in columns.items() if values}


def _normalized_name(name: str) -> str:
    return "_".join(name.strip().lower().replace("-", "_").split())


def resolve_terrain_column(
    env_cfg, terrain_name: str | None, terrain_column: int | None
) -> tuple[int | None, str | None]:
    """Resolve an explicit type or column to one generator column."""
    if terrain_name is not None and terrain_column is not None:
        raise ValueError("--terrain_type and --terrain_column cannot be combined.")
    if terrain_name is None and terrain_column is None:
        return None, None
    columns_by_name = terrain_type_columns(env_cfg)
    _, generator_cfg = _generator_cfg(env_cfg)
    if terrain_column is not None:
        column = int(terrain_column)
        num_cols = int(generator_cfg.num_cols)
        if column < 0 or column >= num_cols:
            raise ValueError(f"--terrain_column must be in [0, {num_cols - 1}], got {column}.")
        actual_name = next(name for name, columns in columns_by_name.items() if column in columns)
        return column, actual_name

    assert terrain_name is not None
    normalized = _normalized_name(terrain_name)
    matches = [name for name in columns_by_name if _normalized_name(name) == normalized]
    if not matches:
        available = ", ".join(columns_by_name)
        raise ValueError(f"Unknown --terrain_type '{terrain_name}'. Available types: {available}.")
    actual_name = matches[0]
    # Selecting a type means selecting its first curriculum column.  All of
    # its columns generate the same type, so this is deterministic and works
    # for both single- and multi-environment Play.
    return columns_by_name[actual_name][0], actual_name


def configure_fixed_terrain_selection(
    env_cfg, terrain_level: int | None, terrain_name: str | None = None, terrain_column: int | None = None
) -> TerrainSelection:
    """Validate and configure a fixed row plus optional generated-terrain column."""
    column, actual_name = resolve_terrain_column(env_cfg, terrain_name, terrain_column)
    if terrain_level is None and (column is not None or actual_name is not None):
        raise ValueError("--terrain_type/--terrain_column must be combined with --terrain_level.")
    if terrain_level is None:
        return TerrainSelection(None, None, None)
    terrain_cfg, generator_cfg = _generator_cfg(env_cfg)
    num_levels = getattr(generator_cfg, "num_rows", None)
    if num_levels is None:
        raise ValueError("Generated terrain does not expose terrain_generator.num_rows.")
    level = int(terrain_level)
    num_levels = int(num_levels)
    if level < 0 or level >= num_levels:
        raise ValueError(f"--terrain_level must be in [0, {num_levels - 1}], got {level}.")
    terrain_cfg.max_init_terrain_level = level
    curriculum_cfg = getattr(env_cfg, "curriculum", None)
    if curriculum_cfg is not None and hasattr(curriculum_cfg, "terrain_levels"):
        curriculum_cfg.terrain_levels = None
    return TerrainSelection(level, column, actual_name)


def configure_fixed_terrain_level(env_cfg, terrain_level: int | None) -> int | None:
    """Validate a requested level and disable only terrain-level curriculum updates."""
    if terrain_level is None:
        return None

    return configure_fixed_terrain_selection(env_cfg, terrain_level).level


def apply_fixed_terrain_level(env, terrain_level: int, terrain_column: int | None = None) -> TerrainLevelState:
    """Move every Play environment to one exact generated-terrain row."""
    unwrapped_env = getattr(env, "unwrapped", env)
    terrain = getattr(getattr(unwrapped_env, "scene", None), "terrain", None)
    if terrain is None:
        raise ValueError("Environment does not expose scene.terrain.")

    required_attributes = ("terrain_origins", "terrain_levels", "terrain_types", "env_origins")
    missing_attributes = [name for name in required_attributes if getattr(terrain, name, None) is None]
    if missing_attributes:
        missing = ", ".join(missing_attributes)
        raise ValueError(f"Environment is not using generated terrain; missing runtime state: {missing}.")

    num_levels = int(terrain.terrain_origins.shape[0])
    level = int(terrain_level)
    if level < 0 or level >= num_levels:
        raise ValueError(f"Terrain level must be in [0, {num_levels - 1}], got {level}.")

    if terrain_column is not None:
        num_columns = int(terrain.terrain_origins.shape[1])
        column = int(terrain_column)
        if column < 0 or column >= num_columns:
            raise ValueError(f"Terrain column must be in [0, {num_columns - 1}], got {column}.")
        terrain.terrain_types.fill_(column)
    terrain.terrain_levels.fill_(level)
    terrain.env_origins[:] = terrain.terrain_origins[level, terrain.terrain_types]
    return inspect_terrain_level(env)


def inspect_terrain_level(env) -> TerrainLevelState:
    """Return and validate the terrain row currently assigned to Play."""
    unwrapped_env = getattr(env, "unwrapped", env)
    terrain = getattr(getattr(unwrapped_env, "scene", None), "terrain", None)
    if terrain is None or getattr(terrain, "terrain_levels", None) is None:
        raise ValueError("Environment does not expose generated-terrain levels.")

    levels = tuple(int(value) for value in terrain.terrain_levels.detach().cpu().tolist())
    unique_levels = tuple(sorted(set(levels)))
    if len(unique_levels) != 1:
        raise RuntimeError(f"Expected one fixed terrain level, found {unique_levels}.")

    terrain_types = tuple(sorted(set(int(value) for value in terrain.terrain_types.detach().cpu().tolist())))
    return TerrainLevelState(
        level=unique_levels[0],
        num_levels=int(terrain.terrain_origins.shape[0]),
        terrain_types=terrain_types,
    )
