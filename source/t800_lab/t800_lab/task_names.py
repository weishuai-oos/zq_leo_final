"""Task identifiers shared by launchers without importing Isaac Lab."""

ANKLE_TASKS = (
    "T800_base-AnkleSmooth",
    "T800_base-AnkleDamping",
    "T800_base-AnkleTargetLimit",
    "T800_base-AnklePose",
)
COMBINED_ANKLE_TASK = "T800_base-AnkleCombined"
COMBINED_V2_ANKLE_TASK = "T800_base-AnkleCombinedV2"
IMPACT_V2_TASK = "T800_base-ImpactV2"
IMPACT_V2_PLUS_TASK = "T800_base-ImpactV2Plus"
DYNAMICS_DR_V2_TASK = "T800_base-ImpactV2-From154500-DR"
DYNAMICS_DR_PLUS_TASK = "T800_base-ImpactV2Plus-From173000-DR"
DYNAMICS_DR_TASKS = (DYNAMICS_DR_V2_TASK, DYNAMICS_DR_PLUS_TASK)
IMPACT_TASKS = ("T800_base-Impact", IMPACT_V2_TASK, IMPACT_V2_PLUS_TASK, *DYNAMICS_DR_TASKS)
TRAIN_TASKS = ("T800_base", *IMPACT_TASKS, *ANKLE_TASKS, COMBINED_ANKLE_TASK, COMBINED_V2_ANKLE_TASK)
ALL_TASKS = tuple(name for task in TRAIN_TASKS for name in (task, task + "-Play"))


def impact_parent_task(task: str) -> str:
    """Resolve the original dynamics/recovery implementation for DR variants."""
    task = task.removesuffix("-Play")
    return {DYNAMICS_DR_V2_TASK: IMPACT_V2_TASK, DYNAMICS_DR_PLUS_TASK: IMPACT_V2_PLUS_TASK}.get(task, task)
