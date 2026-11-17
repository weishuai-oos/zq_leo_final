"""Small environment extension; physics-step injection lives in the action term."""

from isaaclab.envs import ManagerBasedRLEnv

from .impact_runtime import ImpactRuntime
from .mdp import arena_margin, foot_side_ground_contact


class T800ImpactEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._validate_ground_contact_filters()
        self.impact_runtime = ImpactRuntime(self, cfg.impact)

    def _validate_ground_contact_filters(self):
        from pxr import UsdPhysics

        from .mdp import GROUND_SENSOR_PREFIX

        paths = {
            path
            for name, sensor in self.scene.sensors.items()
            if name.startswith(GROUND_SENSOR_PREFIX)
            for path in sensor.cfg.filter_prim_paths_expr
        }
        if not paths:
            raise RuntimeError("Impact fall detection requires ground contact filters.")
        for path in paths:
            prim = self.sim.stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError(
                    f"Impact ground contact filter {path!r} is not a collision prim. "
                    "Filter the terrain collision Mesh, not its parent Xform."
                )

    def _reset_idx(self, env_ids):
        runtime = getattr(self, "impact_runtime", None)
        metrics = runtime.log_reset(env_ids) if runtime is not None else {}
        super()._reset_idx(env_ids)
        if runtime is not None:
            runtime.reset(env_ids)
            self.extras.setdefault("log", {}).update(metrics)

    def step(self, action):
        self._impact_foot_ground_flags = None
        result = super().step(action)
        # The reward pass happened before terminated environments were reset.
        # Do not recompute this from the post-reset state.
        result[-1]["impact_style_scale"] = self.impact_runtime.last_step_style_scale.clone().unsqueeze(-1)
        result[-1].setdefault("log", {})["Impact/arena_root_margin_m"] = arena_margin(self).mean()
        return result

    def _record_impact_ground_contact(self):
        # Called immediately before a physics step. The first call sees the
        # previous control step, already checked; the others see this step's
        # first three samples. Termination checks the final sample directly.
        if self._sim_step_counter % self.cfg.decimation == 1:
            return
        invalid = foot_side_ground_contact(self)
        if self._impact_foot_ground_flags is None:
            self._impact_foot_ground_flags = invalid
        else:
            self._impact_foot_ground_flags |= invalid
