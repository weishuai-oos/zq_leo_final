"""Observe every physics update without replacing IsaacLab's step/reset loop."""

from isaaclab.envs import ManagerBasedRLEnv

from t800_lab.tasks.t800_base.ankle_stable_mdp import ANKLE_JOINT_NAMES
from t800_lab.tasks.t800_base.constants import T800_FOOT_BODY_NAMES

from .stability import AnkleStabilityAccumulator


class T800AnkleStabilityEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        self._capture_ankle_physics = False
        super().__init__(cfg, **kwargs)
        self._ankle_robot = self.scene["robot"]
        self._ankle_contact = self.scene["contact_forces"]
        self._ankle_joint_ids, names = self._ankle_robot.find_joints(list(ANKLE_JOINT_NAMES), preserve_order=True)
        if tuple(names) != tuple(ANKLE_JOINT_NAMES):
            raise ValueError("Ankle stability joint mapping is incomplete.")
        self._ankle_foot_ids, names = self._ankle_robot.find_bodies(T800_FOOT_BODY_NAMES, preserve_order=True)
        self._ankle_sensor_ids, sensor_names = self._ankle_contact.find_bodies(T800_FOOT_BODY_NAMES, preserve_order=True)
        if names != T800_FOOT_BODY_NAMES or sensor_names != T800_FOOT_BODY_NAMES:
            raise ValueError("Ankle stability foot mapping is incomplete.")
        self.ankle_stability = AnkleStabilityAccumulator(
            self.num_envs, self.device, cfg.ankle_substep_stability, self.physics_dt, cfg.decimation,
        )
        self._original_scene_update = self.scene.update
        self.scene.update = self._update_scene_and_ankles

    def _update_scene_and_ankles(self, dt):
        self._original_scene_update(dt)
        if self._capture_ankle_physics:
            data = self._ankle_robot.data
            self.ankle_stability.observe(
                data.joint_vel[:, self._ankle_joint_ids],
                data.body_ang_vel_w[:, self._ankle_foot_ids],
                data.body_quat_w[:, self._ankle_foot_ids],
                self._ankle_contact.data.net_forces_w[:, self._ankle_sensor_ids],
                self._sim_step_counter, dt,
            )

    def step(self, action):
        self.ankle_stability.begin(self._ankle_robot.data.joint_vel[:, self._ankle_joint_ids], self._sim_step_counter)
        self._capture_ankle_physics = True
        try:
            result = super().step(action)
            result[-1].setdefault("log", {}).update(self.ankle_stability.metrics())
            return result
        finally:
            self._capture_ankle_physics = False

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        runtime = getattr(self, "ankle_stability", None)
        if runtime is not None:
            runtime.reset(env_ids)

    def close(self):
        self._capture_ankle_physics = False
        scene = getattr(self, "scene", None)
        if scene is not None and scene.update == self._update_scene_and_ankles:
            scene.update = self._original_scene_update
        super().close()
