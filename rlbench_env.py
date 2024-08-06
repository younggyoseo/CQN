import logging

from gymnasium import spaces
import numpy as np
from typing import Union, Dict, Any, NamedTuple
from collections import deque

from pyrep.const import RenderMode
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.utils import name_to_task_class
from rlbench.action_modes.arm_action_modes import (
    JointPosition,
)

from dm_env import StepType, specs


class TimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    rgb_obs: Any
    low_dim_obs: Any
    demo: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    rgb_obs: Any
    low_dim_obs: Any
    action: Any
    demo: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ExtendedTimeStepWrapper:
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            rgb_obs=time_step.rgb_obs,
            low_dim_obs=time_step.low_dim_obs,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
            demo=time_step.demo or 0.0,
        )

    def low_dim_observation_spec(self):
        return self._env.low_dim_observation_spec()

    def rgb_observation_spec(self):
        return self._env.rgb_observation_spec()

    def low_dim_raw_observation_spec(self):
        return self._env.low_dim_raw_observation_spec()

    def rgb_raw_observation_spec(self):
        return self._env.rgb_raw_observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class RLBench:
    def __init__(
        self,
        task_name: str,
        episode_length: int = 200,
        frame_stack: int = 1,
        dataset_root: str = "",
        arm_max_velocity: float = 1.0,
        arm_max_acceleration: float = 4.0,
        camera_shape: tuple[int] = (84, 84),
        camera_keys: tuple[str] = ("front", "wrist", "left_shoulder", "right_shoulder"),
        state_keys: tuple[str] = ("joint_positions", "gripper_open"),
        renderer: str = "opengl3",
        render_mode: Union[None, str] = "rgb_array",
    ):
        self._task_name = task_name
        self._episode_length = episode_length
        self._frame_stack = frame_stack
        self._dataset_root = dataset_root
        self._arm_max_velocity = arm_max_velocity
        self._arm_max_acceleration = arm_max_acceleration
        self._camera_shape = camera_shape
        self._camera_keys = camera_keys
        self._state_keys = state_keys
        self._renderer = renderer
        self._render_mode = render_mode

        self._launch()
        self._add_gym_camera()
        self._initialize_frame_stack()
        self._construct_action_and_observation_spaces()

    def low_dim_observation_spec(self) -> spaces.Box:
        shape = self.low_dim_observation_space.shape
        spec = specs.Array(shape, np.float32, "low_dim_obs")
        return spec

    def low_dim_raw_observation_spec(self) -> spaces.Box:
        shape = self.low_dim_raw_observation_space.shape
        spec = specs.Array(shape, np.float32, "low_dim_obs")
        return spec

    def rgb_observation_spec(self) -> spaces.Box:
        shape = self.rgb_observation_space.shape
        spec = specs.Array(shape, np.uint8, "rgb_obs")
        return spec

    def rgb_raw_observation_spec(self) -> spaces.Box:
        shape = self.rgb_raw_observation_space.shape
        spec = specs.Array(shape, np.uint8, "rgb_obs")
        return spec

    def action_spec(self) -> spaces.Box:
        shape = self.action_space.shape
        spec = specs.Array(shape, np.float32, "action")
        return spec

    def step(self, action):
        action = self._convert_action_to_raw(action)
        rlb_obs, reward, terminated = self.task.step(action)
        obs = self._extract_obs(rlb_obs)
        self._step_counter += 1

        # Timelimit
        if self._step_counter >= self._episode_length:
            truncated = True
        else:
            truncated = False

        # Handle bootstrap
        if terminated or truncated:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        discount = float(1 - terminated)

        return TimeStep(
            rgb_obs=obs["rgb_obs"],
            low_dim_obs=obs["low_dim_obs"],
            step_type=step_type,
            reward=reward,
            discount=discount,
            demo=0.0,
        )

    def reset(self, **kwargs):
        # Clear deques used for frame stacking
        self._low_dim_obses.clear()
        for frames in self._frames.values():
            frames.clear()

        _, rlb_obs = self.task.reset(**kwargs)
        obs = self._extract_obs(rlb_obs)
        self._step_counter = 0

        return TimeStep(
            rgb_obs=obs["rgb_obs"],
            low_dim_obs=obs["low_dim_obs"],
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            demo=0.0,
        )

    def render(self, mode="rgb_array") -> Union[None, np.ndarray]:
        if mode != self._render_mode:
            raise ValueError(
                "The render mode must match the render mode selected in the "
                'constructor. \nI.e. if you want "human" render mode, then '
                "create the env by calling: "
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                "You passed in mode %s, but expected %s." % (mode, self._render_mode)
            )
        if mode == "rgb_array":
            frame = self._gym_cam.capture_rgb()
            frame = np.clip((frame * 255.0).astype(np.uint8), 0, 255)
            return frame

    def get_demos(self, num_demos):
        """
        1. Collect or fetch demonstrations
        2. Compute action stats from demonstrations, override self._action_stats
        3. Rescale actions in demonstrations to [-1, 1] space
        """
        live_demos = not self._dataset_root
        if live_demos:
            logging.warning("Live demo collection.. Takes a while..")
        raw_demos = self.task.get_demos(num_demos, live_demos)
        demos = []
        for raw_demo in raw_demos:
            demo = self.convert_demo_to_timesteps(raw_demo)
            if demo is not None:
                demos.append(demo)
            else:
                print("Skipping demo for large delta action")
        # override action stats with demonstration-based stats
        self._action_stats = self.extract_action_stats(demos)
        # rescale actions with action stats
        demos = [self.rescale_demo_actions(demo) for demo in demos]
        return demos

    def extract_action_stats(self, demos: list[list[ExtendedTimeStep]]):
        actions = []
        for demo in demos:
            for timestep in demo:
                actions.append(timestep.action)
        actions = np.stack(actions)

        # Gripper one-hot action's stats are hard-coded
        action_max = np.hstack([np.max(actions, 0)[:-1], 1])
        action_min = np.hstack([np.min(actions, 0)[:-1], 0])
        action_stats = {
            "max": action_max,
            "min": action_min,
        }
        return action_stats

    def extract_delta_joint_action(self, obs, next_obs):
        action = np.concatenate(
            [
                (
                    next_obs.misc["joint_position_action"][:-1] - obs.joint_positions
                    if "joint_position_action" in next_obs.misc
                    else next_obs.joint_positions - obs.joint_positions
                ),
                [1.0 if next_obs.gripper_open == 1 else 0.0],
            ]
        ).astype(np.float32)
        return action

    def convert_demo_to_timesteps(self, demo):
        timesteps = []

        # Clear deques used for frame stacking
        self._low_dim_obses.clear()
        for frames in self._frames.values():
            frames.clear()

        for i in range(len(demo)):
            rlb_obs = demo[i]

            obs = self._extract_obs(rlb_obs)
            reward, discount = 0.0, 1.0
            if i == 0:
                # zero action for the first timestep
                action_spec = self.action_spec()
                action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
                step_type = StepType.FIRST
            else:
                prev_rlb_obs = demo[i - 1]
                action = self.extract_delta_joint_action(prev_rlb_obs, rlb_obs)
                if np.any(action[:-1] > 0.2) or np.any(action[:-1] < -0.2):
                    return None
                if i == len(demo) - 1:
                    step_type = StepType.LAST
                    reward = 1.0
                    discount = 0.0
                else:
                    step_type = StepType.MID

            timestep = ExtendedTimeStep(
                rgb_obs=obs["rgb_obs"],
                low_dim_obs=obs["low_dim_obs"],
                step_type=step_type,
                action=action,
                reward=reward,
                discount=discount,
                demo=1.0,
            )
            timesteps.append(timestep)

        return timesteps

    def rescale_demo_actions(
        self, demo: list[ExtendedTimeStep]
    ) -> list[ExtendedTimeStep]:
        new_timesteps = []
        for timestep in demo:
            action = self._convert_action_from_raw(timestep.action)
            new_timesteps.append(timestep._replace(action=action))
        return new_timesteps

    def close(self) -> None:
        self._env.shutdown()

    def _launch(self):
        task_class = name_to_task_class(self._task_name)

        # Setup observation configs
        obs_config = ObservationConfig()
        obs_config.set_all_high_dim(False)
        obs_config.set_all_low_dim(False)
        assert (
            "joint_positions" in self._state_keys
        ), "joint position is required as this code assumes joint control"
        for state_key in self._state_keys:
            setattr(obs_config, state_key, True)
        for camera_key in self._camera_keys:
            camera_config = getattr(obs_config, f"{camera_key}_camera")
            camera_config.rgb = True
            camera_config.image_size = self._camera_shape
            camera_config.render_mode = (
                RenderMode.OPENGL3 if self._renderer == "opengl3" else RenderMode.OPENGL
            )
            setattr(obs_config, f"{camera_key}_camera", camera_config)

        # Setup action mode
        action_mode = MoveArmThenGripper(
            arm_action_mode=JointPosition(False), gripper_action_mode=Discrete()
        )

        # Launch environment and setup spaces
        self._env = Environment(
            action_mode,
            arm_max_velocity=self._arm_max_velocity,
            arm_max_acceleration=self._arm_max_acceleration,
            obs_config=obs_config,
            dataset_root=self._dataset_root,
            headless=True,
        )
        self._env.launch()
        self.task = self._env.get_task(task_class)

        # Episode length counter
        self._step_counter = 0

    def _add_gym_camera(self):
        if self._render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            self._gym_cam = VisionSensor.create([320, 192])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if self._render_mode == "human":
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)

    def _initialize_frame_stack(self):
        # Create deques for frame stacking
        self._low_dim_obses = deque([], maxlen=self._frame_stack)
        self._frames = {
            camera_key: deque([], maxlen=self._frame_stack)
            for camera_key in self._camera_keys
        }

    def _construct_action_and_observation_spaces(self):
        # Setup action/observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self._env.action_shape)
        self.low_dim_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8 * self._frame_stack,), dtype=np.float32
        )  # hard-coded: joint: 7, gripper_open: 1
        self.low_dim_raw_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )  # without frame stacking
        self.rgb_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(len(self._camera_keys), 3 * self._frame_stack, *self._camera_shape),
            dtype=np.uint8,
        )
        self.rgb_raw_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(len(self._camera_keys), 3, *self._camera_shape),
            dtype=np.uint8,
        )  # without frame stacking

        # Set default action stats, which will be overridden by demonstration action stats
        # Required for a case we don't use demonstrations
        action_min = (
            -np.ones(self.action_space.shape, dtype=self.action_space.dtype) * 0.2
        )
        action_max = (
            np.ones(self.action_space.shape, dtype=self.action_space.dtype) * 0.2
        )
        action_min[-1] = 0
        action_max[-1] = 1
        self._action_stats = {"min": action_min, "max": action_max}

    def _convert_action_to_raw(self, action):
        """Convert [-1, 1] action to raw joint space using action stats"""
        assert (max(action) <= 1) and (min(action) >= -1)
        action_min, action_max = self._action_stats["min"], self._action_stats["max"]
        _action_min = action_min - np.fabs(action_min) * 0.2
        _action_max = action_max + np.fabs(action_max) * 0.2
        new_action = (action + 1) / 2.0  # to [0, 1]
        new_action = new_action * (_action_max - _action_min) + _action_min  # original
        return new_action.astype(action.dtype, copy=False)

    def _convert_action_from_raw(self, action):
        """Convert raw action in joint space to [-1, 1] using action stats"""
        action_min, action_max = self._action_stats["min"], self._action_stats["max"]
        _action_min = action_min - np.fabs(action_min) * 0.2
        _action_max = action_max + np.fabs(action_max) * 0.2

        new_action = (action - _action_min) / (_action_max - _action_min)  # to [0, 1]
        new_action = new_action * 2 - 1  # to [-1, 1]
        return new_action.astype(action.dtype, copy=False)

    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        obs = vars(obs)
        out = dict()

        # Get low-dimensional state with stacking
        low_dim_obs = np.hstack(
            [obs[key] for key in self._state_keys], dtype=np.float32
        )
        if len(self._low_dim_obses) == 0:
            for _ in range(self._frame_stack):
                self._low_dim_obses.append(low_dim_obs)
        else:
            self._low_dim_obses.append(low_dim_obs)
        out["low_dim_obs"] = np.concatenate(list(self._low_dim_obses), axis=0)

        # Get rgb observations with stacking
        for camera_key in self._camera_keys:
            pixels = obs[f"{camera_key}_rgb"].transpose(2, 0, 1).copy()
            if len(self._frames[camera_key]) == 0:
                for _ in range(self._frame_stack):
                    self._frames[camera_key].append(pixels)
            else:
                self._frames[camera_key].append(pixels)
        out["rgb_obs"] = np.stack(
            [
                np.concatenate(list(self._frames[camera_key]), axis=0)
                for camera_key in self._camera_keys
            ],
            0,
        )
        return out

    def __del__(
        self,
    ) -> None:
        self.close()


def make(
    task_name,
    episode_length,
    frame_stack,
    dataset_root,
    arm_max_velocity,
    arm_max_acceleration,
    camera_shape,
    camera_keys,
    state_keys,
    renderer,
):
    env = RLBench(
        task_name,
        episode_length=episode_length,
        frame_stack=frame_stack,
        dataset_root=dataset_root,
        arm_max_velocity=arm_max_velocity,
        arm_max_acceleration=arm_max_acceleration,
        camera_shape=camera_shape,
        camera_keys=camera_keys,
        state_keys=state_keys,
        renderer=renderer,
    )
    env = ExtendedTimeStepWrapper(env)
    return env
