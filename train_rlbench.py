import logging
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import rlbench_env
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(rgb_obs_spec, low_dim_obs_spec, action_spec, use_logger, cfg):
    cfg.rgb_obs_shape = rgb_obs_spec.shape
    cfg.low_dim_obs_shape = low_dim_obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.use_logger = use_logger
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(
            self.train_env.rgb_observation_spec(),
            self.train_env.low_dim_observation_spec(),
            self.train_env.action_spec(),
            self.cfg.use_tb or self.cfg.use_wandb,
            self.cfg.agent,
        )
        self.timer = utils.Timer()
        self.logger = Logger(
            self.work_dir, self.cfg.use_tb, self.cfg.use_wandb, self.cfg
        )
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create envs
        self.train_env = rlbench_env.make(
            self.cfg.task_name,
            self.cfg.episode_length,
            self.cfg.frame_stack,
            self.cfg.dataset_root,
            self.cfg.arm_max_velocity,
            self.cfg.arm_max_acceleration,
            self.cfg.camera_shape,
            self.cfg.camera_keys,
            self.cfg.state_keys,
            self.cfg.renderer,
        )
        # create replay buffer
        data_specs = (
            self.train_env.rgb_raw_observation_spec(),
            self.train_env.low_dim_raw_observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
            specs.Array((1,), np.float32, "demo"),
        )

        self.replay_storage = ReplayBufferStorage(
            data_specs, self.work_dir / "buffer", self.cfg.use_relabeling
        )
        self.demo_replay_storage = ReplayBufferStorage(
            data_specs,
            self.work_dir / "demo_buffer",
            self.cfg.use_relabeling,
            is_demo_buffer=True,
        )

        self.replay_loader = make_replay_loader(
            self.work_dir / "buffer",
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.do_always_bootstrap,
            self.cfg.frame_stack,
        )
        self.demo_replay_loader = make_replay_loader(
            self.work_dir / "demo_buffer",
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.do_always_bootstrap,
            self.cfg.frame_stack,
        )
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            replay_iter = iter(self.replay_loader)
            demo_replay_iter = iter(self.demo_replay_loader)
            self._replay_iter = utils.DemoMergedIterator(replay_iter, demo_replay_iter)
        return self._replay_iter

    def eval(self):
        """We use train env for evaluation, because it's convenient"""
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.train_env.reset()
            self.video_recorder.init(self.train_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.rgb_obs,
                        time_step.low_dim_obs,
                        self.global_step,
                        eval_mode=True,
                    )
                time_step = self.train_env.step(action)
                self.video_recorder.record(self.train_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f"{self.global_frame}.mp4")

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        do_eval = False

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.demo_replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.rgb_obs[0])
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f"{self.global_frame}.mp4")
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("demo_buffer_size", len(self.demo_replay_storage))
                        log("step", self.global_step)

                # do evaluation before resetting the environment
                if do_eval:
                    self.logger.log(
                        "eval_total_time", self.timer.total_time(), self.global_frame
                    )
                    self.eval()
                    do_eval = False

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.demo_replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.rgb_obs[0])
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # set a flag to initate evaluation when the current episode terminates
            if self.global_step >= self.cfg.eval_every_frames and eval_every_step(
                self.global_step
            ):
                do_eval = True

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(
                    time_step.rgb_obs,
                    time_step.low_dim_obs,
                    self.global_step,
                    eval_mode=False,
                )

            # try to update the agent
            if not seed_until_step(self.global_step):
                for _ in range(self.cfg.num_update_steps):
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.demo_replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.rgb_obs[0])
            episode_step += 1
            self._global_step += 1

    def load_rlbench_demos(self):
        if self.cfg.num_demos > 0:
            demos = self.train_env.get_demos(self.cfg.num_demos)
            for demo in demos:
                for time_step in demo:
                    self.replay_storage.add(time_step)
                    self.demo_replay_storage.add(time_step)
        else:
            logging.warning("Not using demonstrations")

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        with snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path="cfgs", config_name="config_rlbench")
def main(cfg):
    from train_rlbench import Workspace as W

    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.load_rlbench_demos()
    workspace.train()


if __name__ == "__main__":
    main()
