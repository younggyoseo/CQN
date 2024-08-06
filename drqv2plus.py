from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class MultiViewCNNEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 4
        self.num_views = obs_shape[0]
        self.repr_dim = self.num_views * 256 * 5 * 5  # for 84,84. hard-coded

        self.conv_nets = nn.ModuleList()
        for _ in range(self.num_views):
            conv_net = nn.Sequential(
                nn.Conv2d(obs_shape[1], 32, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            )
            self.conv_nets.append(conv_net)

        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor):
        # obs: [B, V, C, H, W]
        obs = obs / 255.0 - 0.5
        hs = []
        for v in range(self.num_views):
            h = self.conv_nets[v](obs[:, v])
            h = h.view(h.shape[0], -1)
            hs.append(h)
        h = torch.cat(hs, -1)
        return h


class Actor(nn.Module):
    def __init__(
        self,
        repr_dim: int,
        low_dim: int,
        action_shape: Tuple,
        feature_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self._actor_dim = action_shape[0]

        self.rgb_encoder = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.policy = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self._actor_dim),
        )
        self.apply(utils.weight_init)

    def forward(self, rgb_obs: torch.Tensor, low_dim_obs: torch.Tensor, std: float):
        """
        Inputs:
        - rgb_obs: features from visual encoder
        - low_dim_obs: low-dimensional observations

        Outputs:
        - dist: torch distribution for policy
        """
        rgb_h = self.rgb_encoder(rgb_obs)
        low_dim_h = self.low_dim_encoder(low_dim_obs)
        h = torch.cat([rgb_h, low_dim_h], -1)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(
        self,
        repr_dim: int,
        low_dim: int,
        action_shape: tuple,
        feature_dim: int,
        hidden_dim: int,
        out_shape: tuple,
    ):
        super().__init__()
        self._actor_dim = action_shape[0]
        self._out_shape = out_shape
        out_dim = 1
        for s in out_shape:
            out_dim *= s

        self.rgb_encoder = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim * 2 + self._actor_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim * 2 + self._actor_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.apply(utils.weight_init)

    def forward(
        self, rgb_obs: torch.Tensor, low_dim_obs: torch.Tensor, actions: torch.Tensor
    ):
        """
        Inputs:
        - obs: features from visual encoder
        - low_dim_obs: low-dimensional observations
        - action: actions

        Outputs:
        - qs: (batch_size, 2)
        """
        rgb_h = self.rgb_encoder(rgb_obs)
        low_dim_h = self.low_dim_encoder(low_dim_obs)
        h = torch.cat([rgb_h, low_dim_h, actions], -1)
        q1 = self.Q1(h).view(h.shape[0], *self._out_shape)
        q2 = self.Q2(h).view(h.shape[0], *self._out_shape)
        qs = torch.cat([q1, q2], -1)
        return qs


class DistributionalCritic(Critic):
    def __init__(
        self,
        distributional_critic_limit: float,
        distributional_critic_atoms: int,
        distributional_critic_transform: bool,
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.limit = distributional_critic_limit
        self.atoms = distributional_critic_atoms
        self.transform = distributional_critic_transform

    def to_dist(self, qs):
        return torch.cat(
            [
                utils.to_categorical(
                    qs[:, q_idx].unsqueeze(-1),
                    limit=self.limit,
                    num_atoms=self.atoms,
                    transformation=self.transform,
                )
                for q_idx in range(qs.size(-1))
            ],
            dim=-1,
        )

    def from_dist(self, qs):
        return torch.cat(
            [
                utils.from_categorical(
                    qs[..., q_idx],
                    limit=self.limit,
                    transformation=self.transform,
                )
                for q_idx in range(qs.size(-1))
            ],
            dim=-1,
        )

    def compute_distributional_critic_loss(self, qs, target_qs):
        loss = 0.0
        for q_idx in range(qs.size(-1)):
            loss += -torch.sum(
                torch.log_softmax(qs[[..., q_idx]], -1)
                * target_qs.squeeze(-1).detach(),
                -1,
            )
        return loss.unsqueeze(-1)


class DrQV2Agent:
    def __init__(
        self,
        rgb_obs_shape,
        low_dim_obs_shape,
        action_shape,
        device,
        lr,
        weight_decay,
        feature_dim,
        hidden_dim,
        use_distributional_critic,
        distributional_critic_limit,
        distributional_critic_atoms,
        distributional_critic_transform,
        bc_lambda,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_logger,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_logger = use_logger
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.bc_lambda = bc_lambda
        self.use_distributional_critic = use_distributional_critic
        self.distributional_critic_limit = distributional_critic_limit
        self.distributional_critic_atoms = distributional_critic_atoms
        self.distributional_critic_transform = distributional_critic_transform

        # models
        low_dim = low_dim_obs_shape[-1]
        self.encoder = MultiViewCNNEncoder(rgb_obs_shape).to(device)
        self.actor = Actor(
            self.encoder.repr_dim, low_dim, action_shape, feature_dim, hidden_dim
        ).to(device)

        if use_distributional_critic:
            self.critic = DistributionalCritic(
                self.distributional_critic_limit,
                self.distributional_critic_atoms,
                self.distributional_critic_transform,
                self.encoder.repr_dim,
                low_dim,
                action_shape,
                feature_dim,
                hidden_dim,
                out_shape=(self.distributional_critic_atoms, 1),
            ).to(device)
            self.critic_target = DistributionalCritic(
                self.distributional_critic_limit,
                self.distributional_critic_atoms,
                self.distributional_critic_transform,
                self.encoder.repr_dim,
                low_dim,
                action_shape,
                feature_dim,
                hidden_dim,
                out_shape=(self.distributional_critic_atoms, 1),
            ).to(device)
        else:
            self.critic = Critic(
                self.encoder.repr_dim,
                low_dim,
                action_shape,
                feature_dim,
                hidden_dim,
                out_shape=(1,),
            ).to(device)
            self.critic_target = Critic(
                self.encoder.repr_dim,
                low_dim,
                action_shape,
                feature_dim,
                hidden_dim,
                out_shape=(1,),
            ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.AdamW(
            self.encoder.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=lr, weight_decay=weight_decay
        )

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.eval()

        print(self.encoder)
        print(self.critic)

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, rgb_obs, low_dim_obs, step, eval_mode):
        rgb_obs = torch.as_tensor(rgb_obs, device=self.device).unsqueeze(0)
        low_dim_obs = torch.as_tensor(low_dim_obs, device=self.device).unsqueeze(0)
        rgb_obs = self.encoder(rgb_obs)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(rgb_obs, low_dim_obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(
        self,
        rgb_obs,
        low_dim_obs,
        action,
        reward,
        discount,
        next_rgb_obs,
        next_low_dim_obs,
        step,
    ):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_rgb_obs, next_low_dim_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_qs = self.critic_target(next_rgb_obs, next_low_dim_obs, next_action)
            if self.use_distributional_critic:
                target_qs = self.critic_target.from_dist(target_qs)
            target_Q1, target_Q2 = target_qs[..., 0], target_qs[..., 1]
            target_V = torch.min(target_Q1, target_Q2).unsqueeze(1)
            target_Q = reward + (discount * target_V)
            if self.use_logger:
                metrics["critic_target_q"] = target_Q.mean().item()
            if self.use_distributional_critic:
                target_Q = self.critic_target.to_dist(target_Q)

        qs = self.critic(rgb_obs, low_dim_obs, action)

        if self.use_distributional_critic:
            critic_loss = self.critic.compute_distributional_critic_loss(
                qs, target_Q
            ).mean()
        else:
            Q1, Q2 = qs[..., 0], qs[..., 1]
            target_Q = target_Q.squeeze(1)
            critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
            if self.use_logger:
                metrics["critic_q1"] = Q1.mean().item()
                metrics["critic_q2"] = Q2.mean().item()

        if self.use_logger:
            metrics["critic_loss"] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, rgb_obs, low_dim_obs, demo_action, demos, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(rgb_obs, low_dim_obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        qs = self.critic(rgb_obs, low_dim_obs, action)
        if self.use_distributional_critic:
            qs = self.critic.from_dist(qs)
        Q1, Q2 = qs[..., 0], qs[..., 1]
        Q = torch.min(Q1, Q2)

        base_actor_loss = -Q.mean()

        bc_metrics, bc_loss = self.get_bc_loss(dist.mean, demo_action, demos)
        metrics.update(bc_metrics)
        actor_loss = base_actor_loss + self.bc_lambda * bc_loss

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_logger:
            metrics["actor_loss"] = base_actor_loss.mean().item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def get_bc_loss(self, predicted_action, buffer_action, demos):
        metrics = dict()
        bc_loss = 0
        if demos is not None:
            # Only apply loss to demo items
            demos = demos.float()
            bs = demos.shape[0]

            if demos.sum() > 0:
                bc_loss = (
                    F.mse_loss(
                        predicted_action.view(bs, -1),
                        buffer_action.view(bs, -1),
                        reduction="none",
                    )
                    * demos
                )
                bc_loss = bc_loss.sum() / demos.sum()
                if self.use_logger:
                    metrics["actor_bc_loss"] = bc_loss.item()
            if self.use_logger:
                metrics["ratio_of_demos"] = demos.mean().item()
        return metrics, bc_loss

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        (
            rgb_obs,
            low_dim_obs,
            action,
            reward,
            discount,
            next_rgb_obs,
            next_low_dim_obs,
            demos,
        ) = utils.to_torch(batch, self.device)

        # augment
        rgb_obs = rgb_obs.float()
        next_rgb_obs = next_rgb_obs.float()
        rgb_obs = torch.stack(
            [self.aug(rgb_obs[:, v]) for v in range(rgb_obs.shape[1])], 1
        )
        next_rgb_obs = torch.stack(
            [self.aug(next_rgb_obs[:, v]) for v in range(next_rgb_obs.shape[1])], 1
        )
        # encode
        rgb_obs = self.encoder(rgb_obs)
        with torch.no_grad():
            next_rgb_obs = self.encoder(next_rgb_obs)

        if self.use_logger:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(
                rgb_obs,
                low_dim_obs,
                action,
                reward,
                discount,
                next_rgb_obs,
                next_low_dim_obs,
                step,
            )
        )

        # update actor
        metrics.update(
            self.update_actor(
                rgb_obs.detach(),
                low_dim_obs.detach(),
                action,
                demos,
                step,
            )
        )

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
