import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Dict
from itamps.buffer import ReplayBuffer
from itamps.corl import soft_update

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            max_action: float,
            hidden_dim: int = 256,
        ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (max_action - (-max_action)) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (max_action + (-max_action)) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def act(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class SoftActorCritic:
    def __init__(
            self,
            max_action: float,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            q_network: nn.Module,
            q_optimizer: torch.optim.Optimizer,
            discount: float = 0.99,
            alpha: float = 0.4,
            tau: float = 0.05,
            device: str = "cpu",
        ):
            self.max_action = max_action
            self.actor = actor
            self.actor_optimizer = actor_optimizer
            self.qf = q_network
            self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
            self.q_optimizer = q_optimizer
            self.discount = discount
            self.alpha = alpha
            self.tau = tau
            self.device = device
            # self.max_steps = max_steps

    # def load_state_dict(self, state_dict: Dict[str, Any]):
    def train(self, buffer: ReplayBuffer, batch_size: int = 256) -> Dict[str, float]:
        self.total_it += 1

        batch, ind = buffer.sample(batch_size)
        batch = [b.to(self.device) for b in batch]

        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)

        # get targets
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.act(next_observations)
            target_q = self.q_target(next_observations, next_state_actions)
            min_qf_next_target = target_q - self.alpha * next_state_log_pi
            targets = rewards + (1.0 - dones.float()) * self.discount * min_qf_next_target   

        # Update Q
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update actor
        #TODO: same frequency as Q networks?
        pi, log_pi, _ = self.actor.act(observations)
        min_qf_pi = self.qf(observations, pi)
        policy_loss = torch.mean((self.alpha * log_pi) - min_qf_pi)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        # self.actor_lr_schedule.step() #TODO: implement?

        # Update target networks
        #TODO: same frequency as Q networks?
        soft_update(self.q_target, self.qf, self.tau)

        return log_dict