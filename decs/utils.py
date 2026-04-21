import numpy as np
import torch
from typing import  Dict, List, Tuple, Any
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

TensorBatch = List[torch.Tensor]

class ResetBuffer:
    def __init__(
        self,
        state_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer: int = 0
        self._size: int = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["states"].shape[0]
        if n_transitions > self._buffer_size:
            print(n_transitions)
            print(self._buffer_size)
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["states"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> Tuple[TensorBatch, np.ndarray]:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        rewards = self._rewards[indices]
        return [states, rewards], indices

    def add_transition(
        self,
        state: np.ndarray,
        reward: float,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._rewards[self._pointer] = self._to_tensor(reward)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

class MonteCarloRegression:
    def __init__(
        self,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        device: str = "cpu",
    ):
        self.vf = v_network
        self.v_optimizer = v_optimizer
        self.total_it = 0
        self.device = device

    def train(self, buffer: ResetBuffer, batch_size: int = 256) -> Dict[str, float]:
        self.total_it += 1

        log_dict = {}
        reset_batch, _ = buffer.sample(batch_size, reset=True)
        reset_batch = [b.to(self.device) for b in reset_batch]
        (env_states, env_returns) = reset_batch
        env_returns = env_returns.squeeze(dim=-1)
        v_pred = self.vf(env_states)
        v_loss = F.mse_loss(v_pred, env_returns)
        log_dict["value_loss"] = v_loss.item() #TODO: maybe rename
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])
        self.total_it = state_dict["total_it"]

@torch.no_grad()
def sample_env_params(vf: nn.Module, env_dim: int, n_samples: int = 100, device: str = "cpu"):
    # sample environment parameters with categorical distribution
    # uniformly sample parameters with dimension env_dim
    uniform_params = torch.rand(
        n_samples, 
        env_dim,
        dtype=torch.float32, 
        device=device
    ) # 100 x env_dim
    # optional: shift/shrink params
    param_scores = vf(uniform_params) # generate feasibility scores for each parameter
    dist = Categorical(logits=param_scores)
    sample = dist.sample()
    return uniform_params[sample.item()].cpu().data.numpy().flatten()