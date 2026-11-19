import numpy as np
import torch
from typing import  Dict, List, Optional, Tuple, Any
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR

TensorBatch = List[torch.Tensor]

class ResetBuffer:
    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        buffer_size: int,
        device: str = "cpu",
        replay_k: int = 0,
    ):
        self._buffer_size = buffer_size
        self._pointer: int = 0
        self._size: int = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._goals = torch.zeros(
            (buffer_size, goal_dim), dtype=torch.float32, device=device
        )
        self.future_p = 0 if replay_k == 0 else 1 - (1. / (1 + replay_k))
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
        
        # hindsight replay
        if self.future_p != 0:
            goals = self._goals[indices]
            # ignore infeasible transitions
            feasible_mask = rewards >= 0
            feasible_mask = feasible_mask.reshape((-1)).cpu().data.numpy()
            # pick random transitions
            probs = np.random.uniform(size=batch_size)
            probs_mask = probs < self.future_p
            her_indices = np.where(feasible_mask * probs_mask)[0]
            # relabel goal
            states[her_indices,2:4] = goals[her_indices]
            rewards[her_indices] = 1.
        
        return [states, rewards], indices

    def add_transition(
        self,
        state: np.ndarray,
        reward: float,
        goal: np.ndarray = None,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._rewards[self._pointer] = self._to_tensor(reward)
        if goal is not None:
            self._goals[self._pointer] = self._to_tensor(goal)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

class MonteCarloRegression:
    def __init__(
        self,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        max_steps: int, 
        device: str = "cpu",
    ):
        self.vf = v_network
        self.v_optimizer = v_optimizer
        self.total_it = 0
        self.v_lr_schedule = CosineAnnealingLR(self.v_optimizer, max_steps)
        self.device = device

    def train(self, buffer: ResetBuffer, batch_size: int = 256) -> Dict[str, float]:
        self.total_it += 1

        log_dict = {}
        try:
            reset_batch, _ = buffer.sample(batch_size)
        except:
            return log_dict
        reset_batch = [b.to(self.device) for b in reset_batch]
        (env_states, env_returns) = reset_batch
        env_returns = env_returns.squeeze(dim=-1)
        v_pred = self.vf(env_states)
        v_loss = F.mse_loss(v_pred, env_returns)
        log_dict["env_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        self.v_lr_schedule.step()

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

# sample environment parameters from a categorical distribution
@torch.no_grad()
def sample_env_params(
    ablate: str,
    percent: float = 0.,
    vf: nn.Module = None,
    buffer: ResetBuffer = None,
    env_dim: int = 10,
    n_samples: int = 100,
    device: str = "cpu",
    beta: float = 0.1,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
):
    log_dict = {}
    if ablate in ["schedule", "uniform"]:
        while True:
            params = map_env_params(np.random.uniform(0,1,env_dim))
            if d_max is None or d_min is None:
                break
            distance = np.linalg.norm(params[:2] - params[2:4])
            if distance < d_max and distance > d_min:
                break
        if ablate == "schedule":
            params[4:] = 0 # set distractor params to 0
        return params, log_dict
    else:
        # uniformly sample parameters with dimension env_dim
        uniform_params = torch.rand(
            n_samples, 
            env_dim,
            dtype=torch.float32, 
            device=device
        )
        if ablate == "disagreement":
            a, b = vf.both(uniform_params)
            param_scores = torch.var(torch.cat((a[:,None],b[:,None]),1),1)
            log_dict["disagreement"] = torch.max(param_scores).item()
        else:
            if ablate in ["diversity", "fd", "feedbackless"]:
                entropy_scores = entropy_estimation(uniform_params, buffer, device) # generate entropy scores
                log_dict["entropy"] = torch.max(entropy_scores).item()
                param_scores = entropy_scores
            if ablate in ["feasibility", "feedbackless", "fd"]:
                feasibility_scores = vf(uniform_params) # generate feasibility scores for each parameter
                log_dict["feasibility"] = torch.max(feasibility_scores).item()
                param_scores = feasibility_scores
            if ablate in ["fd", "feedbackless"]:
                param_scores = beta * entropy_scores + feasibility_scores # balance with entropy score
                log_dict["blend"] = torch.max(param_scores).item()
        dist = Categorical(logits=param_scores)
        sample = dist.sample()
        params = uniform_params[sample.item()].cpu().data.numpy()
    params = map_env_params(params)
    return params, log_dict

# mapping params to values
def map_env_params(params):
    params[9] = (params[9] - 0.5) * 0.3 # Y_d
    params[8] = (params[8] - 0.5) * 0.4 # X_d
    params[7] = (params[7] - 0.5) * 0.3 # Y_d
    params[6] = (params[6] - 0.5) * 0.4 # X_d
    params[5] = (params[5] - 0.5) * 0.3 # Y_d
    params[4] = (params[4] - 0.5) * 0.4 # X_d
    params[3] = (params[3] - 0.5) * 0.3 # Y_g
    params[2] = (params[2] - 0.5) * 0.4 # X_g
    params[1] = (params[1] - 0.5) * 0.3 # Y_i
    params[0] = (params[0] - 0.5) * 0.4 # X_i
    return params

# based on Liu and Abbeel 2021 (eq. 3)
def entropy_estimation(params: torch.Tensor, buffer: ResetBuffer, device: str, batch_size: int = 100):
    if buffer._size < batch_size:
        return torch.zeros(params.shape[0]).to(device)
    # 1. pick previously picked parameters from buffer at random
    reset_batch, _ = buffer.sample(batch_size)
    reset_batch = [b.to(device) for b in reset_batch]
    (env_states, _) = reset_batch
    # 2. get k neighbors for each param
    distances = torch.cdist(params, env_states, p=2)
    _, nn_indices = torch.topk(distances, k=4, dim=1, largest=False)
    neighbor = env_states[nn_indices[:,-1]] 
    # 3. entropy ~= sum of the log of the distance between each particle and its k-th nearest neighbor
    # queries_expanded = params.unsqueeze(1)
    distances = torch.norm(neighbor - params, dim=-1)
    distances_clamped = distances.clamp_min(1e-12)
    log_sums = torch.log(distances_clamped)#.mean(dim=1)
    return log_sums

# determine whether objects and goal zones are overlapping
def no_overlap(params):
    block_radius = 0.05
    plate_radius = 0.07
    init = params[:2]
    goal = params[2:4]
    init_goal = np.linalg.norm(init - goal) > block_radius + plate_radius
    if np.all(params[4:] == 0):
        return init_goal
    distractor1 = params[4:6]
    distractor2 = params[6:8]
    distractor3 = params[8:]
    init_d1 = np.linalg.norm(init - distractor1) > block_radius*2
    init_d2 = np.linalg.norm(init - distractor2) > block_radius*2
    init_d3 = np.linalg.norm(init - distractor3) > block_radius*2
    d1_d2 = np.linalg.norm(distractor1 - distractor2) > block_radius*2
    d1_d3 = np.linalg.norm(distractor1 - distractor3) > block_radius*2
    d2_d3 = np.linalg.norm(distractor2 - distractor3) > block_radius*2
    goal_d1 = np.linalg.norm(distractor1 - goal) > block_radius + plate_radius
    goal_d2 = np.linalg.norm(distractor2 - goal) > block_radius + plate_radius
    goal_d3 = np.linalg.norm(distractor3 - goal) > block_radius + plate_radius
    init_d = init_d1 and init_d2 and init_d3
    goal_d = goal_d1 and goal_d2 and goal_d3
    d_d = d1_d2 and d1_d3 and d2_d3
    return init_goal and init_d and d_d and goal_d
