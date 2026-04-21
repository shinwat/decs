from itamps.utils import evaluate_policy_with_timeout, get_buffer_shapes, reward_fun, sample_trajectory_with_timeout
from itamps.corl import TwinQ, ValueFunction, set_seed, wandb_init
from itamps.buffer import HER_sampler, HindsightReplayBuffer
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pyrallis
import torch
from cleanrl import Actor, SoftActorCritic
from utils import MonteCarloRegression, ResetBuffer, sample_env_params

@dataclass
class TrainConfig:
    # environment-specific
    env: str = "tiago-push"  # environment name
    online_episodes: int = int(1e3) # 1e5 # Max episodes to run environment
    # env model-specific
    env_buffer_size: int = 1_000
    env_dim: int = 4 #TODO: increase
    n_samples: int = 100 # #samples to pick environment from
    # skill model-specific
    T: int = 50 # skill horizon (training)
    action_dim: int = 4
    state_dim: int = 10
    hidden_dim: int = 256
    n_hidden: int = 2
    replay_k: int = 4
    batch_size: int = 256  # Batch size for all networks
    buffer_size: int = 100_000  # Replay buffer size
    actor_lr: float = 3e-4  # Actor learning rate
    qf_lr: float = 3e-4  # learning rate
    vf_lr: float = 3e-4  # learning rate
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    alpha: float = 10.0 #TODO: get proper number
    checkpoints_path: Optional[str] = "data/models/push/"
    save_checkpoints: bool = False
    load_model: str = ""  # "" doesn't load
    # evaluation
    eval_config_path: Optional[str] = "data/confs/100" # path to evaluation configs
    eval_freq: int = int(1e2) # 1e4 # How often (episodes) we evaluate and save checkpoints
    # miscellaneous
    device: str = "cpu"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    # Wandb logging
    wandb: bool = True
    project: str = "DECS"
    name: str = "SAC"
    group: str = "test"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

@pyrallis.wrap()
def train(config: TrainConfig):

    seed = config.seed
    set_seed(seed)
    max_action = float(1)
    device = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # initialize environment buffer
    env_replay_buffer = ResetBuffer(
        config.env_dim,
        config.env_buffer_size,
        config.device
    )

    # initialize skill buffer
    buffer_shapes = get_buffer_shapes(config.T, config.state_dim, config.action_dim)
    sampler = HER_sampler(replay_k=config.replay_k, reward_fun=reward_fun)
    replay_buffer = HindsightReplayBuffer(
        buffer_shapes,
        config.buffer_size,
        config.T,
        sampler,
        device
    )

    # initialize environment model
    env_network = ValueFunction(
        config.env_dim, 
        config.hidden_dim, 
        config.n_hidden
    ).to(config.device)
    v_optimizer = torch.optim.Adam(env_network.parameters(), lr=config.vf_lr)
    kwargs = {
        "v_network": env_network,
        "v_optimizer": v_optimizer,
        "device": device
    }
    env_trainer = MonteCarloRegression(**kwargs)

    # initialize skill model
    q_network = TwinQ(
        config.state_dim, 
        config.action_dim, 
        config.hidden_dim, 
        config.n_hidden
    ).to(device)
    actor = Actor(
        config.state_dim, 
        config.action_dim, 
        max_action, 
        config.hidden_dim, 
        # config.n_hidden, 
    ).to(device)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    #TODO: automatic entropy tuning? l. 202
    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "discount": config.discount,
        "alpha": config.alpha,
        "tau": config.tau,
        "device": config.device,
    }
    trainer = SoftActorCritic(**kwargs)

    # optionally load model #TODO: implement load_state_dict
    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))

    # set up wandb
    if config.wandb:
        run = wandb_init(asdict(config))
        run.define_metric("episode")
        run.define_metric("eval/*", step_metric = "episode")

    eval_config_paths = []
    for root, _, files in os.walk(config.eval_config_path):
        for file in files:
            eval_config_paths.append((root,file))

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")
    # training loop
    for t in range(config.online_episodes):

        # 1. sample environment parameters from a score function
        env_params = sample_env_params(
            env_trainer.vf,
            config.env_dim,
            config.n_samples,
            config.device
        )

        # 2. generate environment
        # 3. solve the task
        # 4. roll out the skill and save trace to replay buffer
        #TODO: review log
        log_dict = sample_trajectory_with_timeout(
            timeout=60,
            buffer=replay_buffer,
            model=trainer,
            bootstrap=True,
            affordance='Alignable',
            stats=[-1, 1],
            env_params=env_params, #TODO: will not work
        )
        # return is 0 if plan is empty, else task return
        mc_return = 0 if log_dict is None else log_dict['success']
        
        if config.wandb:
            for log in log_dict:
                run.log(log[0], step=log[-1]) #wrt episode

        # 5. add parameter-return tuple to reset buffer
        env_replay_buffer.add_transition(env_params, mc_return)
        
        # 6. train environment model
        log_dict = env_trainer.train(env_replay_buffer)
        if config.wandb:
            run.log(log_dict, step=env_trainer.total_it)

        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            print('Buffer size: ', replay_buffer._size)
            successes = []
            unsolved = 0
            for eval_config_path in eval_config_paths:
                print('Evaluating config: ', eval_config_path)
                logs = evaluate_policy_with_timeout(
                    timeout=60,
                    # direct=config.direct,
                    config_path=os.path.join(*eval_config_path),
                    model=trainer,
                    stats=[-1, 1],
                )
                if logs is None:
                    unsolved += 1
                    continue
                successes.append(logs['success'])
            success_rate = np.mean(successes)
            print("---------------------------------------")
            print(f"Success rate: {success_rate}")
            print("---------------------------------------")
            if config.wandb:
                eval_log = {
                    "episode": t,
                    "eval/success_rate": success_rate,
                    "eval/unsolved": unsolved,
                }
                run.log(eval_log)
            # save checkpoint
            if config.checkpoints_path is not None and config.save_checkpoints:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

    if config.wandb:
        run.finish()

    return

if __name__ == "__main__":
    train()