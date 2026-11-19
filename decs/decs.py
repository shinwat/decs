from itamps.utils import evaluate_policy_with_timeout, get_buffer_shapes, reward_fun, sample_trajectory_with_timeout
from itamps.corl import TwinQ, TwinV, ValueFunction, GaussianPolicy, ImplicitQLearning, SoftActorCritic, SoftActor, SoftQNetwork, SoftQs, set_seed, wandb_init
from itamps.buffer import HER_sampler, HindsightReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pyrallis
import torch
from utils import MonteCarloRegression, ResetBuffer, no_overlap, sample_env_params

ABLATIONS = ["fd", "uniform", "feasibility", "diversity", "disagreement", "schedule", "feedbackless"]
@dataclass
class TrainConfig:
    # ablations
    ablate: str = "fd" # uniform | feasibility | diversity | disagreement | schedule | fd | feedbackless
    # environment-specific
    env: str = "tiago-push"  # environment name
    online_episodes: int = int(1e3) # 1e5 # Max episodes to run environment
    # env model-specific
    env_buffer_size: int = 10_000
    env_batch_size: int = 32
    env_hidden_dim: int = 256
    env_dim: int = 10
    goal_dim: int = 2
    n_samples: int = 100 # number of samples to pick environment from
    env_replay_k: int = 4
    diversity_ratio: float = 2.0
    env_lr: float = 1e-3  # learning rate
    ignore_return: bool = False # whether to ignore the return of the skill
    # skill model-specific
    T: int = 50 # skill horizon (training)
    action_dim: int = 4
    state_dim: int = 10
    hidden_dim: int = 256
    n_hidden: int = 2
    buffer_type: str = "her"
    replay_k: int = 4
    batch_size: int = 256  # Batch size for skill networks
    dense: bool = False
    buffer_size: int = 500_000  # Replay buffer size
    actor_lr: float = 1e-4  # Actor learning rate
    qf_lr: float = 1e-4  # learning rate
    vf_lr: float = 1e-4  # learning rate
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 10.0 # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.9 # Coefficient for asymmetric loss
    alpha: float = 0.2 # Entropy regularization coefficient
    autotune: bool = False # automatic entropy tuning l. 202
    checkpoints_path: Optional[str] = None # Path to save model checkpoints
    save_checkpoints: bool = True
    results_path: Optional[str] = None # Path to save evaluation results
    load_model: str = ""  # "" doesn't load
    # evaluation
    eval_config_path: Optional[str] = "data/confs/100" # path to evaluation configs
    eval_env_path: Optional[str] = "data/envs/100"
    eval_freq: int = int(1e2) # 1e4 # How often (episodes) we evaluate and save checkpoints
    # miscellaneous
    device: str = "cpu"
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    direct: bool = True
    timeout: int = 5
    # Wandb logging
    wandb: bool = False
    project: str = "DECS"
    name: str = "SAC"
    group: str = "test"

    def __post_init__(self):
        self.name = f"{self.name}-{self.ablate}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
        if self.results_path is not None:
            self.results_path = os.path.join(self.results_path, self.name)
@pyrallis.wrap()
def train(config: TrainConfig):

    if config.ablate not in ABLATIONS:
        print(f'Please choose one of {ABLATIONS} for the ablation.')
        return

    seed = config.seed
    set_seed(seed)
    max_action = float(1)
    device = "cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu"

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
    if config.results_path is not None:
        print(f"Results path: {config.results_path}")
        os.makedirs(config.results_path, exist_ok=True)

    # initialize environment buffer
    env_replay_buffer = ResetBuffer(
        config.env_dim,
        config.goal_dim,
        config.env_buffer_size,
        device,
        config.env_replay_k
    )

    # initialize skill buffer
    if config.buffer_type == "her": 
        buffer_shapes = get_buffer_shapes(config.T, config.state_dim, config.action_dim)
        sampler = HER_sampler(replay_k=config.replay_k, reward_fun=reward_fun, dense=config.dense)
        replay_buffer = HindsightReplayBuffer(
            buffer_shapes,
            config.buffer_size,
            config.T,
            sampler,
            device
        )
    else:
        buffer_class =  PrioritizedReplayBuffer if config.buffer_type == "per" else ReplayBuffer
        replay_buffer = buffer_class(
            config.state_dim,
            config.action_dim,
            config.buffer_size,
            device
        )

    # initialize environment model
    env_network = TwinV(
        config.env_dim, 
        config.hidden_dim, 
        config.n_hidden
    ).to(config.device)
    v_optimizer = torch.optim.Adam(env_network.parameters(), lr=config.env_lr)
    kwargs = {
        "v_network": env_network,
        "v_optimizer": v_optimizer,
        "max_steps": config.online_episodes,
        "device": device
    }
    env_trainer = MonteCarloRegression(**kwargs)

    # initialize skill model
    if config.name == "IQL":
        q_network = TwinQ(
            config.state_dim, 
            config.action_dim, 
            config.hidden_dim, 
            config.n_hidden
        ).to(device)
        actor = GaussianPolicy(
            config.state_dim, 
            config.action_dim, 
            max_action, 
            config.hidden_dim, 
            config.n_hidden, 
        ).to(device)
        q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        v_network = ValueFunction(
            config.state_dim,
            config.hidden_dim,
            config.n_hidden
        ).to(config.device)
        v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
        kwargs = {
            "max_action": max_action,
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "q_network": q_network,
            "q_optimizer": q_optimizer,
            "v_network": v_network,
            "v_optimizer": v_optimizer,
            "discount": config.discount,
            "tau": config.tau,
            "device": config.device,
            # IQL
            "beta": config.beta,
            "iql_tau": config.iql_tau,
            "max_steps": config.online_episodes * config.T,
        }
        trainer = ImplicitQLearning(**kwargs)
    else: # SAC
        q_network = SoftQs(
            config.state_dim, 
            config.action_dim, 
            config.hidden_dim, 
        ).to(device)
        # qf2 = SoftQNetwork(
        #     config.state_dim, 
        #     config.action_dim, 
        #     config.hidden_dim, 
        # ).to(device)
        q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
        actor = SoftActor(
            config.state_dim, 
            config.action_dim, 
            max_action, 
            config.hidden_dim, 
            config.n_hidden, 
        ).to(device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        kwargs = {
            "max_action": max_action,
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "q_network": q_network,
            # "qf2": qf2,
            "q_optimizer": q_optimizer,
            "discount": config.discount,
            "tau": config.tau,
            "device": config.device,
            # SAC
            "alpha": config.alpha,
            "autotune": config.autotune
        }
        trainer = SoftActorCritic(**kwargs)

    skill_modules = {'push': {'plan-push-motion': trainer}}

    # optionally load model #TODO: implement load_state_dict
    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))

    # set up wandb
    if config.wandb:
        os.environ["WANDB_SILENT"] = "true"
        run = wandb_init(asdict(config))
        run.define_metric("episode")
        run.define_metric("eval/*", step_metric = "episode")

    success_rates = []
    admittance_rates = []
    env_steps = []

    eval_config_dataset = np.load(Path(config.eval_config_path), allow_pickle=True).tolist()
    eval_env_dataset = np.load(Path(config.eval_env_path), allow_pickle=True).tolist()

    print("---------------------------------------")
    print(f"Training {config.name}, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")
    # training loop
    num_unsolved = 0
    for t in range(config.online_episodes + 1):

        # Evaluate episode
        if t % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            print('Buffer size: ', replay_buffer._size)
            successes = []
            for eval_config_data, eval_env_data in zip(eval_config_dataset, eval_env_dataset):
                # print('Evaluating env: ', eval_env_data)
                logs = evaluate_policy_with_timeout(
                    problem='push',
                    env_params=eval_env_data,
                    timeout=60,
                    direct=config.direct,
                    config_data=eval_config_data,
                    model=trainer,
                    stats=[-1, 1],
                )
                if logs is None:
                    continue
                successes.append(logs['success'])
            success_rate = np.mean(successes)
            print("---------------------------------------")
            print(f"Success rate: {success_rate}")
            print("---------------------------------------")
            success_rates.append(success_rate)
            if t > 0:
                admittance_rate = 100 - num_unsolved
                num_unsolved = 0
                admittance_rates.append(admittance_rate)
            if config.wandb:
                train_log = {
                    "episode": t,
                    "eval/success_rate": success_rate,
                }
                if t > 0:
                    train_log["admittance_rate"] = admittance_rate
                run.log(train_log)
            # save checkpoint
            if config.checkpoints_path is not None and config.save_checkpoints:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

        # don't bother training after evaluation is over
        if t == config.online_episodes:
            continue

        # 1. sample environment parameters from a score function
        while True:
            env_params, log_dict = sample_env_params(
                config.ablate,
                t/config.online_episodes,
                env_trainer.vf,
                env_replay_buffer,
                config.env_dim,
                config.n_samples,
                config.device,
                config.diversity_ratio
            )
            if no_overlap(env_params): # don't count cases where goal is reached or there are collisions
                break
        # print('env params: ', env_params)
        if config.wandb and log_dict is not None:
            run.log(log_dict)#, step=env_trainer.total_it) #TODO: align to episodes

        # 2. generate environment
        # 3. solve the task
        # 4. roll out the skill and save trace to replay buffer
        #TODO: review log
        logs = sample_trajectory_with_timeout(
            problem='push',
            timeout=60,
            max_time=5, # for planning
            direct=config.direct,
            buffer=replay_buffer,
            skill_modules=skill_modules,
            bootstrap=True,
            affordance='Alignable',
            stats=[-1, 1],
            dense=config.dense,
            env_params=env_params,
            seed=config.seed,
            use_heuristic=False,
        )
        if logs is None:
            continue
        unsolved = logs['time'] == -1.
        num_unsolved += int(unsolved)
        uncollected = logs['time'] == 0.
        env_steps.append(trainer.total_it)

        if config.wandb and not uncollected and not unsolved:
                for log in logs['logs']:
                    run.log(log[0], step=log[-1]) #wrt episode

        # ablation where environment buffer is not needed
        if config.ablate in ["uniform", "schedule", "disagreement"]: # 
            continue

        # ablation where infeasible plans are not taken into account
        if unsolved and config.ablate == "feedbackless":
            print('not adding infeasible episode to buffer.')
        else:
            # return is -1 if plan is empty, else task return (0 or 1)
            achieved = None
            if unsolved:
                mc_return = -1 
            elif uncollected:
                mc_return = 0
            else:
                mc_return = 0 if config.ignore_return else logs['success']
                achieved = logs['achieved']
            # print('achieved position: ', achieved)

            # 5. add parameter-return tuple to reset buffer
            env_replay_buffer.add_transition(env_params, mc_return, achieved)
        
        # ablation where environment sampler need not be trained
        if config.ablate == "diversity":
            continue

        # 6. train environment model
        log_dict = env_trainer.train(env_replay_buffer, config.env_batch_size)
        if config.wandb:
            run.log(log_dict)#, step=env_trainer.total_it) #TODO: align to episodes

    np.savetxt(os.path.join(config.results_path, "success_rate.csv"), np.asarray(success_rates), delimiter=",")
    np.savetxt(os.path.join(config.results_path, "admittance_rate.csv"), np.asarray(admittance_rates), delimiter=",")
    np.savetxt(os.path.join(config.results_path, "env_steps.csv"), np.asarray(env_steps), delimiter=",")

    if config.wandb:
        run.finish()

    return

if __name__ == "__main__":
    train()