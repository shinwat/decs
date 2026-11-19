from decs.utils import no_overlap, sample_env_params
from itamps.utils import FixedSizeQueue, sample_trajectory_with_timeout
import numpy as np
import os
from dataclasses import dataclass
import pyrallis

@dataclass
class CollectConfig:
    path: str = "" # Path to save the demos
    size: int = 10 # Size of the dataset
    dense: bool = False # Whether to use dense rewards
    collect_conf: bool = True # Whether to collect configurations
    collect_traj: bool = True # Whether to collect trajectories
    collect_env: bool = True # Whether to collect environment parameters
    collect_stat: bool = True # Whether to collect plan statistics
    direct: bool = True # Whether to bypass the GUI
    max_time: int = 5 # time allotted for planning, in seconds
    empty: bool = False # whether or not the table is empty
    d_min: float = 0.2 # minimum distance between start and goal
    d_max: float = 0.4 # maximum distance between start and goal

@pyrallis.wrap()
def collect(config: CollectConfig):

    traj_buffer = FixedSizeQueue(config.size)
    conf_buffer = FixedSizeQueue(config.size)
    env_buffer = FixedSizeQueue(config.size)
    stats_buffer = FixedSizeQueue(config.size)
    action_stats = []
    action_std = []
    states_stats = []
    states_std = []
    traj_length = []
    infeasible_cnt = 0
    bad_demo_cnt = 0
    ablate = "schedule" if config.empty else "uniform"
    while len(traj_buffer.queue) < config.size:
        while True:
            env_params, _ = sample_env_params(
                ablate,
                d_min=config.d_min,
                d_max=config.d_max,
            )
            if no_overlap(env_params):
                break
        new_trajectory = sample_trajectory_with_timeout(
            problem='push',
            timeout=15,
            max_time=config.max_time,
            direct=config.direct,
            env_params=env_params,
        )
        if new_trajectory is None:
            continue
        # something went wrong during trajectory execution/collection
        if new_trajectory['time'] == 0.:
            print('no trajectory returned.')
            bad_demo_cnt += 1
            continue
        # plan was not solved
        if new_trajectory['time'] == -1.:
            print('no plan.')
            infeasible_cnt += 1
            continue
        new_trajectory.pop('heuristic', None) # don't need heuristic info
        stats_buffer.enqueue(new_trajectory.pop('time', None))
        conf_buffer.enqueue(new_trajectory.pop('config', None))
        traj_buffer.enqueue(new_trajectory)
        env_buffer.enqueue(env_params)
        actions = new_trajectory['actions']
        action_stats.append([max(np.abs(i)) for i in zip(*actions)])
        action_std.append([np.std(i) for i in zip(*actions)])
        states = new_trajectory['states']
        states_stats.append([max(np.abs(i)) for i in zip(*states)])
        states_std.append([np.std(i) for i in zip(*states)])
        traj_length.append(len(new_trajectory['dones']))
    print("Number of infeasible tasks: ", infeasible_cnt)
    print("Number of solved tasks with unusable demos: ", bad_demo_cnt)
    print("average trajectory length: ", np.mean(traj_length))
    print("max over actions:", [max(i) for i in zip(*action_stats)])
    print("mean std over actions:", [np.mean(i) for i in zip(*action_std)])
    print("max over states:", [max(i) for i in zip(*states_stats)])
    print("mean std over states:", [np.mean(i) for i in zip(*states_std)])
    if config.collect_traj:
        demo_path = os.path.join(config.path, "demos_{}.npy".format(len(traj_buffer.queue)))
        np.save(demo_path, traj_buffer.queue)
    if config.collect_conf:
        conf_path = os.path.join(config.path, "confs_{}.npy".format(len(conf_buffer.queue)))
        np.save(conf_path, conf_buffer.queue)
    if config.collect_env:
        env_path = os.path.join(config.path, "envs_{}.npy".format(len(env_buffer.queue)))
        np.save(env_path, env_buffer.queue)
    if config.collect_stat:
        stat_path = os.path.join(config.path, "stats_{}.npy".format(len(stats_buffer.queue)))
        np.save(stat_path, stats_buffer.queue)

if __name__ == "__main__":
    collect()