# decs
Designing environments for composable skills

## Requirements
* Linux machine / Mac OS X (?) (tested on Ubuntu 22.04.4 LTS)
* Python3 (tested on 3.8.16)
* (Optional) conda / virtualenv (I use conda)
* (Optional) Nvidia GPU for faster training

## Installation

### PDDLStream (fork)
This fork has a *new robot*, [Tiago](https://pal-robotics.com/robots/tiago/), integrated in the PyBullet environment.
```
$ git clone --recursive --branch feasibility git@github.com:shinwat/pddlstream.git
$ cd pddlstream
pddlstream$ git submodule update --init --recursive
pddlstream$ ./downward/build.py
```
### ITAMPS
This repo contains the interface for PDDLStream and utility functions for model training.
```
$ git clone git@github.com:shinwat/itamps.git
$ cd itamps
itamps$ conda env create -f environment/environment.yml
itamps$ pip install -e .
```

### Clone this repo
```
itamps$ cd ..
$ git clone git@github.com:shinwat/decs.git
$ cd decs
```

### (Optional) Set up Weights & Biases
Make an account at [wandb](https://wandb.ai/) for logging if you haven't already.

## Usage

### Assumed data structure
```
decs/
└----data/
     ├----confs/
     |    └-----confs_100.npy # task configuration for evaluation
     |    └-----envs_100.npy # environment configuration for evaluation
     ├----models/
     |    ├------<ABLATION>/
     |    |      ├-----------<ALGO_ABLATION_UUID>/
     |    |      |           ├-----checkpoint_0.pt
     |    |      |           ⋮
     |    |      |           ├-----checkpoint_1000.pt
     |    |      |           └-----config.yaml
     |    |      ├----------- .../ # more seeds
     |    |      ⋮
     |    ├------ .../ # more ablations
     |    ⋮
     └----results/
          ├-------<ABLATION>/
          |       ├----------<ALGO_ABLATION_UUID>/
          |       |          ├-----admittance_rate.csv
          |       |          ├-----env_steps.csv
          |       |          └-----success_rate.csv
          |       ├----------- .../ # more seeds
          |       ⋮
          ├------ .../ # more ablations
          ⋮
```

### Reproducing results from the paper
```
decs$ chmod +x decs.sh
decs$ ./decs.sh <ablation> <from_seed> <to_seed> # note: this script can take a long time
```
The choices are between `fd`, `uniform`, `feasibility`, `diversity`, `disagreement`, `schedule`, and `feedbackless`. The paper uses 20 seeds (0 - 19).

### Plotting the results
The following scripts bundles the individual results from each ablation into 2 folders:
`<ABLATION>_success_rates.csv` and `<ABLATION>_admittance_rates.csv`.
```
decs$ cd scripts
scripts$ python bundle_results.py --directory "../data/results/<ABLATION>"  # repeat for every ablation
```
Make a folder called `success` and `admittance` under `results`:
```
scripts$ cd ../results
results$ mkdir success
results$ mkdir admittance
results$ mv <ABLATION>/<ABLATION>_success_rates.csv success # repeat for every ablation
results$ mv <ABLATION>/<ABLATION>_admittance_rates.csv admittance # repeat for every ablation
```
Finally, call the plotting script:
```
scripts$ make_plots.py --directory "../data/results/success" # for the success rate
scripts$ make_plots.py --directory "../data/results/admittance" # for the feasibility rate
```