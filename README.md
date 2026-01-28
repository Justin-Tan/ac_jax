<h1 align='center'>ac-jax</h1>
Reinforcement learning for the Andrews-Curtis conjecture. Essentially we would like to train an autonomous agent to find sequences of group operations which can trivialise a balanced presentation of the trivial group,

$$
\langle x_1, \ldots, x_n \, \vert \, r_1, \ldots, r_n \rangle \overset{\textsf{RL}}{\longrightarrow} \langle x_1, \ldots, x_n \, \vert \, x_1, \ldots, x_n \rangle
$$

## Installation

The below instructions are for a CPU install. If your machine has a GPU compatible with CUDA v12, please use the package spec in [assets/requirements_gpu.txt](assets/requirements_gpu.txt) instead. For other CUDA versions please follow the [Jax installation instructions](https://github.com/jax-ml/jax?tab=readme-ov-file#installation). 

```bash
# clone the repository:
git clone https://github.com/Justin-Tan/ac_jax.git
cd ac_jax

# setup virtual environment
python3 -m venv ac_env
source ac_env/bin/activate

# install dependencies
pip install -r requirements.txt
```

## Usage
* The main entry point is in [`ac_jax.ppo_train`](ac_jax/ppo_train.py). Launching this will train an actor-critic agent via PPO using curriculum learning. General hyperparameters of the optimisation may be easily modified via the `config` file present in `ac_jax.ppo_train`.
* The neural architecture parameterising the actor and critic may be found in [`ac_jax.agents`](ac_jax/agents.py). The default architecture is a transformer--based shared encoder with independent actor and critic heads, although this may be easily swapped out for a standard MLP or convolutional encoder.
* The primitive environment transition logic and the reward structure are specified in [`ac_jax.env.ac_env`](ac_jax/env/ac_env.py).
* The transition logic for synchronous processing of environments necessary for parallelisation on GPU is contained in [`ac_jax.env.curriculum`](ac_jax/env/curriculum.py).
* Data generation routines are provided in [`ac_jax.env.generate_curriculum_data`](ac_jax/env/generate_curriculum_data.py). This generates balanced presentations randomly scrambled from the trivial presentation via a sequence of elementary AC moves. The resulting dataset is divided into tiers of difficulty balanced on the scramble depth and is intended to be used in a curriculum learning schedule.

1. Generate curriculum data
```bash
# check arguments
python3 -m ac_jax.env.generate_curriculum_data -h

# generate data (example args)
python3 -m ac_jax.env.generate_curriculum_data --total-easy 10000 --total-med 10000 --total-hard 10000 --fname ac_dataset_30k_64
```
Note the compilation time for data generation may be long, but runs fast once compiled. 

2. Train PPO agent
The default architecture are two independent fully connected networks parameterising the actor and critic. For a more performant model you may swap this out for a transformer--based model via the `model-type` argument as below:
```bash
# check arguments
python3 -m ac_jax.ppo_train -h

# check all hyperparameters
vim ac_jax/ppo_train.py

# run training (for example)
python3 -m ac_jax.ppo_train --model-type mlp --learning-rate 1e-4 --num-envs 512 --initial-pool data/initial_presentations.npy --curriculum-data-path data/ac_dataset_30k_64.npz
```

For more details regarding the agent and implementation, please see [this short report](assets/ac_jax_report.pdf).