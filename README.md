<h1 align='center'>ac-jax</h1>
Reinforcement learning for the Andrews-Curtis conjecture. Essentially we would like to train an autonomous agent to find sequences of group operations which trivialise a balanced presentation of the trivial group,

$$
\langle x_1, \ldots, x_n  \vert r_1, \ldots, r_n \rangle \overset{\textsf{RL}}{\longrightarrow} \langle x_1, \ldots, x_n  \vert  x_1, \ldots, x_n \rangle
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

#### 1. Generate curriculum data
```bash
# check arguments
python3 -m ac_jax.env.generate_curriculum_data -h

# generate data (example args)
python3 -m ac_jax.env.generate_curriculum_data --total-easy 10000 --total-med 15000 --total-hard 10000 --fname data/ac_dataset_35k_64
```
Note the compilation time for data generation may be long due to the loop logic, but it should run fast once compiled. 

#### 2. Train PPO agent
The default architecture consists of two independent fully connected networks parameterising the actor and critic. By default we train on a small dataset consisting of ~1200 balanced presentations of the Miller-Schupp series outlined in [Shehper et. al.](https://arxiv.org/abs/2408.15332) which is included in this repo. 
```bash
# check arguments
python3 -m ac_jax.ppo_train -h

# check all hyperparameters
vim ac_jax/ppo_train.py

# run training for 50M frames (may be slow to compile at the start)
python3 -m ac_jax.ppo_train --model-type mlp --learning-rate 2.5e-4 --num-envs 1024 --env-steps 5e7
```
* This takes around 15 minutes on an NVIDIA RTX 5000, and consistently plateaus at around 330/1190 of the above Miller-Schupp instances solved.
* For a more performant model, one may use instead the generated curriculum dataset above, toggle the curriculum training flag, and swap to a transformer-based agent;
```bash
# run training for 200M frames.
python3 -m ac_jax.ppo_train --model-type transformer --learning-rate 2.5e-4 --num-envs 2048 --env-steps 2e8 --curriculum-train --curriculum-data-path data/ac_dataset_35k_64.npz
```
* This takes around 4.5h on an RTX 5000, and should net around 480/1190 instances solved, which may be increased via top--k stochastic sampling from the policy.
* Diminishing returns; increasing the number of samples in the curriculum dataset to $O(10^6)$ will see the transformer-based agent plateau at about 510/1190 instances solved in 5e8 frames. A more principled reward structure/increased horizon length/longer maximum relator length may be necessary for an RL agent to reliably crack the other half of the dataset. 

## Evaluation
Metrics and checkpoints are locally captured and periodically saved to disk. [`ac_jax.evaluation`](ac_jax/evaluation.py) contains useful utilities for evaluating saved checkpoints, including top-k/beam search/MCTS logic using the value function as a heuristic. For more details regarding the agent and implementation, please see [this short report](assets/ac_jax_report.pdf).
