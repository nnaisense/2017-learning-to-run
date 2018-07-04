# The Winning Solution for the Learning To Run Challenge 2017 by [NNAISENSE](nnaisense.com)

This repository contains the code used to train the winning controller for the [Learning to Run Challenge 2017](https://www.crowdai.org/challenges/nips-2017-learning-to-run) along with the resulting models.

Note, however, that reproducing the results using this code is hard due to several reasons. First, the learning process (mostly in Stage I: Global Policy Optimization) was manually supported -- multiple runs were executed and visually inspected to select the most promising one for the subsequent stages. Second, the original random seeds were lost. Third, the whole learning process required significant computational resources (at least a couple of weeks of a 128-CPUs machine). You have been warned.

## Cite
```bibtex
@incollection{jaskowski2018rltorunfast,
  author      = "Ja\'skowski, Wojciech and Lykkeb{\o}, Odd Rune and Toklu, Nihat Engin and Trifterer, 
                 Florian and Buk, Zden\v{e}k and Koutn\'{i}k, Jan and Gomez, Faustino",
  title       = "{Reinforcement Learning to Run... Fast}",
  editor      = "Escalera, Sergio and Weimer, Markus",
  booktitle   = "NIPS 2017 Competition Book",
  publisher   = "Springer",
  address     = "Springer",
  year        = 2018
}
```

## Results

|                    |      | noisy          | plain          | robust         |
|--------------------|------|----------------|----------------|----------------|
| all episodes       | avg  | 43.53          | 43.23          | 41.52          |
|                    | conf | [43.26, 43.79] | [42.94, 43.50] | [41.25, 41.78] |
|                    | SD   | 8.56           | 9.14           | 8.56           |
|                    | max  | 47.24          | 47.13          | 45.81          |
|                    |      |                |                |                |
| completed episodes | avg  | 46.01          | 46.06          | 43.92          |
|                    | conf | [46.00, 46.02] | [46.05, 46.07] | [43.89, 43.94] |
|                    | SD   | 0.35           | 0.38           | 0.73           |
|                    |      |                |                |                |
| max5avg10          |      | 45.91          | 45.89          | 43.91          |
| % fall             |      | 8.75           | 9.98           | 8.20           |
| % fall confidence  |      | [7.87, 9.65]   | [9.07, 10.95]  | [7.34, 9.04]   |
|                    |      |                |                |                |
| # episodes         |      | 4000           | 4000           | 4000           |

The upper section shows the statistics for all episodes, including those
where the runner fell down. The middle section measures performance based 
only on those episodes which completed the entire run (i.e. without falling down). 
The 'best of 5 runs of 10 random episodes' and 95% confidence intervals were obtained with 
bootstrapping; all averages were computed over 4000 episodes.


## Install
```bash
sudo apt-get install mpich zlib1g-dev cmake
conda env create -f environment.yml --name ltr
source activate ltr
pip install git+https://github.com/stanfordnmbl/osim-rl.git@a49a7de
```

## Executing the Best Models

### The `plain' model
```bash
./run_walker.py --transform_inputs=new_8 --memory_size=8 --diff=2 --stdclip=5 --actions=binary \
--hid_size=256 --num_hid_layers=2 --optim_batchsize=256 --n_obstacles=10 --evaluate \
--load_model=models/plain --three --nologs --n_eval_episodes=50 --nostochastic --nobind --render
```

### The `noisy' model
```bash
./run_walker.py --transform_inputs=new_8 --memory_size=8 --diff=2 --stdclip=5 --actions=binary \
--hid_size=256 --num_hid_layers=2 --optim_batchsize=256 --n_obstacles=10 --evaluate \
--load_model=models/noisy --three --nologs --n_eval_episodes=50 --nostochastic --nobind --new8_fix \
--render
```

### The `robust' model
```bash
./run_walker.py --transform_inputs=new_8 --memory_size=8 --diff=2 --stdclip=5 --actions=binary \
--hid_size=256 --num_hid_layers=2 --optim_batchsize=256 --n_obstacles=10 --three \
--load_model=models/robust --running_avg_len=740 --timesteps_per_batch=1024 --nostochastic \
--nobind --evaluate --n_eval_episodes=50 --new8_fix --nologs --render
```

## Learning

The parameters used in the following commands are exemplary. The commands do not guarantee to obtain our models (or good models at all).

### Stage I: Global Policy Initialization

```bash
./run_walker.py --transform_inputs=new_8 --memory_size=8 --diff=2 --stdclip=5 --actions=binary \
--hid_size=256 --num_hid_layers=2 --optim_batchsize=256 --gamma=0.99 --force_override --entcoeff=0 \
--optim_stepsize=0.0003 --lam=0.9 --save_every=10 -c 2 --symmetric_training --exp_name=stage1 --nologs \
--nobind
```

### Stage II: Policy Refinement

```bash
./run_walker.py --transform_inputs=new_8 --memory_size=8 --diff=2 --stdclip=5 --actions=binary \
--hid_size=256 --num_hid_layers=2 --optim_batchsize=256 --gamma=0.99 --entcoeff=0 \
--optim_stepsize=0.0003 --lam=0.9 --save_every=10 --symmetric_training \
--load_model=results/[...]/stage1/[...] --step_timeout=5.0 --exp_name=stage2 \
--force_override --nologs -c 32  --nobind
```

### Stage III: Policy Specialization

```bash
./run_walker.py --transform_inputs=new_8 --memory_size=8 --diff=2 --stdclip=5 --actions=binary \
--hid_size=256 --num_hid_layers=2 --optim_batchsize=256 --gamma=0.994 --entcoeff=0 --optim_stepsize=0.00006 \
--lam=0.95 --save_every=10 --symmetric_training --step_timeout=10.0 --horizon_hack --n_obstacles=10 --three --load_model=results/[...]/stage2/[...] --exp_name=stage3 --running_avg_len=740 --force_override \
--timesteps_per_batch=1024 --nologs -c 128 --nobind
```
