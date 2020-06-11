# Robust Reinforcement Learning via Adversarial with Langevin Dynamics with DDPG and TD3

Action robust PyTorch implementation with Deep Deterministic Policy Gradients (DDPG) and Twin Delayed Deep Deterministic Policy Gradients (TD3). Implementations are based on [DDPG](https://github.com/tesslerc/ActionRobustRL) and [TD3](https://github.com/sfujim/TD3).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym).
Networks are trained using [PyTorch 1.2](https://github.com/pytorch/pytorch) and Python 3.7.

### How to train
The paper results can be reproduced by running:
```
python run_experiment.py
```
Hyper-parameters, optimizer, and configuration of one-player and two-player can be modified with different arguments to run_experiment.py.

Experiments on single environments can be run by calling:
```
python main.py --env HalfCheetah-v2
```
Hyper-parameters can be modified with different arguments to main.py.

### How to evaluate
Once models has been trained, run:
```
python run_test.py
```
so that you can evaluate several models. Hyper-parameters, optimizer, and configuration of one-player and two-player can be modified with different arguments to run_test.py.

Test on single model can be run by calling:
```
python test.py --env HalfCheetah-v2
```
Hyper-parameters can be modified with different arguments to test.py.

### How to visualize

See `plot.ipynb` for an example of how to access and visualize your models.
