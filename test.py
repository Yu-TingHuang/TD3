import argparse
import os
import gym
import numpy as np
import pickle
import copy
import random

import torch
from torch.distributions import uniform

from TD3 import TD3
from utils import load_model
import ast

parser = argparse.ArgumentParser()
"""

model_noise - relative mass vs. noise probability
friction_noise - relative friction vs. noise probability
model_friction - relative mass vs. relative friction

"""
parser.add_argument('--eval_type', default='model',
                    choices=['model', 'model_noise', 'friction', 'friction_noise', "model_friction"])
parser.add_argument("--two_player", type=ast.literal_eval)
parser.add_argument('--env', default="Swimmer-v2")
parser.add_argument('--alpha', default=0)
parser.add_argument("--optimizer")
parser.add_argument("--action_noise")

args = parser.parse_args()

if(args.two_player):
    base_dir = os.getcwd() + '/models_TwoPlayer/'
else:
    base_dir = os.getcwd() + '/models_OnePlayer/'


def eval_model(_env, alpha):
    total_reward = 0
    with torch.no_grad():
        state = torch.Tensor([_env.reset()])
        while True:
            action = agent.select_action(np.array(state))
            if random.random() < alpha:
                action = noise.sample(action.shape).view(action.shape)

            state, reward, done, _ = _env.step(action)
            total_reward += reward

            #state = torch.Tensor([state])
            if done:
                break
    return total_reward


test_episodes = 100
for env_name in [args.env]:#os.listdir(base_dir):

    env = gym.make(env_name)
    agent = TD3(state_dim = env.observation_space.shape[0],
                action_dim = env.action_space.shape[0],
                max_action = float(env.action_space.high[0]),
                optimizer = 0,
                two_player = args.two_player,
                discount=0.99,
                tau=0.005,
                beta=0.9,
                alpha=0,
                epsilon=0,
                policy_noise=0,
                noise_clip=0,
                policy_freq=2,
                expl_noise=0)

    noise = uniform.Uniform(torch.Tensor([-1.0]), torch.Tensor([1.0]))

    basic_bm = copy.deepcopy(env.env.model.body_mass.copy())
    basic_friction = copy.deepcopy(env.env.model.geom_friction.copy())

    env_dir = base_dir + env_name + '/'
    for optimizer in [args.optimizer]: #['RMSprop', 'SGLD_thermal_0.01', 'SGLD_thermal_0.001', 'SGLD_thermal_0.0001', 'SGLD_thermal_1e-05']:
        for noise_type in [args.action_noise]:
            noise_dir = env_dir + optimizer + '/' + noise_type + '/alpha_' + str(args.alpha) + '/'

            if os.path.exists(noise_dir):
                for subdir in sorted(os.listdir(noise_dir)):#[str(args.seed)]
                    results = {}

                    run_number = 0
                    dir = noise_dir + subdir #+ '/' + str(run_number)
                    print(dir)
                    if os.path.exists(noise_dir + subdir):#\
             # 		and not os.path.isfile(noise_dir + subdir + '/results_' + args.eval_type):
                        while os.path.exists(dir):
                            load_model(agent=agent, basedir=dir)
                           # agent.eval()

                            if 'model_noise' in args.eval_type:
                                test_episodes = 10
                                for mass in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]: #np.linspace(0.8, 1.2, 10):
                                    if mass not in results:
                                        results[mass] = {}
                                    for alpha in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: #np.linspace(0, 0.5, 10):
                                        if alpha not in results[mass]:
                                            results[mass][alpha] = []
                                        for _ in range(test_episodes):
                                            for idx in range(len(basic_bm)):
                                                env.env.model.body_mass[idx] = basic_bm[idx] * mass
                                            r = eval_model(env, alpha)
                                            results[mass][alpha].append(r)

                            elif 'friction_noise' in args.eval_type:
                                test_episodes = 10
                                for friction in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]: #np.linspace(0.8, 1.2, 10):
                                    if friction not in results:
                                        results[friction] = {}
                                    for alpha in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]: #np.linspace(0, 0.5, 10):
                                        if alpha not in results[friction]:
                                            results[friction][alpha] = []
                                        for _ in range(test_episodes):
                                            for idx in range(len(basic_friction)):
                                                env.env.model.geom_friction[idx] = basic_friction[idx] * friction
                                            r = eval_model(env, alpha)
                                            results[friction][alpha].append(r)

                            elif 'model_friction' in args.eval_type:
                                test_episodes = 10
                                for friction in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]: #np.linspace(0.8, 1.2, 10):
                                    if friction not in results:
                                        results[friction] = {}
                                    for mass in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]: #np.linspace(0, 0.5, 10):
                                        if mass not in results[friction]:
                                            results[friction][mass] = []
                                        for _ in range(test_episodes):
                                            for idx in range(len(basic_friction)):
                                                env.env.model.geom_friction[idx] = basic_friction[idx] * friction
                                            for idx in range(len(basic_bm)):
                                                env.env.model.body_mass[idx] = basic_bm[idx] * mass
                                            r = eval_model(env, 0)
                                            results[friction][mass].append(r)

                            elif 'model' in args.eval_type:
                                for mass in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]: #np.linspace(0.8, 1.2, 20):
                                    if mass not in results:
                                        results[mass] = []
                                    for _ in range(test_episodes):
                                        for idx in range(len(basic_bm)):
                                            env.env.model.body_mass[idx] = basic_bm[idx] * mass
                                        r = eval_model(env, 0)
                                        results[mass].append(r)
                            elif 'friction' in args.eval_type:
                                for friction in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]: #np.linspace(0.8, 1.2, 20):
                                    if friction not in results:
                                        results[friction] = []
                                    for _ in range(test_episodes):
                                        for idx in range(len(basic_friction)):
                                            env.env.model.geom_friction[idx] = basic_friction[idx] * friction
                                        r = eval_model(env, 0)
                                        results[friction].append(r)
                            else:
                                for alpha in np.linspace(0, 0.2, 20):
                                    if alpha not in results:
                                        results[alpha] = []
                                    for _ in range(test_episodes):
                                        r = eval_model(env, alpha)
                                        results[alpha].append(r)
                            run_number += 1
                            dir = noise_dir + subdir + '/' + str(run_number)
                        with open(noise_dir + subdir + '/results_' + args.eval_type, 'wb') as f:
                            pickle.dump(results, f)

env.close()
