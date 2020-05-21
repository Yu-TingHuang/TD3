import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import RMSprop
from torch.optim import SGLD
from torch.optim import ExtraAdam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def sgld_update(target, source, beta):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - beta) + param.data * beta)

class TD3(object):
    def __init__(self,
                state_dim,
                action_dim,
                max_action,
                optimizer,
                two_player,
                discount=0.99,
                tau=0.005,
                beta=0.9,
                alpha=0.1,
                epsilon=1e-3,
                policy_noise=0.2,
                noise_clip=0.5,
                policy_freq=2,
                expl_noise=0.1):

         self.actor = Actor(state_dim, action_dim, max_action).to(device)
         self.actor_target = copy.deepcopy(self.actor)
         self.actor_bar = copy.deepcopy(self.actor)
         self.actor_outer = copy.deepcopy(self.actor)

         self.adversary = Actor(state_dim, action_dim, max_action).to(device)
         self.adversary_target = copy.deepcopy(self.actor)
         self.adversary_bar = copy.deepcopy(self.actor)
         self.adversary_outer = copy.deepcopy(self.actor)
         if(optimizer == 'SGLD'):
             self.actor_optimizer = SGLD(self.actor.parameters(), lr=1e-4, noise=epsilon, alpha=0.999)
             self.adversary_optimizer = SGLD(self.actor.parameters(), lr=1e-4, noise=epsilon, alpha=0.999)
         elif(optimizer == 'RMSprop'):
             self.actor_optimizer = RMSprop(self.actor.parameters(), lr=1e-4, alpha=0.999)
             self.adversary_optimizer = RMSprop(self.actor.parameters(), lr=1e-4, alpha=0.999)
         else:
             self.actor_optimizer = ExtraAdam(self.actor.parameters(), lr=1e-4)
             self.adversary_optimizer = ExtraAdam(self.actor.parameters(), lr=1e-4)

         self.critic = Critic(state_dim, action_dim).to(device)
         self.critic_target = copy.deepcopy(self.critic)
         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

         self.max_action = max_action
         self.discount = discount
         self.tau = tau
         self.policy_noise = policy_noise
         self.noise_clip = noise_clip
         self.policy_freq = policy_freq
         self.total_it = 0

         self.expl_noise = expl_noise
         self.action_dim = action_dim
         self.alpha = alpha
         self.beta = beta
         self.optimizer = optimizer
         self.two_player = two_player

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if(self.optimizer == 'SGLD' and self.two_player):
            mu = self.actor_outer(state).cpu().data.numpy().flatten()
            adv_mu = self.adversary_outer(state).cpu().data.numpy().flatten()
        else:
            mu = self.actor(state).cpu().data.numpy().flatten()
            adv_mu = self.adversary(state).cpu().data.numpy().flatten()

        mu = (mu + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)).clip(-self.max_action, self.max_action)
        adv_mu = (adv_mu + np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)).clip(-self.max_action, self.max_action)
        mu = mu * (1 - self.alpha)
        adv_mu = adv_mu * self.alpha

        action = mu + adv_mu
        return action

    def train(self, sgld_outer_update, replay_buffer, batch_size=100):

        self.total_it += 1

	# Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = ((1 - self.alpha) * self.actor_target(next_state) + self.alpha * self.adversary_target(next_state) + noise).clamp(-self.max_action, self.max_action)

	    # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            with torch.no_grad():
                if(self.optimizer == 'SGLD' and self.two_player):
                    actor_action = self.actor_outer(next_state)
                else:
                    actor_action = self.actor_target(next_state)
            action = (1 - self.alpha) * actor_action + self.alpha * self.adversary(next_state)
            adversary_loss = self.critic.Q1(state, action).mean()
            self.adversary_optimizer.zero_grad()
            adversary_loss.backward()
            self.adversary_optimizer.step()

            with torch.no_grad():
                if(self.optimizer == 'SGLD' and self.two_player):
                    adversary_action = self.adversary_outer(next_state)
                else:
                    adversary_action = self.adversary_target(next_state)
            action = (1 - self.alpha) * self.actor(next_state) + self.alpha * adversary_action
            actor_loss = -self.critic.Q1(state, action).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if(self.optimizer == 'SGLD' and self.two_player):
                self.sgld_inner_update()
            self.soft_update()

        if(sgld_outer_update and self.optimizer == 'SGLD' and self.two_player):
            self.sgld_outer_update()


    def save(self, base_dir):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

    def sgld_inner_update(self): #target source
        sgld_update(self.actor_bar, self.actor, self.beta)
        sgld_update(self.adversary_bar, self.adversary, self.beta)

    def sgld_outer_update(self): #target source
        sgld_update(self.actor_outer, self.actor_bar, self.beta)
        sgld_update(self.adversary_outer, self.adversary_bar, self.beta)

    def soft_update(self):
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.adversary_target, self.adversary, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
