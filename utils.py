import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

def save_model(actor, adversary, basedir=None):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    actor_path = "{}/ddpg_actor".format(basedir)
    adversary_path = "{}/ddpg_adversary".format(basedir)

    # print('Saving models to {} {}'.format(actor_path, adversary_path))
    torch.save(actor.state_dict(), actor_path)
    torch.save(adversary.state_dict(), adversary_path)


def load_model(agent, basedir=None):
    actor_path = "{}/ddpg_actor".format(basedir)
    adversary_path = "{}/ddpg_adversary".format(basedir)

    print('Loading models from {} {}'.format(actor_path, adversary_path))
    agent.actor.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
    agent.adversary.load_state_dict(torch.load(adversary_path, map_location=lambda storage, loc: storage))
