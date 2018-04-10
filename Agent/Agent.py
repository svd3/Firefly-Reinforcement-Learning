import gym
from numpy import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from DQN import DQN, DoubleQNet
from ReplayMemory import ReplayMemory, transition

BATCH_SIZE = 64
REPLAY_MEMORY = 500000
EPSILON  = 0.25 # Initial exploration factor - for epsilon greedy policy

class Agent():
    def __init__(self, env, n_actions):
        #super(Agent, self).__init__()
        self.env = env
        self.state_dim = env.observation_space.sample().reshape(-1,1).shape[0] #safe to reshape!!
        self.n_actions = n_actions
        self.QNet = DoubleQNet(self.state_dim, self.n_actions, 0.99)
        self.replay = ReplayMemory(REPLAY_MEMORY)

        self.steps = 0
        self.epsilon = EPSILON
        self.base_epsilon = EPSILON
        self.eps_decay = 1.0/50 #### change this

        self.batch_size = BATCH_SIZE

        #self.transition = namedtuple('transition', ('state', 'action', 'next_state', 'reward'))

    def epsilon_update(self, eps_decay=1.0/50):
        self.epsilon = self.base_epsilon * np.exp(-self.steps * eps_decay)
        self.steps += 1

    def select_action(self, state):
        if random.uniform() < self.epsilon:
            # Select Random Action
            action = random.randint(0, 9)
            return action
        # Choose optimal action (based on current Value approximation)
        _,action = self.QNet.Q_optimal(state)
        return action.data.numpy()[0]

    def batch(self):
        sample_batch = self.replay.sample(self.batch_size)
        sample_batch = transition(*zip(*sample_batch))
        return sample_batch

    def reinforce(self, epochs=100):
        self.QNet.learn(self.batch, epochs)

    def set_batch_size(self, batch_size=BATCH_SIZE):
        self.batch_size = batch_size

    """def append(self, experience):
        self.replay.append(self.transition(*experience))"""
