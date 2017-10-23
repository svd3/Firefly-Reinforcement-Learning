import gym
#import myenv

import torch
from torch.autograd import Variable

import cPickle as pickle
import numpy as np
from numpy import random
from ReplayMemory0 import ReplayMemory, transition


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

#env = gym.make('Firefly-v0')
#agent = Agent(env)
mem = ReplayMemory(10)


for episode in range(10):
    avg_reward = 0.0
    state = 0
    for t in range(100):
        action = random.randint(0,9) #agent.select_action(state)
        nextstate = random.randint(0,9)
        reward = random.uniform(-1.0,1.0)

        avg_reward += reward
        mem.append((state, action, reward, nextstate))
        state = nextstate


print mem.memory[0]
print mem.priority
print mem.probability
batch = mem.sample(3)
print batch
batch = transition(*zip(*batch))
print batch.state
state_dim = 1
print Variable(torch.FloatTensor(batch.state).view(-1,state_dim))
