import gym
import myenv
import numpy as np
import math, copy
from numpy import random


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

#env = gym.make('Firefly-v0')
"""
action = np.array([0,0])
for i in range(10):
    abc, reward, done, _ = env.step(action)
    action = -np.array(abc[1])
    print abc, reward, done
"""
class DQN(nn.Module):
    def __init__(self, n_action, inputs, hidden=128):
        super(DQN, self).__init__()
        self.n_action = n_action
        self.input_dims = inputs
        self.hidden_dim = hidden

        self.fc1 = nn.Linear(self.input_dims, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.n_action)

    def forward(self, x):
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y= self.fc3(y)
        return y

def gen_batch(L, n_act):
    reward =  np.zeros(L)
    states = random.uniform(-5, 5, L)
    action = random.randint(0,n_act,L)
    #reward = np.zeros((L,1))
    for i in range(L):
        if action[i]==0:
            reward[i] = (states[i]-1)**2
        else:
            reward[i] = (states[i]+2.5)**2
    return (states, action, reward)

n_act=2
model = DQN(n_act,1)
model2 = DQN(n_act,1)
optimizer = optim.Adam(model.parameters(),lr = 0.1)


for t in range(500):
    if t%100 == 0:
        optimizer = optim.Adam(model.parameters(),lr = 0.1*np.exp(-t/50))
    optimizer.zero_grad()
    states, action, reward = gen_batch(64,2)
    states = Variable(torch.FloatTensor(states).view(-1,1))
    action = Variable(torch.LongTensor(action).view(-1,1))
    reward = Variable(torch.FloatTensor(reward).view(-1,1), requires_grad = False)

    exp_val = model(states).gather(1,action)
    loss = F.mse_loss(exp_val, reward) # input, target
    loss.backward()
    #for param in model.parameters():
    #    param.grad.data.clamp_(-1, 1)
    optimizer.step()
model2 = copy.deepcopy(model)
## Testing

x = np.zeros((1000,1))
x[:,0] = np.linspace(-5,5,1000)
y_tar0 = (x-1)**2
y_tar1 = (x+2.5)**2
x = Variable(torch.FloatTensor(x))
y = model2(x)
ymax = model2(x).max(1)[0]
print ymax
y = y.data.numpy()
x = x.data.numpy()

import matplotlib.pyplot as plt
plt.plot(x,y[:,1],x, y_tar1)
plt.show()
plt.plot(x,y[:,0],x, y_tar0)
plt.show()
