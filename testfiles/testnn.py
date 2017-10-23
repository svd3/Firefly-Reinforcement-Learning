import numpy as np
import math
from numpy import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import ReplayMemory as RM

class DQN(nn.Module):
    def __init__(self, input_dims, n_action, hidden = 128):
        super(DQN, self).__init__()
        self.n_action = n_action
        self.input_dims = input_dims
        self.hidden_dim = hidden

        # Advantage approximator
        self.fc1 = nn.Linear(self.input_dims, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.n_action)

        """# Value approximator
        self.fc1_v = nn.Linear(self.input_dims, self.hidden_dim)
        self.fc2_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3_v = nn.Linear(self.hidden_dim, 1)

        # Q aggregator
        #self.fc1_q = nn.Linear(self.input_dims, self.hidden_dim)
        #self.fc2_q ="""

    def forward(self, x):
        adv = F.relu(self.fc1(x))
        adv = F.relu(self.fc2(adv))
        adv = self.fc3(adv)
        return adv

class Learner():
    def __init__(self, input_dims, out, gamma):
        #self.trainNet = None
        #super(Learner, self.trainNet).__init__(input_dims, out)
        self.trainNet = DQN(input_dims, out)
        self.gamma = gamma
        self.base_lr = 0.03 #Base learning rate
        self.lr_decay = 1.0/50 # lr = 0.1 * e^(-t/50) every update step
        self.lr_step = 100
        self.epochs = 1000

    def learn(self, targetNet, gen_batch):
        for epoch in range(self.epochs):
            if epoch%self.lr_step == 0:
                learningRate = self.base_lr*np.exp(-epoch*self.lr_decay)
                self.optimizer = optim.Adam(self.trainNet.parameters(), lr = learningRate)
            self.optimizer.zero_grad()
            ### SAMPLE NEW BATCHES!!!!
            states, action, reward = gen_batch(64,2)
            next_states = Variable(torch.FloatTensor(states).view(-1,1))
            states = Variable(torch.FloatTensor(states).view(-1,1))
            action = Variable(torch.LongTensor(action).view(-1,1))
            reward = Variable(torch.FloatTensor(reward).view(-1,1), requires_grad = False)

            pred_reward = self.trainNet(states).gather(1,action)
            reward = self.gamma*targetNet(next_states).max(1)[0].view(-1,1) + reward

            loss = F.smooth_l1_loss(pred_reward, reward) # input, target
            loss.backward()
            self.optimizer.step()
        print "Done."

def gen_batch1(L, n_act):
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

targetNet = DQN(1,2)
mylearner = Learner(1,2,0.99)
mylearner.learn(targetNet, gen_batch1)

x = np.zeros((1000,1))
x[:,0] = np.linspace(-5,5,1000)
y_tar0 = (x-1)**2
y_tar1 = (x+2.5)**2
x = Variable(torch.FloatTensor(x))
y = mylearner.trainNet(x)
#ymax = model(x).max(1)[0]
#print ymax
y = y.data.numpy()
x = x.data.numpy()

import matplotlib.pyplot as plt
plt.plot(x,y[:,1],x, y_tar1)
plt.show()
plt.plot(x,y[:,0],x, y_tar0)
plt.show()
