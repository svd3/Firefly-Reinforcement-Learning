import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
#import ReplayMemory as RM
import copy

class DQN(nn.Module):
    def __init__(self, input_dims, n_actions, hidden = 128):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.hidden_dim = hidden

        # Advantage approximator
        self.fc1 = nn.Linear(self.input_dims, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.n_actions)

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


class DoubleQNet():
    def __init__(self, state_dim, n_actions, gamma):
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.dqn = DQN(state_dim, n_actions)
        self.target = DQN(state_dim, n_actions)#copy.deepcopy(self.dqn)

        self.gamma = gamma
        self.base_lr = 0.03 #Base learning rate
        self.lr_decay = 1.0/50 # lr = 0.1 * e^(-t/50) every update step
        self.lr_step = 100
        #self.EPOCHS = 1000

    def learn(self, batch, epochs = 500):
        for epoch in range(epochs):
            if epoch%self.lr_step == 0:
                #learningRate = self.base_lr*np.exp(-epoch*self.lr_decay)
                learningRate = self.base_lr*np.exp(-epoch/self.lr_step)
                self.optimizer = optim.Adam(self.dqn.parameters(), lr = learningRate)
            self.optimizer.zero_grad()
            ### SAMPLE NEW BATCHES!!!!
            """states, action, reward, next_states = sample(batch)"""#RM.transition(*zip(*batch))
            sample_batch = batch()
            states = sample_batch.state
            action = sample_batch.action
            reward = sample_batch.reward
            next_states = sample_batch.next_state

            next_states = Variable(torch.FloatTensor(next_states).view(-1, self.state_dim))
            states = Variable(torch.FloatTensor(states).view(-1, self.state_dim))
            action = Variable(torch.LongTensor(action).view(-1, 1))
            reward = Variable(torch.FloatTensor(reward).view(-1,1), requires_grad = False)

            pred_reward = self.dqn(states).gather(1,action)
            reward = self.gamma*self.target(next_states).max(1)[0].view(-1,1) + reward #target

            #loss = F.smooth_l1_loss(pred_reward, reward) # input, target
            loss = torch.clamp(F.mse_loss(pred_reward, reward), min = -1.0, max = 1.0 ) # input, target
            loss.backward()
            self.optimizer.step()
        #print "Done."

    def Q_value(self, state, action):
        state = Variable(torch.FloatTensor(state).view(-1, self.state_dim), volatile = True)
        action = Variable(torch.LongTensor(action).view(-1, 1))
        value = self.dqn(state).gather(1, action).data
        return value

    def Q_optimal(self, state):
        state = Variable(torch.FloatTensor(state).view(-1, self.state_dim), volatile = True)
        value = self.dqn(state).max(1)
        return value

    def set_learning_params(self, base_lr = 0.03, lr_decay = 1.0/50, lr_step = 100, epochs = 1000):
        self.base_lr = base_lr
        self.lr_decay = lr_decay
        self.lr_step = lr_step
        self.epochs = epochs

    def updateTarget(self):
        self.target = copy.deepcopy(self.dqn)
