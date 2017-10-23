from DQN import DQN, DDQN
import numpy as np
from numpy import random
import torch
from torch.autograd import Variable

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

myddqn = DDQN(1, 2, 0.99)
myddqn.learn(gen_batch(64,2))

x = np.zeros((1000,1))
x = np.linspace(-5,5,1000)
y_tar0 = (x-1)**2
y_tar1 = (x+2.5)**2

y0 = myddqn.Q_value(x,np.repeat(0,1000))
y1 = myddqn.Q_value(x,np.repeat(1,1000))
y0 = y0.numpy()
y1 = y1.numpy()
y2 = myddqn.Q_value(x)
print y2
import matplotlib.pyplot as plt
plt.plot(x,y1,x, y_tar1)
plt.show()
plt.plot(x,y0,x, y_tar0)
plt.show()
