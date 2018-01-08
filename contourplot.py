import gym
import myenv

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import cPickle as pickle

from itertools import count
from collections import deque

from Agent import Agent

with open('agent33.pkl', 'rb') as input:
    agent = pickle.load(input)


matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.05
X = np.arange(-5.0, 5.0, delta)
Y = np.arange(-5.0, 5.0, delta)
#X, Y = np.meshgrid(x, y)

state = np.array([[X,Y], [0,0]])
Z = np.zeros((len(X), len(Y)))
print Z.shape
i=j=0
for x in X:
    j = 0
    for y in Y:
        state = np.array([[x,y], [0,0]])
        a,b = agent.QNet.Q_optimal(state)
        Z[i][j] = a.data.numpy()
        j += 1
    i+=1
# difference of Gaussians


print "here"
# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')
plt.show()
