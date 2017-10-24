import gym
import myenv

import numpy as np
import random
import time, gc, sys
import cPickle as pickle

from itertools import count
from collections import deque

from show_example import show
from Agent import Agent

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

env = gym.make('Firefly-v0')

checkpoint = False
checkpoint_filename = 'agent_' + time.strftime("%m%d%H%M%S") + '.pkl'

if checkpoint == True:
    with open('agent32.pkl', 'rb') as input:
        agent = pickle.load(input)
else: agent = Agent(env,9)

#show(env, agent, 100)

"""
print "Initial Training..."
agent.QNet.set_learning_params(lr_step = 100)
agent.reinforce(epochs = 100)
agent.QNet.updateTarget()
print "Initially trained..."
"""

episodes = 5000
episode_duration = deque()
succ = 0
for episode in range(episodes):
    avg_reward = 0.0
    env.reset()
    state = env.state
    for t in count():
        action = int(agent.select_action(state))
        nextstate, reward, done, _ = env.step(action)
        if t % 100 == 0: print ".",
        sys.stdout.flush()
        avg_reward += reward
        #agent.replay.push((state, action, reward, nextstate))
        agent.replay.append((state, action, reward, nextstate))
        state = nextstate

        agent.QNet.set_learning_params(base_lr = 1e-4)
        agent.reinforce(epochs = 1)

        if t % 10 == 0:
            agent.QNet.updateTarget()
            gc.collect()

        if done:
            succ += 1
            episode_duration.append(t+1)
            avg_reward /= (t+1)
            print "\nActual run:: Success!!!", succ, " duration: ", t+1, " avg_reward: ", avg_reward
            agent.epsilon_update(1.0/2000.0)
            print "epsilon:", agent.epsilon
            gc.collect()
            break
    agent.QNet.updateTarget()
    save_object(agent, checkpoint_filename)

print('Complete')
