import gym
import myenv

import numpy as np
#import random
#import time, gc, sys

from itertools import count
from collections import deque

#from Agent import Agent

def show(env, agent, episodes =20):
    action_map = env.action_map
    succ = 0
    episode_duration = []
    for episode in range(episodes):
        avg_reward = 0.0
        env.reset()
        state = env.state
        state_dim = state.shape[0]
        for t in count():
            direction = np.clip(-state[0:2]*10, -1, 1)
            d = [int(e) for e in direction]
            action = [k for k, v in action_map.iteritems() if v == d][0]
            nextstate, reward, done, _ = env.step(action)
            #if t%20 == 0: print nextstate
            avg_reward += reward

            agent.replay.append((state, action, reward, nextstate))

            state = nextstate

            if done:
                succ += 1
                episode_duration.append(t+1)
                avg_reward /= (t+1)
                print "Success!!!", succ, "duration: ", t+1, "avg_reward: ", avg_reward
                break
    print "examples: ", len(agent.replay.memory)
