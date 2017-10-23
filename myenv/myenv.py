import gym
from gym.utils import seeding
from gym import spaces
from collections import namedtuple
import numpy as np
from numpy import random
import math

class MyEnv(gym.Env):
    def __init__(self):
        self.counter = 0
        self.targetUpdatefreq = 100 # Not being used
        self.max_action = 0.01
        # state: [rel_pos.x , rel_pos.y, vel.x, vel.y]
        self.max_vel = 0.1
        self.min_vel = -0.1
        self.low_state = np.array([-5, -5 , -0.1, -0.1])
        self.high_state = np.array([5, 5 , 0.1, 0.1])
        self.action_space = spaces.Discrete(9)
        #self.action_space = spaces.MultiDiscrete([ [-1,1], [-1,1] ])
        #self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3)))
        self.observation_space = spaces.Box(self.low_state, self.high_state)
        self.state = self.observation_space.sample()
        self._seed()
        self._reset()

        self.action_map = {  0: [0,0],
                        1: [0,1],
                        2: [0,-1],
                        3: [1,0],
                        4: [1,1],
                        5: [1,-1],
                        6: [-1,0],
                        7: [-1,1],
                        8: [-1,-1]
                    }

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        action = self.action_map[action]
        pos = self.state[0:2]
        vel = self.state[2:4]
        accel = self.max_action * np.array(action)
        #accel = np.clip(action, -self.max_action, self.max_action)

        done = math.sqrt(sum(pos**2)) <= 0.1
        reward = 0
        if done:
            #print "reward state: ", self.state
            reward = 1.0
        reward -= (0.05 * sum(accel**2) + 0.05)

        vel = 0.99 * vel + accel
        vel = np.clip(vel, -0.1, 0.1)
        pos += vel

        pos = np.clip(pos, -5, 5)
        #relPos = pos - self.target
        self.state = np.append(pos,vel)
        #self.state = [tuple(pos),tuple(relPos), tuple(vel)]
        return self.state, reward, done, {}

    def _reset(self):
        pos = random.uniform(-5,5,2)
        #relPos = pos - self.target
        self.counter = 0
        self.state = np.append(pos,[0,0])
        #self.state = [tuple(pos),tuple(relPos), tuple([0,0])]

        return self.state
