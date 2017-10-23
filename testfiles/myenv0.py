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
        self.targetUpdatefreq = 100
        self.max_action = 0.05
        self.pos_range = np.array([[0, 0], [5, 5], [-5,-5]])
        self.max_vel = np.array([0.1, 0.1])
        self.min_vel = np.array([-0.1, -0.1])
        #self.low_state = np.array([self.pos_range[0], -self.pos_range[1], -self.max_vel]) # pos(x,y), relPos(x,y), vel(x,y)
        #self.high_state = np.array([self.pos_range[1], self.pos_range[1], self.max_vel])

        self.low_state = np.array([[0,0], [-5, -5], [-0.1, -0.1]])
        self.high_state = np.array([[5,5], [5, 5], [0.1, 0.1]])
        #self.action_space = spaces.Box(-self.max_action, self.max_action, shape = (2,))
        #self.action_space = spaces.Discrete(9)
        self.action_space = spaces.MultiDiscrete([ [-1,1], [-1,1] ])
        #self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3)))
        self.observation_space = spaces.Box(self.low_state, self.high_state)

        self._seed()
        self._reset()


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        pos = np.array(self.state[0])
        vel = np.array(self.state[2])
        accel = np.clip(action, -self.max_action, self.max_action)

        vel = 0.9 * vel + accel

        pos += vel

        pos = np.clip(pos, 0, 5)
        relPos = pos - self.target

        #if (position==self.min_position and velocity<0): velocity = 0

        done = math.sqrt(sum(relPos**2)) <= 0.2

        reward = 0
        if done:
            reward = 100.0
        reward -= sum(accel**2) + 1

        self.state = np.array([pos, relPos, vel])
        #self.state = [tuple(pos),tuple(relPos), tuple(vel)]
        return self.state, reward, done, {}

    def _reset(self):
        self.target = random.uniform(0,5,2)
        try:
            pos = self.state[0]
        except AttributeError:
            pos = random.uniform(0,5,2)

        relPos = pos - self.target
        self.counter = 0
        self.state = np.array([pos, relPos, [0,0]])
        #self.state = [tuple(pos),tuple(relPos), tuple([0,0])]

        return np.array(self.state)
