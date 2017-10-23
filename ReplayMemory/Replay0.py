import random
import numpy as np
from collections import deque, namedtuple

transition = namedtuple('Tr', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory():
    def __init__(self, capacity):
        #super(ReplayMemory, self).__init__(maxlen = capacity)
        self.memory = deque(maxlen = capacity)
        self.priority = deque(maxlen = capacity)
        self.probability = []

    def append(self, args):
        self.memory.append(args)
        if args[2] > 0: self.priority.append(2.0)
        else: self.priority.append(1.0)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return self.memory
        self.probability = np.divide(self.priority, sum(self.priority))
        idx = np.random.choice(range(len(self.memory)), batch_size, p = self.probability)
        return [self.memory[i] for i in idx]
