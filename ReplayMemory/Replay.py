import random
from collections import deque, namedtuple

transition = namedtuple('Tr', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(deque):
    def __init__(self, capacity):
        super(ReplayMemory, self).__init__(maxlen = capacity)
        #self.transition = namedtuple('Tr', ('state', 'action', 'reward', 'next_state'))

    def push(self, args):
        self.append(transition(*args))

    def sample(self, batch_size):
        if len(self) < batch_size:
            return self
        return random.sample(self, batch_size)
