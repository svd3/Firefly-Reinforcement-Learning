import gym
import myenv
import numpy as np

from itertools import count

env = gym.make('Firefly-v10')
episodes = 5
#episode_duration = deque(maxlen = episodes)
succ = 0
for episode in range(episodes):
    avg_reward = 0.0
    env.reset()
    state = env._get_state()
    print "episode:", episode
    #state_ = np.append(state[4:6], state[2:4]) # restructure state
    for t in count():
        action = abs(np.random.randint(-8,9))#eval(raw_input())
        if action > 8 or action < 0: action = 0
        nextstate, reward, done, _ = env.step(action)

        env.render()

        avg_reward += reward

        state = nextstate
        #state_ = np.append(state[4:6], state[2:4])

        if done:
            succ += 1
            avg_reward /= (t+1)
            print "\nActual run:: Success!!!", succ, " duration: ", t+1, " avg_reward: ", avg_reward

            break

print('Complete')
