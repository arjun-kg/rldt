#INCOMPLETE

import numpy as np
import gym
from gym import error, spaces, utils # need this!

high = 10
QUERY0 = 0
QUERY1 = 1
QUERY2 = 2
PREDICT0 = 3 
PREDICT1 = 4


class DTLearner(gym.Env): 
    def __init__(self, os_size = None):
        if os_size is None:
            os_size = 1
        self._reset()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(np.zeros(os_size) - high, np.zeros(os_size)+ high) #INCOMPLETE
        self.done = False
        self.state = None

    def _step(self,action):
        assert self.action_space.contains(action)
        state = self.state
        val = 0
        if action == QUERY0:
            val = data[0]
        elif action == QUERY1:
            val = data[1]
        elif action == QUERY2:
            val = data[2]
        elif action == PREDICT0:
            if label == 0:
                reward = 1
            else:
                reward = -1
            done = True
        elif action == PREDICT1:
            if label == 1:
                reward = 1
            else:
                reward = -1
            done = True
        
        elif action == PREDICT2:
            if label == 2:
                reward = 1
            else:
                reward = -1
            done = True

        self.state += [val]
        return self.state,reward, done, dummy

    def _reset(self):
        pass 