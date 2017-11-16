#INCOMPLETE

import numpy as np
import gym


class DTLearner(gym.Env): 
    def __init__(self):
        self._reset()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(-high, high) #INCOMPLETE
        self.done = False
        self.state = None

    def _step(self,action):
        assert self.action_space.contains(action)
        state = self.state
        
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
        return self.state,reward, done
