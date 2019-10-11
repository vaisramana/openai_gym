


import numpy as np
import agent.wrapper

#UP = 0
#RIGHT = 1
#DOWN = 2
#LEFT = 3


class bespoke(agent.wrapper.agent):
    def __init__(self, env):
        self.predefined_actions = np.ones(env.shape, dtype=int)
        print(self.predefined_actions)
        self.predefined_actions[-1, :] = 0
        self.predefined_actions[:, -1] = 2
        self.predefined_actions = self.predefined_actions.flatten()
        print(self.predefined_actions)
        
    
    def act(self, observation):
        pos_idx = observation
        return self.predefined_actions[pos_idx] 

    def learn(self, *args):
        pass





