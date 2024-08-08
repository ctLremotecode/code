import numpy as np
from VEC_env import VECEnv

env = VECEnv()

class StateNormalization(object):
    def __init__(self):
        self.high_state = np.array([150,150,150,150,10,8,20,15])
        self.low_state = np.array([120,120,120,120,5,3,10,10])
    def state_normal(self, state):
        res = (state-self.low_state)/(self.high_state - self.low_state)
        return res
