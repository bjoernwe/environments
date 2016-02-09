import os
import numpy as np
import scipy

from enum import Enum
from matplotlib import animation

import environment


class EnvData(environment.Environment):
    """
    An environment that serves as a unified interface to different static
    data sets.
    """

    Datasets = Enum('Datasets', 'eeg meg')
    

    def __init__(self, dataset):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        """
        
        if dataset == self.Datasets.eeg:
            self.data = np.load(os.path.dirname(__file__) + '/eeg.npy')
        elif dataset == self.Datasets.meg:
            self.data = np.load(os.path.dirname(__file__) + '/meg.npy')
        else:
            assert False

        self.counter = 0
        super(EnvData, self).__init__(ndim = self.data.shape[1],
                                      initial_state = self.data[0],
                                      noisy_dim_dist = environment.Noise.normal)
        return
    
    
    
    def _do_action(self, action):
        """Perform the given action and return the resulting state of the
        environment and the reward as well.
        --------------------------------------
        Parameters:
        action:     str - direction of the action to be performed
        --------------------------------------
        Return:
        new_state:    np.ndarray - coordinates of the agent after the step
        reward = 0
        """
        
        frames, dims = self.data.shape
        self.counter += 1
        if self.counter < frames:
            self.current_state = self.data[self.counter]
        else:
            print 'Warning: Not more than %d video frames available (%d)!' % (frames, self.counter) 
            self.current_state = np.zeros(dims)
        return self.current_state, 0
    
    

def main():
    for dat in EnvData.Datasets:
        env = EnvData(dataset=dat)
        print "%s: %d frames with %d dimensions" % (dat, env.data.shape[0], env.data.shape[1])



if __name__ == '__main__':
    main()
