import matplotlib.pyplot as plt
import numpy as np

import environment


class EnvCosine(environment.Environment):
    """A simple environment in which generates cosine waves of different speed.
    """

    def __init__(self, ndim=1, pace=100, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        ndim:        int - Number of cosines/dimensions
        pace:        int - Number of steps for period of slowest cosine.
        seed:        int - 
        """
        self.ndim = ndim
        self.speed_factor = 2*np.pi/pace
        self.counter = 0
        super(EnvCosine, self).__init__(ndim = ndim,
                                        initial_state = np.ones(ndim),
                                        noisy_dim_dist = environment.Noise.uniform,
                                        seed=seed)
        self.offsets = [self.rnd.randint(0, pace) for _ in range(ndim)]
        self.current_state = self._f(self.counter)
        return
    
    
    def _f(self, counter):
        c = (self.speed_factor * counter)
        return np.array([np.cos((2**i)*c+self.offsets[i]) for i in range(self.ndim)])
    
    
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
        
        self.counter += 1
        current_state = self._f(self.counter)
        return current_state, 0


if __name__ == '__main__':
    
    # sample data
    steps = 100
    env = EnvCosine(ndim=3, pace=100)
    data = env.generate_training_data(n_train=steps, n_test=0, whitening=False)[0][0]
    
    print 'Possible actions:'
    for action, describtion in env.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
    
    # plot data
    plt.plot(data)
    plt.show()
    