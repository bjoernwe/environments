import matplotlib.pyplot as plt
import numpy as np

import environment


class EnvSine(environment.Environment):
    """A simple environment in which generates a sine wave.
    """

    def __init__(self, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        seed:        int - 
        """
        super(EnvSine, self).__init__(ndim = 1,
                                      initial_state = np.sin(0),
                                      noisy_dim_dist = environment.Noise.uniform,
                                      seed=seed)
        self.counter = 0
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
        
        self.counter += 1
        return np.sin(self.counter), 0


if __name__ == '__main__':
    
    # sample data
    steps = 10
    env = EnvSine()
    data = env.generate_training_data(num_steps=steps, whitening=False, n_chunks=1)[0][0]
    
    print 'Possible actions:'
    for action, describtion in env.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
    
    # plot data
    plt.plot(data)
    plt.show()
    