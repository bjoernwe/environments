import numpy as np
import matplotlib.pyplot as plt

import environment


class EnvKai(environment.Environment):
    """A simple two-dimensional environment in which the first component is
    noise and the second component inherits the value of the first from the
    previous time step.
    """

    def __init__(self, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        seed:        int - 
        """
        super(EnvKai, self).__init__(ndim = 2,
                                     initial_state = np.zeros(2),
                                     noisy_dim_dist = environment.Noise.normal,
                                     seed=seed)
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
        
        new_state = self.rnd.normal(size=2)
        new_state[1] = self.current_state[0] + new_state[1] * 1e-4
        self.current_state = new_state
        return self.current_state, 0


if __name__ == '__main__':
    
    # sample data
    steps = 10
    env = EnvKai()
    data = env.do_actions(num_steps=steps)[0]
    
    print 'Possible actions:'
    for action, describtion in env.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
    
    # plot data
    plt.plot(data)
    plt.show()
    