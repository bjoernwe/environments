import matplotlib.pyplot as plt

import environment


class EnvEvent(environment.Environment):
    """A simple environment in which a binary event occurs randomly with a certain probability.
    """

    def __init__(self, prob=.1, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        prob:        float - probability of event
        seed:        int - 
        """
        super(EnvEvent, self).__init__(ndim=1, initial_state=0, seed=seed, noisy_dim_dist=environment.Noise.binary)
        self.prob = prob
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
        
        if self.rnd.rand() < self.prob:
            current_state = 1
        else:
            current_state = 0
        current_state += 1e-10 * self.rnd.randn()
            
        return current_state, 0


if __name__ == '__main__':
    
    # sample data
    steps = 100
    env = EnvEvent()
    data = env.do_actions(num_steps=steps)[0]
    
    print 'Possible actions:'
    for action, describtion in env.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
    
    # plot data
    plt.plot(data)
    plt.show()
    