import matplotlib.pyplot as plt

import environment


class EnvLadder(environment.Environment):
    """A simple environment in which the system's state is a natural number 
    which is increased every time step. After reaching the last state it is
    reset.
    """

    def __init__(self, num_states=10, max_steps=1, allow_stay=False, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        seed:        int - 
        """
        super(EnvLadder, self).__init__(ndim = 1,
                                        initial_state = 0,
                                        noisy_dim_dist = environment.Noise.uniform,
                                        seed = seed)
        self.num_states = num_states
        self.max_steps = max_steps
        self.allow_stay = allow_stay
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
        
        shift = 0 if self.allow_stay else 1
        current_state = self.get_current_state()
        current_state += self.rnd.randint(self.max_steps) + shift
        current_state = current_state % self.num_states
        
        return current_state, 0


if __name__ == '__main__':
    
    # sample data
    steps = 10
    env = EnvLadder(num_states=5, max_steps=2)
    data = env.do_actions(num_steps=steps)[0]
    
    print 'Possible actions:'
    for action, describtion in env.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
    
    # plot data
    plt.plot(data)
    plt.show()
    