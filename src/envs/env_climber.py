import matplotlib.pyplot as plt
import numpy as np

import environment


class EnvClimber(environment.Environment):
    """A simple environment in which the current state increases by one every
    step until a randomly chosen height or the maximum is reached (in which case
    he is put back to zero).
    """

    def __init__(self, max_height=50, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        max_height:  int - maximum height for the climber
        seed:        int - 
        """
        super(EnvClimber, self).__init__(seed=seed)
        self.ndim = 1
        self.noisy_dim_dist = 'uniform'
        self.current_state = np.zeros(1)
        self.max_height = max_height
        self.target_height = self.rnd.randint(0, self.max_height)
        return
    
    
    def _do_action(self, action):
        """Perform a random step and return resulting state of the environment 
        as well as the reward (in this case 0).
        --------------------------------------
        Parameters:
        action:     0
        --------------------------------------
        Return:
        new_state:    np.ndarray - coordinates of the agent after the step
        reward = 0
        """
        
        # perform step
        self.current_state += 1
        if self.current_state > self.target_height:
            self.current_state[:] = 0
            self.target_height = self.rnd.randint(0, self.max_height)
        
        return self.current_state, 0


if __name__ == '__main__':
    
    # sample data
    steps = 1000
    env = EnvClimber()
    data = env.do_random_steps(num_steps=steps)[0]
    
    print 'Possible actions:'
    for action, describtion in env.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
    
    #data -= np.mean(data)
    #data /= np.var(data)
    
    #import gpfa
    #print gpfa.calc_predictability_det_of_avg_cov(data, k=5)
    #print gpfa.calc_predictability_det_of_avg_cov(data[::-1], k=5)

    # plot data
    plt.plot(data)
    plt.show()
    