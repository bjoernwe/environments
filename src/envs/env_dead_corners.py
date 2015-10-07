import matplotlib.pyplot as plt
import numpy as np

import environment


class EnvDeadCorners(environment.Environment):
    """Simulates a random walk in a square. Reaching one of the corners
    puts the 'agent' back to the center of the square.
    """

    def __init__(self, sigma=0.2, corner_size=.2, ndim=2, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        sigma:       float - standard deviation
        ndim:        int - dimensionality of the generated cube
        seed:        int - 
        """
        super(EnvDeadCorners, self).__init__(seed=seed)
        self.ndim = ndim
        self.noisy_dim_dist = 'uniform'
        self.sigma = sigma
        self.corner_size = corner_size
        self.current_state = np.zeros(ndim)
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
        for d in range(self.ndim):
            self.current_state[d] += self.sigma * self.rnd.normal()
        self.current_state = np.clip(self.current_state, -1, 1)
        # 
        dists_to_corners = 1 - np.abs(self.current_state)
        if np.all(dists_to_corners < self.corner_size):
            self.current_state = np.zeros(self.ndim)
            print 'reset'
        return self.current_state, 0


if __name__ == '__main__':
    
    # sample data
    steps = 2000
    cube = EnvDeadCorners()
    data = cube.do_random_steps(num_steps=steps)[0]
    
    print 'Possible actions:'
    for action, describtion in cube.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
    
    # plot data
    plt.scatter(data[:,0], data[:,1])
    #plt.plot(data[:,0])
    #plt.plot(data[:,1])
    plt.show()
    