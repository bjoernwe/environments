import matplotlib.pyplot as plt
import numpy as np

import environment


class EnvRibbon(environment.Environment):
    """A simple environment in which the agent moves along a ribbon (like an 
    eight)."""

    def __init__(self, step_size=1, sigma_noise=.05, seed=None):
        self.step_size = step_size
        self.sigma_noise = sigma_noise
        self.phi = 0
        super(EnvRibbon, self).__init__(ndim = 2,
                                        initial_state = np.array([np.cos(self.phi),np.sin(2*self.phi)]),
                                        actions_dict = {0: 'NONE'},
                                        noisy_dim_dist = environment.Noise.uniform,
                                        seed=seed)
        return
    
    
    def _render(self, phi):
        x = np.cos(phi) + self.sigma_noise * self.rnd.randn()
        y = np.sin(2*phi) + self.sigma_noise * self.rnd.randn()
        return np.array([x,y])
        
    
    def _do_action(self, action):
        """Walks one step along the ribbon.
        Returns new state and new angle.
        """
        
        if action == 0:
            self.phi += self.step_size
            self.phi = self.phi % (2 * np.pi)
        else:
            assert False

        return self._render(self.phi), self.phi


if __name__ == '__main__':
    
    # sample data
    steps = 1000
    env = EnvRibbon(step_size=1)
    data, actions, _ = env.do_random_steps(num_steps=steps)
    
    print 'Possible actions:'
    for action, describtion in env.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
        
    # plot data
    plt.plot(data[:,0], data[:,1], '.')
    plt.show()
    