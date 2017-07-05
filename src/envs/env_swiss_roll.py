import numpy as np
import matplotlib.pyplot as plt

import environment


class EnvSwissRoll(environment.Environment):

    fourpi = 4. * np.pi

    def __init__(self, sigma=0.5, seed=None):
        """
        Initializes the environment including an initial state.
        """
        self.sigma = sigma
        self.t = EnvSwissRoll.fourpi / 2.
        super(EnvSwissRoll, self).__init__(ndim = 2,
                                           initial_state = self._f(self.t),
                                           noisy_dim_dist = environment.Noise.uniform,
                                           seed = seed)
        #self.rnd = random.Random()
        #if seed is not None:
        #    self.rnd.seed(seed)
            
        #self.actions = None
        #self.current_state = self._f(self.t)
        
    
    @classmethod    
    def _f(cls, phi):
        """
        Maps an angle phi to x, y values of the swiss roll.
        """
        x = np.cos(phi)*(1-.7*phi/cls.fourpi)
        y = np.sin(phi)*(1-.7*phi/cls.fourpi)
        return np.array([x, y])
        

    def _do_action(self, action):
        """
        Performs an random step on the swiss roll and returns the new data value
        along with a reward/label value (in this case the angle on the spiral).
        """

        # random walk
        #self.t += self.t * self.rnd.gauss(mu=0, sigma=1) / self.fourpi
        self.t += self.sigma * self.rnd.normal()
        
        # bounds
        self.t = np.clip(self.t, 0, self.fourpi)
        
        # result
        return self._f(self.t), self.t



if __name__ == '__main__':

    steps = 1000
    env = EnvSwissRoll(sigma=0.5)
    data, actions, _ = env.do_random_steps(num_steps=steps)
    
    print 'Possible actions:'
    for action, describtion in env.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
        
    # plot data
    plt.plot(data[:,0], data[:,1], '.')
    plt.show()
    