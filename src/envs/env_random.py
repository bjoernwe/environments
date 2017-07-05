import matplotlib.pyplot as plt

import environment

from environment import Noise


class EnvRandom(environment.Environment):

    def __init__(self, ndim=1, noise_dist=Noise.normal, seed=None):
        """
        Initializes the environment including an initial state.
        """
        super(EnvRandom, self).__init__(ndim = ndim,
                                        initial_state = 0.0,
                                        noisy_dim_dist = Noise.normal,
                                        seed=seed)

        
    def _noise(self):
        """
        Creates new noise. 
        """
        noise = None
        if self.noisy_dim_dist == Noise.normal:
            noise = self.rnd.randn(self.ndim)
        elif self.noisy_dim_dist == Noise.uniform:
            noise = self.rnd.rand(self.ndim)
        elif self.noisy_dim_dist == Noise.binary:
            noise = self.rnd.randint(2, size=self.ndim)
        else:
            print 'I do not understand noisy_dim_dist ==', self.noisy_dim_dist
            assert False
        return noise
        

    def _do_action(self, action):
        """
        Returns new noise and a return value (zero).
        """
        return self._noise(), 0



if __name__ == '__main__':

    steps = 1000
    env = EnvRandom(ndim=2, seed=1)
    data, _, _ = env.generate_training_data(n_train=10, n_test=0)[0]
    
    print 'Possible actions:'
    for action, describtion in env.get_actions_dict().iteritems():
        print '  %2d = %s' % (action, describtion)
        
    # plot data
    plt.plot(data[:,0], data[:,1], '.')
    plt.show()
    