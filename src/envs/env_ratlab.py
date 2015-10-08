import os
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation

import environment


class EnvRatlab(environment.Environment):
    """Returns a video from ratlab (320x40=12800 pixels, 5000 frames).
    """

    def __init__(self, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        seed:        int - 
        """
        self.video = np.load(os.path.dirname(__file__) + '/ratlab.npy')
        self.n_frames, self.ndim = self.video.shape
        super(EnvRatlab, self).__init__(ndim = self.ndim,
                                        initial_state = self.video[0],
                                        noisy_dim_dist = environment.Noise.normal,
                                        seed = seed)
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
        if self.counter < self.n_frames:
            self.current_state = self.video[self.counter]
        else:
            print 'Warning: Not more than %d video frames available (%d)!' % (self.n_frames, self.counter) 
            self.current_state = np.zeros(self.ndim)
        return self.current_state, 0



def main():

    nx = 40
    ny = 320
    env = EnvRatlab()
    
    print env.generate_training_data(num_steps=100, noisy_dims=2)

    fig = plt.figure()
    data = 255 - np.reshape(env.current_state, (nx, ny))
    im = plt.imshow(data, cmap='gist_gray_r', vmin=0, vmax=255)

    def init():
        im.set_data(np.zeros((nx, ny)))
    
    def animate(i):
        data = 255 - np.reshape(env.do_action()[0], (nx, ny))
        im.set_data(data)
        return im
    
    _ = animation.FuncAnimation(fig, animate, init_func=init, frames=nx * ny, interval=25)
    plt.show()
    


if __name__ == '__main__':
    main()
