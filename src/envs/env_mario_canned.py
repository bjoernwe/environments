import os
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation

import environment
from envs.env_mario_rgb import EnvMarioRGB


class EnvMarioCanned(environment.Environment):
    """Returns the video of Super Mario (240x320=76800 pixels, 3000 frames).
    """

    def __init__(self, window_only=False, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        seed:        int - 
        """
        self.video = np.load(os.path.dirname(__file__) + '/mario.npy')
        self.n_frames, _ = self.video.shape
        self.window_only = window_only
        if window_only:
            self.image_height = 20
            self.image_width  = 20
            self.window_mask = np.zeros((120, 160), dtype=bool)
            self.window_mask[50:70,70:90] = True
            self.window_mask = self.window_mask.flatten()
            initial_state = self.video[0,self.window_mask]
            assert len(initial_state) == 20*20
        else:
            self.image_height = 120
            self.image_width  = 160
            initial_state = self.video[0]
        self.counter = 0
        super(EnvMarioCanned, self).__init__(ndim = self.image_height * self.image_width,
                                      initial_state = initial_state,
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
        
        self.counter += 1
        if self.counter < self.n_frames:
            if self.window_only:
                self.current_state = self.video[self.counter, self.window_mask]
            else:
                self.current_state = self.video[self.counter]
        else:
            print 'Warning: Not more than %d video frames available (%d)!' % (self.n_frames, self.counter) 
            self.current_state = np.zeros(self.ndim)
        return self.current_state, 0



def generate_data(N=3000):
    env = EnvMarioRGB(grayscale=True, scaling=.5)
    data = env.generate_training_data(actions=[6,8,9,10], num_steps=N, noisy_dims=0, whitening=False, expansion=1, chunks=1)[0][0]
    np.save('mario.npy', data)

    
    
def main():

    env = EnvMarioCanned(window_only=True)
    nx, ny = env.image_height, env.image_width

    fig = plt.figure()
    data = np.reshape(env.current_state, (nx, ny))
    im = plt.imshow(data, cmap='gist_gray', vmin=0, vmax=255)

    def init():
        im.set_data(np.zeros((nx, ny)))
    
    def animate(i):
        data = np.reshape(env.do_action()[0], (nx, ny))
        im.set_data(data)
        return im
    
    _ = animation.FuncAnimation(fig, animate, init_func=init, frames=nx*ny, interval=25)
    plt.show()
    


if __name__ == '__main__':
    #generate_data()
    main()
