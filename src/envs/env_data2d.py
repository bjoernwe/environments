import os
import numpy as np

from enum import Enum
from matplotlib import animation

import environment


class EnvData2D(environment.Environment):
    """
    """

    Datasets = Enum('Datasets', 'face mario ratlab')
    

    def __init__(self, dataset):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        """
        
        if dataset == self.Datasets.face:
            self.data = np.load(os.path.dirname(__file__) + '/faces.npy')
            self.image_shape = (28, 20)
        elif dataset == self.Datasets.mario:
            self.data = np.load(os.path.dirname(__file__) + '/mario.npy')
            self.image_shape = (120, 160)
        elif dataset == self.Datasets.ratlab:
            self.data = np.load(os.path.dirname(__file__) + '/ratlab.npy')
            self.image_shape = (40, 320)
        else:
            assert False

        assert self.image_shape[0] * self.image_shape[1] == self.data.shape[1]
            
        self.counter = 0
        super(EnvData2D, self).__init__(ndim = self.data.shape[1],
                                        initial_state = self.data[0],
                                        noisy_dim_dist = environment.Noise.normal)
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
        
        frames, dims = self.data.shape
        self.counter += 1
        if self.counter < frames:
            self.current_state = self.data[self.counter]
        else:
            print 'Warning: Not more than %d video frames available (%d)!' % (frames, self.counter) 
            self.current_state = np.zeros(dims)
        return self.current_state, 0
    
    
    
    def show_animation(self, invert=True):    
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
    
        if invert:
            data = 255 - np.reshape(self.current_state, self.image_shape)
        else:
            data = np.reshape(self.current_state, self.image_shape)
        im = plt.imshow(data, cmap='gist_gray_r', vmin=0, vmax=255)

        def init():
            im.set_data(np.zeros(self.image_shape))
    
        def animate(i):
            if invert:
                data = 255 - np.reshape(self.do_action()[0], self.image_shape)
            else:
                data = np.reshape(self.do_action()[0], self.image_shape)
            im.set_data(data)
            return im
    
        _ = animation.FuncAnimation(fig, animate, init_func=init, frames=self.data.shape[1], interval=25)
        plt.show()




def main():
    env = EnvData2D(dataset=EnvData2D.Datasets.mario)
    env.show_animation()



if __name__ == '__main__':
    main()
