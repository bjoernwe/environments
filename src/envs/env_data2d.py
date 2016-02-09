import os
import numpy as np
import scipy

from enum import Enum
from matplotlib import animation

import environment


class EnvData2D(environment.Environment):
    """
    """

    Datasets = Enum('Datasets', 'face mario ratlab tumor')
    

    def __init__(self, dataset, scaling=1.):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        """
        
        if dataset == self.Datasets.face:
            self.data_raw = np.load(os.path.dirname(__file__) + '/faces.npy')
            self.image_shape_raw = (28, 20)
        elif dataset == self.Datasets.mario:
            self.data_raw = np.load(os.path.dirname(__file__) + '/mario.npy')
            self.image_shape_raw = (120, 160)
        elif dataset == self.Datasets.ratlab:
            self.data_raw = np.load(os.path.dirname(__file__) + '/ratlab.npy')
            self.image_shape_raw = (40, 320)
        elif dataset == self.Datasets.tumor:
            self.data_raw = np.load(os.path.dirname(__file__) + '/tumor.npy')
            self.image_shape_raw = (300, 250)
        else:
            assert False

        assert self.image_shape_raw[0] * self.image_shape_raw[1] == self.data_raw.shape[1]
        
        # scale image
        self.scaling = scaling
        if scaling != 1.:
            scaled_rows = []
            for row in self.data_raw:
                scaled_image = scipy.misc.imresize(row.reshape(self.image_shape_raw), size=scaling)
                scaled_rows.append(scaled_image.flatten())
                self.image_shape = scaled_image.shape
            self.data = np.array(scaled_rows)
        else:
            self.data = self.data_raw
            self.image_shape = self.image_shape_raw
        
        assert self.image_shape[0] * self.image_shape[1] == self.data.shape[1]
        assert self.data.shape[0] == self.data_raw.shape[0]
            
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
    for dat in EnvData2D.Datasets:
        env = EnvData2D(dataset=dat)
        print "%s: %d frames with %d x %d = %d dimensions" % (dat, env.data.shape[0], env.image_shape[0], env.image_shape[1], env.data.shape[1])
    
    env = EnvData2D(dataset=EnvData2D.Datasets.mario, scaling=.5)
    env.show_animation()



if __name__ == '__main__':
    main()
