import os
import matplotlib.pyplot as plt
import numpy as np

import environment


class EnvEEG(environment.Environment):
    """Returns a video from ratlab (320x40=12800 pixels, 5000 frames).
    """

    def __init__(self, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        seed:        int - 
        """
        self.data = np.load(os.path.dirname(__file__) + '/eeg.npy')
        self.n_frames, self.ndim = self.data.shape
        super(EnvEEG, self).__init__(ndim = self.ndim,
                                     initial_state = self.data[0],
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
            self.current_state = self.data[self.counter]
        else:
            print 'Warning: Not more than %d data frames available (%d)!' % (self.n_frames, self.counter) 
            self.current_state = np.zeros(self.ndim)
        return self.current_state, 0



def convert_csv():
    import csv
    reader = csv.reader(open('subj1_series1_data.csv', 'rb'))
    rows = [row for row in reader]
    data = np.array(rows)
    data = np.array(data[1:,1:], dtype=int)
    np.save('eeg.npy', data)
    print('stored %dx%d data points in eeg.npy') % data.shape



def main():
    env = EnvEEG()
    data = env.do_actions(num_steps=2000)[0]
    plt.plot(data[:,0])
    plt.plot(data[:,1])
    plt.plot(data[:,2])
    plt.show()
    


if __name__ == '__main__':
    #convert_csv()
    main()
    