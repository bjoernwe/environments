import joblib
import mdp
import os
import numpy as np

from enum import Enum

import environment


class EnvData(environment.Environment):
    """
    An environment that serves as a unified interface to different static
    data sets.
    """

    Datasets = Enum('Datasets', 'EEG EEG2 EEG2_stft_128 MEG WAV_11k WAV_22k WAV2_22k WAV3_22k WAV4_22k')
    

    def __init__(self, dataset, cachedir=None, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        """
        if dataset == self.Datasets.EEG:
            self.data = np.load(os.path.dirname(__file__) + '/eeg.npy')
        elif dataset == self.Datasets.EEG2:
            # http://bbci.de/competition/iv/
            self.data = np.load(os.path.dirname(__file__) + '/eeg2.npy')
        elif dataset == self.Datasets.EEG2_stft_128:
            #self.data = np.load(os.path.dirname(__file__) + '/eeg2_stft_128.npy')
            self.data = np.memmap(filename=os.path.dirname(__file__) + '/eeg2_stft_128.mm', mode='r', dtype=np.float32, shape=(29783, 7611))
        elif dataset == self.Datasets.MEG:
            self.data = np.load(os.path.dirname(__file__) + '/meg.npy') * 1e10
        elif dataset == self.Datasets.WAV_11k:
            self.data = np.load(os.path.dirname(__file__) + '/wav_11k.npy')
        elif dataset == self.Datasets.WAV_22k:
            self.data = np.load(os.path.dirname(__file__) + '/wav_22k.npy')
        elif dataset == self.Datasets.WAV2_22k:
            # https://www.freesound.org/people/Luftrum/sounds/48411/
            self.data = np.load(os.path.dirname(__file__) + '/wav2_22k.npy')
        elif dataset == self.Datasets.WAV3_22k:
            # https://www.freesound.org/people/Leandros.Ntounis/sounds/163995/
            self.data = np.load(os.path.dirname(__file__) + '/wav3_22k.npy')
        elif dataset == self.Datasets.WAV4_22k:
            # https://www.freesound.org/people/inchadney/sounds/66785/
            self.data = np.load(os.path.dirname(__file__) + '/wav4_22k.npy')
        else:
            print dataset
            assert False

        self.counter = 0
        super(EnvData, self).__init__(ndim = self.data.shape[1],
                                      initial_state = self.data[0],
                                      noisy_dim_dist = environment.Noise.normal,
                                      cachedir=cachedir,
                                      seed=seed)
        
        if cachedir is not None:
            if seed is None:
                print('Warning: cachedir should be used in combination with a fixed seed!')
            mem = joblib.Memory(cachedir=cachedir)
            self.generate_training_data = mem.cache(self.generate_training_data)
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



    def generate_training_data(self, num_steps, num_steps_test, actions=None, noisy_dims=0, keep_variance=1., expansion=1, whitening=True, n_chunks=1):
        """
        Generates a list of data chunks. Each chunks is a 3-tuple of generated
        data, corresponding actions and reward values/labels. PCA (keep_variance) 
        and whitening are calculated from the first chunk only.
        """
        
        assert n_chunks <= 2
        chunks = []
        
        # data
        self.counter = self.rnd.randint(0, self.data.shape[0]-num_steps+1) - 1
        counter2 = range(-1, self.counter-num_steps_test+1)
        data, actions, rewards = self.do_actions(actions=actions, num_steps=num_steps)
        counter2 += range(self.counter, self.data.shape[0]-num_steps_test)
        if data.ndim == 1:
            data = np.array(data, ndmin=2).T 
        chunks.append((data, actions, rewards))

        # data test        
        if n_chunks == 2:
            N = num_steps_test if num_steps_test else num_steps
            self.counter = self.rnd.choice(counter2)
            data, actions, rewards = self.do_actions(actions=actions, num_steps=N)
            if data.ndim == 1:
                data = np.array(data, ndmin=2).T 
            chunks.append((data, actions, rewards))
            
        # PCA
#         if keep_variance < 1.:
#             pca = mdp.nodes.PCANode(output_dim=keep_variance, reduce=True)
#             if chunks[0][0].shape[1] <= chunks[0][0].shape[0]:
#                 pca.train(chunks[0][0])
#                 chunks = [(pca.execute(data), actions, rewards) for (data, actions, rewards) in chunks]
#             else:
#                 pca.train(chunks[0][0].T)
#                 pca.stop_training()
#                 U = chunks[0][0].T.dot(pca.v)
#                 chunks = [(data.dot(U), actions, rewards) for (data, actions, rewards) in chunks]
            
        # expansion
        if expansion > 1:
            expansion_node = mdp.nodes.PolynomialExpansionNode(degree=expansion)
            chunks = [(expansion_node.execute(data), actions, rewards) for (data, actions, rewards) in chunks]
            if keep_variance < 1.:
                pca = mdp.nodes.PCANode(output_dim=keep_variance, reduce=True)
                if chunks[0][0].shape[1] <= chunks[0][0].shape[0]:
                    pca.train(chunks[0][0])
                    chunks = [(pca.execute(data), actions, rewards) for (data, actions, rewards) in chunks]
                else:
                    pca.train(chunks[0][0].T)
                    pca.stop_training()
                    U = chunks[0][0].T.dot(pca.v)
                    chunks = [(data.dot(U), actions, rewards) for (data, actions, rewards) in chunks]

        # whitening
        if whitening:
            whitening_node = mdp.nodes.WhiteningNode(reduce=True)
            whitening_node.train(chunks[0][0])
            chunks = [(whitening_node.execute(data), actions, rewards) for (data, actions, rewards) in chunks]
    
        return chunks
    


def main():
    for dat in EnvData.Datasets:
        env = EnvData(dataset=dat)
        print "%s: %d frames with %d dimensions" % (dat, env.data.shape[0], env.data.shape[1])
        #chunks = env.generate_training_data(num_steps=10, num_steps_test=5, n_chunks=2)
        
        

def create_stfts():
    import scipy.io.wavfile
    import stft
    wavs = [('wav_11k.npy', 'Wagon Wheel 11k.wav'),
            ('wav_22k.npy', 'Wagon Wheel 22k.wav'),
            ('wav2_22k.npy', '48411_luftrum_forestsurroundings_22k.wav'),
            ('wav3_22k.npy', '163995_leandros-ntounis_crowd-in-a-bar-lcr.wav'),
            ('wav4_22k.npy', '66785_inchadney_morning-in-the-forest_22k.wav'),
            ]
    for filename_out, filename_in in wavs:
        wav = scipy.io.wavfile.read('/home/weghebvc/Download/%s' % filename_in)[1]
        ft = stft.spectrogram(wav, framelength=512).T
        ft = np.hstack([ft.real, ft[:,1:-1].imag])
        print filename_out, ft.shape
        np.save(filename_out, ft)
        
        
        
def create_eeg1():
    #import urllib
    #urllib.urlretrieve("https://www.kaggle.com/c/grasp-and-lift-eeg-detection/download/train.zip", "train.zip")
    #from subprocess import call
    #call(["unzip", "train.zip"])
    import csv
    with open('subj1_series1_data.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            print ', '.join(row)
        
        
        
def plot_pca(dataset):
    import matplotlib.pyplot as plt
    data = EnvData(dataset=dataset).data
    C = np.cov(data[:50000].T)
    print C.shape
    E, _ = np.linalg.eigh(C)
    plt.plot(E)
    plt.show()



if __name__ == '__main__':
    main()
    #create_stfts()
    #create_eeg1()
    #plot_pca(EnvData.Datasets.WAV_22k)
    #plot_pca(EnvData.Datasets.WAV3_22k)
    #plot_pca(EnvData.Datasets.WAV4_22k)
