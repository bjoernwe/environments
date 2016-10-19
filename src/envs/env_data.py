import joblib
import mdp
import os
import numpy as np

from enum import Enum

import environment


Datasets = Enum('Datasets', 'EEG EEG2 EEG2_stft_128 HAPT MEG STFT1 STFT2 STFT3')


class EnvData(environment.Environment):
    """
    An environment that serves as a unified interface to different static
    data sets.
    """

    def __init__(self, dataset, time_embedding=1, cachedir=None, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        """
        if dataset == Datasets.EEG:
            self.data = np.load(os.path.dirname(__file__) + '/data_eeg.npy')
        elif dataset == Datasets.EEG2:
            # http://bbci.de/competition/iv/
            self.data = np.load(os.path.dirname(__file__) + '/data_eeg2.npy')
        elif dataset == Datasets.EEG2_stft_128:
            self.data = np.load(os.path.dirname(__file__) + '/data_eeg2_stft_128.npy')
            #self.data = np.memmap(filename=os.path.dirname(__file__) + '/data_eeg2_stft_128.mm', mode='r', dtype=np.float32, shape=(29783, 7611))
        elif dataset == Datasets.HAPT:
            # http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
            self.data   = np.load(os.path.dirname(__file__) + '/data_hapt.npy')
            self.labels = np.load(os.path.dirname(__file__) + '/data_hapt_labels.npy')
        elif dataset == Datasets.MEG:
            self.data = np.load(os.path.dirname(__file__) + '/data_meg.npy') * 1e10
        elif dataset == Datasets.STFT1:
            # https://www.freesound.org/people/Luftrum/sounds/48411/
            self.data = np.load(os.path.dirname(__file__) + '/data_stft1.npy')
        elif dataset == Datasets.STFT2:
            # https://www.freesound.org/people/Leandros.Ntounis/sounds/163995/
            self.data = np.load(os.path.dirname(__file__) + '/data_stft2.npy')
        elif dataset == Datasets.STFT3:
            # https://www.freesound.org/people/inchadney/sounds/66785/
            self.data = np.load(os.path.dirname(__file__) + '/data_stft3.npy')
        else:
            print dataset
            assert False

        self.counter = 0
        super(EnvData, self).__init__(ndim = self.data.shape[1],
                                      initial_state = self.data[0],
                                      time_embedding = time_embedding,
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
            current_state = self.data[self.counter]
        else:
            print 'Warning: Not more than %d video frames available (%d)!' % (frames, self.counter) 
            current_state = np.zeros(dims)
        return current_state, 0



    def generate_training_data(self, n_train, n_test, n_validation=None, actions=None, noisy_dims=0, pca=1., pca_after_expansion=1., expansion=1, whitening=True):
        """
        Generates [training, test] or [training, test, validation] data as a 
        3-tuple each. Each tuple contains data, corresponding actions and reward 
        values/labels. PCA and whitening are trained from the first training
        data only.
        """
        
        if n_validation:
            print "Don't have validation sets for data files yet."
            assert n_validation is None
            
        results = [] # [tuple_train, tuple_test, tuple_validation]
        
        # data
        self.counter = self.rnd.randint(0, self.data.shape[0]-n_train+1) - 1    # -1 because counter is incremented immediately
        counter2 = range(-1, self.counter-n_test+1)
        data, actions, rewards = self.do_actions(actions=actions, num_steps=n_train)
        counter2 += range(self.counter, self.data.shape[0]-n_test)
        if data.ndim == 1:
            data = np.array(data, ndmin=2, dtype=data.dtype).T 
        results.append((data, actions, rewards))

        # data test        
        if n_test > 0:
            self.counter = self.rnd.choice(counter2)
            data, actions, rewards = self.do_actions(actions=actions, num_steps=n_test)
            if data.ndim == 1:
                data = np.array(data, ndmin=2).T 
            results.append((data, actions, rewards))
        else:
            results.append((None, None, None))
            
        # PCA
        if pca < 1.:
            pca_node = mdp.nodes.PCANode(output_dim=pca, reduce=True)
            if results[0][0].shape[1] <= results[0][0].shape[0]:
                pca_node.train(results[0][0])
                results = [(pca_node.execute(data), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in results]
            else:
                pca_node.train(results[0][0].T)
                pca_node.stop_training()
                U = results[0][0].T.dot(pca_node.v)
                results = [(data.dot(U), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in results]
            
        # expansion
        if expansion > 1:
            expansion_node = mdp.nodes.PolynomialExpansionNode(degree=expansion)
            results = [(expansion_node.execute(data), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in results]
            if pca_after_expansion < 1.:
                pca_node = mdp.nodes.PCANode(output_dim=pca_after_expansion, reduce=True)
                if results[0][0].shape[1] <= results[0][0].shape[0]:
                    pca_node.train(results[0][0])
                    results = [(pca_node.execute(data), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in results]
                else:
                    pca_node.train(results[0][0].T)
                    pca_node.stop_training()
                    U = results[0][0].T.dot(pca_node.v)
                    results = [(data.dot(U), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in results]

        # whitening
        if whitening:
            whitening_node = mdp.nodes.WhiteningNode(reduce=True)
            whitening_node.train(results[0][0])
            results = [(whitening_node.execute(data), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in results]

        # replace (None, None, None) tuples by single None value
        results = [(data, actions, rewards) if data is not None else None for (data, actions, rewards) in results]
        results.append((None)) # validation set
        assert len(results) == 3
        return results
    


def main():
    for dat in Datasets:
        env = EnvData(dataset=dat)
        print "%s: %d frames with %d dimensions" % (dat, env.data.shape[0], env.data.shape[1])
        #chunks = env.generate_training_data(num_steps=10, num_steps_test=5, n_chunks=2)
        
        

def create_stfts():
    import scipy.io.wavfile
    import stft
    wavs = [('data_stft1.npy', 'Wagon Wheel 22k.wav'),
            ('data_stft2.npy', '163995_leandros-ntounis_crowd-in-a-bar-lcr.wav'),
            ('data_stft3.npy', '66785_inchadney_morning-in-the-forest_22k.wav'),
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
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            print ', '.join(row)
            
            
            
def create_hapt():
    # http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
    import csv
    data = []
    for csvfilename in ['/home/weghebvc/Download/HAPT Data Set/Train/X_train.txt', '/home/weghebvc/Download/HAPT Data Set/Test/X_test.txt']:
        with open(csvfilename, 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ')
            for row in csvreader:
                data.append([float(r) for r in row])
    data = np.array(data, dtype=np.float16)
    print data.shape
    labels = []
    for csvfilename in ['/home/weghebvc/Download/HAPT Data Set/Train/y_train.txt', '/home/weghebvc/Download/HAPT Data Set/Test/y_test.txt']:
        with open(csvfilename, 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ')
            for row in csvreader:
                labels.append([float(r) for r in row])
    labels = np.array(labels, dtype=np.int8)[:,0]
    print labels.shape
    assert data.shape[0] == labels.shape[0]
    np.save('data_hapt.npy', data)
    np.save('data_hapt_labels.npy', labels)
    
    
    
def create_physionet1():
    import wfdb
    dbnames = [#'202_38w0d', 
               #'ucddb002',
               #'mgh002',
               'ice001_l_1of1']
    for db in dbnames:
        dat = np.array(wfdb.rdsamp('/home/bjoern/Downloads/' + db)[0], dtype=np.float16)
        np.save('/home/bjoern/Downloads/' + db + '.npy', dat)
    
    
    
def create_physionet2():
    import csv
    data = []
    with open('/home/bjoern/Downloads/n1_data.txt', 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        csvreader.next()
        counter = 0
        for row in csvreader:
            counter += 1
            if counter % 1000 == 0:
                print counter
            data.append([float(r) if len(r) else np.nan for r in row])
    data = np.array(data, dtype=np.float16)
    print data.shape
    np.save('/home/bjoern/Downloads/n1.npy', data)
    
        
        
def plot_pca(dataset):
    import matplotlib.pyplot as plt
    data = EnvData(dataset=dataset).data
    C = np.cov(data[:50000].T)
    print C.shape
    E, _ = np.linalg.eigh(C)
    plt.plot(E)
    plt.show()



if __name__ == '__main__':
    #main()
    #create_stfts()
    #create_eeg1()
    #create_hapt()
    create_physionet1()
    #create_physionet2()
    #plot_pca(EnvData.Datasets.WAV_22k)
    #plot_pca(EnvData.Datasets.WAV3_22k)
    #plot_pca(EnvData.Datasets.WAV4_22k)
