import os
import numpy as np
import mdp
import joblib
import scipy

from enum import Enum
from matplotlib import animation

import environment


Datasets = Enum('Datasets', 'Crowd1 Crowd2 Crowd3 Dancing Face GoProBike Mario Mouth RatLab SpaceInvaders Traffic Tumor')


class EnvData2D(environment.Environment):
    """
    An environment that serves as a unified interface to different static 2D
    data sets.
    """
    

    def __init__(self, dataset, window=None, scaling=1., time_embedding=1, limit_data=None, additive_noise=0.0, cachedir=None, seed=0):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        """
        
        self.labels = None
        if dataset == Datasets.Crowd1:
            self.data_raw = np.load(os.path.dirname(__file__) + '/../../datasets/data_crowd1.npy')
            self.image_shape_raw = (180, 320)
        elif dataset == Datasets.Crowd2:
            self.data_raw = np.load(os.path.dirname(__file__) + '/../../datasets/data_crowd2.npy')
            self.image_shape_raw = (180, 320)
        elif dataset == Datasets.Crowd3:
            self.data_raw = np.load(os.path.dirname(__file__) + '/../../datasets/data_crowd3.npy')
            self.image_shape_raw = (180, 320)
        elif dataset == Datasets.Dancing:
            self.data_raw = np.load(os.path.dirname(__file__) + '/../../datasets/data_dancing.npy')
            self.image_shape_raw = (180, 320)
        elif dataset == Datasets.Face:
            self.data_raw = np.load(os.path.dirname(__file__) + '/../../datasets/data_faces.npy')
            self.image_shape_raw = (28, 20)
        elif dataset == Datasets.GoProBike:
            self.data_raw = np.memmap(filename=os.path.dirname(__file__) + '/../../datasets/data_gopro_bike.mm',
                                      mode='r', shape=(30151, 14400))
            self.image_shape_raw = (90, 160)
        elif dataset == Datasets.Mario:
            #self.data_raw = np.load(os.path.dirname(__file__) + '/data_mario.npy')
            self.data_raw = np.memmap(filename=os.path.dirname(__file__) + '/../../datasets/data_mario.mm', dtype=np.uint8, mode='r', shape=(20001, 19200))
            self.image_shape_raw = (120, 160)
            self.labels = [None] + list(np.load(os.path.dirname(__file__) + '/../../datasets/data_mario_labels.npy'))
            assert self.data_raw.shape[0] == len(self.labels)
        elif dataset == Datasets.Mouth:
            self.data_raw = np.load(os.path.dirname(__file__) + '/../../datasets/data_mouth.npy')
            self.image_shape_raw = (35, 60)
        elif dataset == Datasets.RatLab:
            self.data_raw = np.load(os.path.dirname(__file__) + '/../../datasets/data_ratlab.npy')
            self.image_shape_raw = (40, 320)
        elif dataset == Datasets.SpaceInvaders:
            #self.data_raw = np.load(os.path.dirname(__file__) + '/data_space_invaders.npy')
            self.data_raw = np.memmap(filename=os.path.dirname(__file__) + '/../../datasets/data_space_invaders.mm', mode='r', shape=(19098, 4160))
            self.image_shape_raw = (52, 80)
        elif dataset == Datasets.Tumor:
            self.data_raw = np.load(os.path.dirname(__file__) + '/../../datasets/data_tumor.npy')
            self.image_shape_raw = (300, 250)
        elif dataset == Datasets.Traffic:
            self.data_raw = np.load(os.path.dirname(__file__) + '/../../datasets/data_traffic.npy')
            #self.data_raw = np.memmap(filename=os.path.dirname(__file__) + '/data_traffic.mm', mode='r', shape=(23435, 13500))
            self.image_shape_raw = (90, 150)
        else:
            assert False

        assert self.image_shape_raw[0] * self.image_shape_raw[1] == self.data_raw.shape[1]
        
        # limit length
        if limit_data:
            self.data_raw = self.data_raw[:limit_data]
            if self.labels is not None:
                self.labels = self.labels[:limit_data]

        # scale image
        self.window = window
        self.scaling = scaling
        if window is not None or scaling != 1.:
            new_rows = []
            for row in self.data_raw:
                image = row.reshape(self.image_shape_raw)
                if window is not None:
                    ((x1,y1),(x2,y2)) = window
                    image = image[x1:x2,y1:y2]
                if scaling != 1.:
                    image = scipy.misc.imresize(image, size=scaling)
                new_rows.append(image.flatten())
                self.image_shape = image.shape
            self.data = np.array(new_rows)
        else:
            self.data = self.data_raw
            self.image_shape = self.image_shape_raw
            
        assert self.image_shape[0] * self.image_shape[1] == self.data.shape[1]
        assert self.data.shape[0] == self.data_raw.shape[0]
            
        # additive noise
        self.additive_noise = additive_noise
        if self.additive_noise > 0.0:
            self.data += self.additive_noise * self.rnd.randn(self.data.shape)
        
        self.counter = 0
        super(EnvData2D, self).__init__(ndim = self.data.shape[1],
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
            if self.labels:
                last_reward = self.labels[self.counter]
            else:
                last_reward = 0
        else:
            print 'Warning: Not more than %d video frames available (%d)!' % (frames, self.counter) 
            current_state = np.zeros(dims)
            last_reward = None
        return current_state, last_reward
    
    
    
    def generate_training_data(self, n_train, n_test, n_validation=None, actions=None, noisy_dims=0, pca=1., pca_after_expansion=1., expansion=1, additive_noise=0, whitening=True):
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
        self.counter = self.rnd.randint(0, self.data.shape[0]-n_train-n_test+1) - 1    # -1 because counter is incremented immediately
        #counter2 = range(-1, self.counter-n_test+1)
        data, actions, rewards = self.do_actions(actions=actions, num_steps=n_train)
        #counter2 += range(self.counter, self.data.shape[0]-n_test)
        if data.ndim == 1:
            data = np.array(data, ndmin=2, dtype=data.dtype).T 
        results.append((data, actions, rewards))

        # data test        
        if n_test > 0:
            #self.counter = self.rnd.choice(counter2)
            data, actions, rewards = self.do_actions(actions=actions, num_steps=n_test)
            if data.ndim == 1:
                data = np.array(data, ndmin=2).T 
            results.append((data, actions, rewards))
        else:
            results.append((None, None, None))
            
        # PCA
        if pca != 1. or type(pca) != type(1.): # catch integer 1 (one dimension)
            pca_node = mdp.nodes.PCANode(output_dim=pca, reduce=True)
            if results[0][0].shape[1] <= results[0][0].shape[0]:
                pca_node.train(results[0][0])
                results = [(pca_node.execute(data), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in results]
                self.pca1 = pca_node
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
                pca_node_2 = mdp.nodes.PCANode(output_dim=pca_after_expansion, reduce=True)
                if results[0][0].shape[1] <= results[0][0].shape[0]:
                    pca_node_2.train(results[0][0])
                    results = [(pca_node_2.execute(data), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in results]
                    self.pca2 = pca_node_2
                else:
                    pca_node_2.train(results[0][0].T)
                    pca_node_2.stop_training()
                    U = results[0][0].T.dot(pca_node_2.v)
                    results = [(data.dot(U), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in results]

        # additive noise
        if additive_noise:
            noise_node = mdp.nodes.NoiseNode(noise_args=(0, additive_noise))
            results = [(noise_node.execute(data), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in results]

        # whitening
        if whitening:
            whitening_node = mdp.nodes.WhiteningNode(reduce=True)
            whitening_node.train(results[0][0])
            results = [(whitening_node.execute(data), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in results]
            self.whitening_node = whitening_node

        # replace (None, None, None) tuples by single None value
        results = [(data, actions, rewards) if data is not None else None for (data, actions, rewards) in results]
        results.append((None)) # validation set
        assert len(results) == 3
        return results
    


    def show_animation(self, invert=True):
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
    
        if invert:
            data = 255 - np.reshape(self.get_current_state(), self.image_shape)
        else:
            data = np.reshape(self.get_current_state(), self.image_shape)
        im = plt.imshow(data, cmap='gist_gray_r', vmin=0, vmax=255, interpolation='none')

        def init():
            im.set_data(np.zeros(self.image_shape))
    
        def animate(i):
            if invert:
                data = 255 - np.reshape(self.do_action()[0], self.image_shape)
            else:
                data = np.reshape(self.do_action()[0], self.image_shape)
            im.set_data(data)
            if i % 10 == 0:
                plt.gca().set_title(i)
            return im
    
        _ = animation.FuncAnimation(fig, animate, init_func=init, frames=self.data.shape[0], interval=250)
        plt.show()



def create_traffic_data():
    import scipy.misc
    import scipy.ndimage
    rows = []
    for i in range(1, 23436):
        image = scipy.ndimage.imread(fname='/home/weghebvc/Download/Urban1/image%06d.jpg' % i, flatten=True)
        image = scipy.misc.imresize(image, size=.25)
        rows.append(image.flatten())
        if i%100 == 0:
            print i
    X = np.vstack(rows)
    print X.shape
    print X.dtype
    np.save('traffic.npy', X)
    mm = np.memmap('traffic.mm', mode='w+', dtype=np.uint8, shape=X.shape)
    mm[:,:] = X[:,:]
    mm.flush()



def create_space_invader_data():
    import scipy.misc
    import scipy.ndimage
    rows = []
    for i in range(19098):
        image = scipy.ndimage.imread(fname='/home/weghebvc/workspace/PythonPlayground/src/img/%06d.png' % i, flatten=True)
        image = scipy.misc.imresize(image, size=.25)
        rows.append(image.flatten())
        if i%100 == 0:
            print i, image.shape
    X = np.vstack(rows)
    print X.shape
    print X.dtype
    np.save('space_invaders.npy', X)
    mm = np.memmap('space_invaders.mm', mode='w+', dtype=np.uint8, shape=X.shape)
    mm[:,:] = X[:,:]
    mm.flush()



def create_bike_data():
    import scipy.misc
    import scipy.ndimage
    rows = []
    for i in range(1, 30152):
        image = scipy.ndimage.imread(fname='/scratch/weghebvc/bike_video/%08d.png' % i, flatten=True)
        image = scipy.misc.imresize(image, size=.25)
        rows.append(image.flatten())
        if i%100 == 0:
            print i, image.shape
    X = np.vstack(rows)
    print X.shape
    print X.dtype
    np.save('gopro_bike.npy', X)
    mm = np.memmap('gopro_bike.mm', mode='w+', dtype=np.uint8, shape=X.shape)
    mm[:,:] = X[:,:]
    mm.flush()



def main():
    for dat in Datasets:
        env = EnvData2D(dataset=dat)
        print "%s: %d frames with %d x %d = %d dimensions" % (dat, env.data.shape[0], env.image_shape[0], env.image_shape[1], env.data.shape[1])
    #    env.show_animation()
    #env = EnvData2D(dataset=Datasets.Mario, scaling=1.)
    #env = EnvData2D(dataset=Datasets.Mario, scaling=(50,50), window=((0,20),(120,140)))
    #env = EnvData2D(dataset=Datasets.Mario, window=((50,70),(90,90)), scaling=1.)
    #env.show_animation()
    #env = EnvData2D(dataset=EnvData2D.Datasets.Mario, window=((76,76),(84,84)), scaling=1.)
    #env = EnvData2D(dataset=Datasets.SpaceInvaders, scaling=1)
    #env = EnvData2D(dataset=Datasets.SpaceInvaders, scaling=(50,50), window=((0,14),(52,66)))
    #env = EnvData2D(dataset=Datasets.SpaceInvaders, window=((16,30),(36,50)), scaling=1)
    #env.show_animation()
    #env = EnvData2D(dataset=EnvData2D.Datasets.SpaceInvaders, window=((22,36),(30,44)), scaling=1)
    #env = EnvData2D(dataset=Datasets.Traffic, scaling=1)
    #env = EnvData2D(dataset=Datasets.Traffic, scaling=(50,50), window=((0,30),(90,120)))
    #env = EnvData2D(dataset=Datasets.Traffic, window=((35,65),(55,85)), scaling=1)
    #env.show_animation()
    #env = EnvData2D(dataset=EnvData2D.Datasets.Traffic, window=((41,71),(49,79)), scaling=1)
    #env.show_animation()

    env = EnvData2D(dataset=Datasets.RatLab, window=((0, 140), (40, 180)), scaling=.5)
    env.show_animation()

    #env = EnvData2D(dataset=Datasets.GoProBike, scaling=1.)
    #env.show_animation()
    #env = EnvData2D(dataset=Datasets.GoProBike, window=((25,70),(45,90)), scaling=1.)
    #env = EnvData2D(dataset=Datasets.GoProBike, window=((70, 70), (90, 90)), scaling=1.)
    #env.show_animation()

    import matplotlib.pyplot as plt

    # w/o PCA
    #env = EnvData2D(dataset=Datasets.SpaceInvaders, window=((16,30),(36,50)), scaling=1)
    #env = EnvData2D(dataset=Datasets.Mario, window=((70, 70), (90, 90)), scaling=1.)
    #env = EnvData2D(dataset=Datasets.Traffic, window=((35, 65), (55, 85)), scaling=1)
    #(dat_train, _, _), _, _ = env.generate_training_data(n_train=10000, n_test=100, whitening=False)
    #plt.imshow(np.cov(dat_train.T), cmap=plt.get_cmap('Greys'))

    # with PCA
    #env = EnvData2D(dataset=Datasets.SpaceInvaders, window=((16, 30), (36, 50)), scaling=1)
    #env = EnvData2D(dataset=Datasets.Mario, window=((70, 70), (90, 90)), scaling=1.)
    #env = EnvData2D(dataset=Datasets.Traffic, window=((35, 65), (55, 85)), scaling=1)
    #(dat_train, _, _), _, _ = env.generate_training_data(n_train=10000, n_test=100, whitening=False, pca=.99)
    #dat_train = dat_train.dot(env.pca1.v.T)

    #plt.figure()
    #plt.imshow(np.cov(dat_train.T), cmap=plt.get_cmap('Greys'))

    # with PCA and whitening
    #env = EnvData2D(dataset=Datasets.SpaceInvaders, window=((16, 30), (36, 50)), scaling=1)
    #env = EnvData2D(dataset=Datasets.Mario, window=((70, 70), (90, 90)), scaling=1.)
    #env = EnvData2D(dataset=Datasets.Traffic, window=((35, 65), (55, 85)), scaling=1)
    #(dat_train, _, _), _, _ = env.generate_training_data(n_train=10000, n_test=100, whitening=True, pca=.99)
    #E, U = np.linalg.eigh(env.whitening_node.v)
    #Winv = U.dot(np.diag(E**-1)).dot(U.T)
    #dat_train = dat_train.dot(Winv.dot(env.pca1.v.T))

    #plt.figure()
    #plt.imshow(np.cov(dat_train.T), cmap=plt.get_cmap('Greys'))

    #plt.show()



if __name__ == '__main__':
    main()
    #create_traffic_data()
    #create_space_invader_data()
    #create_bike_data()
