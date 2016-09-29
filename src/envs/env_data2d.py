import os
import numpy as np
import mdp
import joblib
import scipy

from enum import Enum
from matplotlib import animation

import environment


class EnvData2D(environment.Environment):
    """
    An environment that serves as a unified interface to different static 2D
    data sets.
    """

    Datasets = Enum('Datasets', 'Crowd1 Crowd2 Crowd3 Dancing Face Mario Mouth RatLab SpaceInvaders Traffic Tumor')
    

    def __init__(self, dataset, window=None, scaling=1., additive_noise=0.0, cachedir=None, seed=0):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        """
        
        self.labels = None
        if dataset == self.Datasets.Crowd1:
            self.data_raw = np.load(os.path.dirname(__file__) + '/data_crowd1.npy')
            self.image_shape_raw = (180, 320)
        elif dataset == self.Datasets.Crowd2:
            self.data_raw = np.load(os.path.dirname(__file__) + '/data_crowd2.npy')
            self.image_shape_raw = (180, 320)
        elif dataset == self.Datasets.Crowd3:
            self.data_raw = np.load(os.path.dirname(__file__) + '/data_crowd3.npy')
            self.image_shape_raw = (180, 320)
        elif dataset == self.Datasets.Dancing:
            self.data_raw = np.load(os.path.dirname(__file__) + '/data_dancing.npy')
            self.image_shape_raw = (180, 320)
        elif dataset == self.Datasets.Face:
            self.data_raw = np.load(os.path.dirname(__file__) + '/data_faces.npy')
            self.image_shape_raw = (28, 20)
        elif dataset == self.Datasets.Mario:
            #self.data_raw = np.load(os.path.dirname(__file__) + '/data_mario.npy')
            self.data_raw = np.memmap(filename=os.path.dirname(__file__) + '/data_mario.mm', dtype=np.uint8, mode='r', shape=(20001, 19200))
            self.image_shape_raw = (120, 160)
            self.labels = [None] + list(np.load(os.path.dirname(__file__) + '/data_mario_labels.npy'))
            assert self.data_raw.shape[0] == len(self.labels)
        elif dataset == self.Datasets.Mouth:
            self.data_raw = np.load(os.path.dirname(__file__) + '/data_mouth.npy')
            self.image_shape_raw = (35, 60)
        elif dataset == self.Datasets.RatLab:
            self.data_raw = np.load(os.path.dirname(__file__) + '/data_ratlab.npy')
            self.image_shape_raw = (40, 320)
        elif dataset == self.Datasets.SpaceInvaders:
            #self.data_raw = np.load(os.path.dirname(__file__) + '/data_space_invaders.npy')
            self.data_raw = np.memmap(filename=os.path.dirname(__file__) + '/data_space_invaders.mm', mode='r', shape=(19098, 4160))
            self.image_shape_raw = (52, 80)
        elif dataset == self.Datasets.Tumor:
            self.data_raw = np.load(os.path.dirname(__file__) + '/data_tumor.npy')
            self.image_shape_raw = (300, 250)
        elif dataset == self.Datasets.Traffic:
            self.data_raw = np.load(os.path.dirname(__file__) + '/data_traffic.npy')
            #self.data_raw = np.memmap(filename=os.path.dirname(__file__) + '/data_traffic.mm', mode='r', shape=(23435, 13500))
            self.image_shape_raw = (90, 150)
        else:
            assert False

        assert self.image_shape_raw[0] * self.image_shape_raw[1] == self.data_raw.shape[1]
        
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
            if self.labels:
                self.last_reward = self.labels[self.counter]
            else:
                self.last_reward = 0
        else:
            print 'Warning: Not more than %d video frames available (%d)!' % (frames, self.counter) 
            self.current_state = np.zeros(dims)
            self.last_reward = None
        return self.current_state, self.last_reward
    
    
    
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
            data = np.array(data, ndmin=2, dtype=np.float32).T 
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
            if i % 10 == 0:
                plt.gca().set_title(i)
            return im
    
        _ = animation.FuncAnimation(fig, animate, init_func=init, frames=self.data.shape[1], interval=25)
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



def main():
    for dat in EnvData2D.Datasets:
        env = EnvData2D(dataset=dat)
        print "%s: %d frames with %d x %d = %d dimensions" % (dat, env.data.shape[0], env.image_shape[0], env.image_shape[1], env.data.shape[1])
    env = EnvData2D(dataset=EnvData2D.Datasets.Mario, scaling=1.)
    #env = EnvData2D(dataset=EnvData2D.Datasets.Mario, window=((70,70),(90,90)), scaling=1.)
    #env = EnvData2D(dataset=EnvData2D.Datasets.Mario, window=((76,76),(84,84)), scaling=1.)
    #env = EnvData2D(dataset=EnvData2D.Datasets.SpaceInvaders, scaling=1)
    #env = EnvData2D(dataset=EnvData2D.Datasets.SpaceInvaders, window=((16,30),(36,50)), scaling=1)
    #env = EnvData2D(dataset=EnvData2D.Datasets.SpaceInvaders, window=((22,36),(30,44)), scaling=1)
    #env = EnvData2D(dataset=EnvData2D.Datasets.Traffic, scaling=1)
    #env = EnvData2D(dataset=EnvData2D.Datasets.Traffic, window=((35,65),(55,85)), scaling=1)
    #env = EnvData2D(dataset=EnvData2D.Datasets.Traffic, window=((41,71),(49,79)), scaling=1)
    env.show_animation()



if __name__ == '__main__':
    main()
    #create_traffic_data()
    #create_space_invader_data()
