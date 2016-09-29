import collections
import enum
import joblib
import numpy as np

import mdp


Noise = enum.Enum('Noise', 'normal uniform binary')


class Environment(object):
    '''
    Base class for all environments.
    
    To implement a subclass you need to initialize the list of possible actions
    (integers) and set the environment to some initial state vector. Also, the 
    _do_action method has to be implemented by the subclass. If you need a 
    random generator, use self.rnd if possible because it has a seed.
    '''


    def __init__(self, ndim, initial_state, actions_dict={0: None}, noisy_dim_dist=Noise.uniform, cachedir=None, seed=None):
        """
        Initializes the environment including an initial state.
        """
        self.ndim = ndim
        self.actions_dict = actions_dict
        self.noisy_dim_dist = noisy_dim_dist
        self.current_state = initial_state
        self.last_action = None
        self.last_reward = None
        self.rnd = np.random.RandomState(seed)
        if cachedir is not None:
            if seed is None:
                print('Warning: cachedir should be used in combination with a fixed seed!')
            mem = joblib.Memory(cachedir=cachedir)
            self.generate_training_data = mem.cache(self.generate_training_data)
            
            
    def get_number_of_possible_actions(self):
        """
        Returns the number N for the possible actions 0, ..., N-1
        """
        return len(self.actions_dict)
    

    def get_actions_dict(self):
        """
        Returns a dictionary of actions
        """
        return self.actions_dict
    
    
    def get_current_state(self):
        """
        Returns the current state of the environment.
        """
        return self.current_state
    
    
    def get_last_action(self):
        """
        Returns the last action performed.
        """
        return self.last_action
    
    
    def get_last_reward(self):
        """
        Returns the last reward received in the previous step.
        """
        return self.last_reward
    
        
    def do_actions(self, actions=None, num_steps=1):
        """
        Performs random actions (given as list or one single integer) and 
        returns three results:
        1) a matrix containing the resulting states
        2) a vector of actions performed, one shorter than the state vector 
           because there is no action for the last state yet
        3) a vector containing the rewards received in each step
        """
        
        states = np.zeros((num_steps, self.ndim))
        states[0] = self.get_current_state()
        performed_actions = np.zeros(num_steps-1)
        rewards = np.zeros(num_steps-1)
        
        for i in range(num_steps-1):
            states[i+1], performed_actions[i], rewards[i] = self.do_action(action=actions)
            
        return (states, performed_actions, rewards)

    
    def do_action(self, action=None):
        """
        Performs the given action and returns the resulting state, the action
        and the received reward. If no action is given, a random action is
        selected. If a list of actions is give, one is randomly selected from
        the list.
        """
        
        # select random action
        if action is None:
            action = self.rnd.choice(self.actions_dict.keys())
        elif isinstance(action, collections.Iterable):
            action = self.rnd.choice(action)

        # perform action        
        self.current_state, reward = self._do_action(action=action)
        self.last_action = action
        self.last_reward = reward
        
        return self.current_state, self.last_action, self.last_reward
        

    def _do_action(self, action):
        """
        Performs the given action and returns the resulting state as well as 
        some reward value.
        """
        raise RuntimeError('method not implemented yet')
    
    
    def generate_training_data(self, num_steps, actions=None, noisy_dims=0, pca=1., pca_after_expansion=1., expansion=1, whitening=True, n_chunks=1):
        """
        Generates a list of data chunks. Each chunks is a 3-tuple of generated
        data, corresponding actions and reward values/labels. PCA (keep_variance) 
        and whitening are calculated from the first chunk only.
        """
        # rev: 4
        
        # for every chunk ...
        chunks = []
        for c in range(n_chunks):
            
            # number of steps
            N = num_steps
            if isinstance(num_steps, collections.Iterable):
                N = num_steps[c]

            # data
            data, actions, rewards = self.do_actions(actions=actions, num_steps=N)
            
            # make sure data has two dimensions
            if data.ndim == 1:
                data = np.array(data, ndmin=2).T 
    
            # add noisy dim
            for _ in range(noisy_dims):
                if self.noisy_dim_dist == Noise.normal:
                    noise = self.rnd.randn(N)
                elif self.noisy_dim_dist == Noise.uniform:
                    noise = self.rnd.rand(N)
                elif self.noisy_dim_dist == Noise.binary:
                    noise = self.rnd.randint(2, size=N)
                else:
                    print 'I do not understand noisy_dim_dist ==', self.noisy_dim_dist
                    assert False
                data = np.insert(data, data.shape[1], axis=1, values=noise)
    
            chunks.append((data, actions, rewards))
            
        # PCA
        if pca < 1.:
            pca = mdp.nodes.PCANode(output_dim=pca, reduce=True)
            if chunks[0][0].shape[1] <= chunks[0][0].shape[0]:
                pca.train(chunks[0][0])
                chunks = [(pca.execute(data), actions, rewards) for (data, actions, rewards) in chunks]
            else:
                pca.train(chunks[0][0].T)
                pca.stop_training()
                U = chunks[0][0].T.dot(pca.v)
                chunks = [(data.dot(U), actions, rewards) for (data, actions, rewards) in chunks]
            
        # expansion
        if expansion > 1:
            expansion_node = mdp.nodes.PolynomialExpansionNode(degree=expansion)
            chunks = [(expansion_node.execute(data), actions, rewards) for (data, actions, rewards) in chunks]
            if pca_after_expansion < 1.:
                pca = mdp.nodes.PCANode(output_dim=pca_after_expansion, reduce=True)
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
