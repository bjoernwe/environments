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


    def __init__(self, ndim, initial_state, actions_dict={0: None}, time_embedding=1, noisy_dim_dist=Noise.uniform, cachedir=None, seed=None):
        """
        Initializes the environment including an initial state.
        """
        self.ndim = ndim
        self.ndim_embedded = ndim * time_embedding
        self.actions_dict = actions_dict
        self.noisy_dim_dist = noisy_dim_dist
        #self.current_state = initial_state
        self.time_embedding = time_embedding
        self.last_states = [initial_state] * self.time_embedding
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
        #return self.current_state
        return np.hstack(self.last_states)
    
    
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
        
        states = np.zeros((num_steps, self.ndim_embedded))
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
        current_state, reward = self._do_action(action=action)
        self.last_action = action
        self.last_reward = reward
        
        # time embedding
        self.last_states = [current_state] + self.last_states[:-1]
        assert len(self.last_states) == self.time_embedding
        current_state_embedded = np.hstack(self.last_states)
        
        return current_state_embedded, self.last_action, self.last_reward
        

    def _do_action(self, action):
        """
        Performs the given action and returns the resulting state as well as 
        some reward value.
        """
        raise RuntimeError('method not implemented yet')
    
    
    def generate_training_data(self, n_train, n_test, n_validation=None, actions=None, noisy_dims=0, pca=1., pca_after_expansion=1., expansion=1, whitening=True):
        """
        Generates [training, test] or [training, test, validation] data as a 
        3-tuple each. Each tuple contains data, corresponding actions and reward 
        values/labels. PCA and whitening are trained from the first training
        data only.
        """
        
        # for every chunk ...
        chunks = []
        chunk_sizes = [n_train, n_test] + [n_validation if n_validation else None]
        for num_steps in chunk_sizes:

            if num_steps <= 0 or num_steps is None:
                
                chunks.append((None, None, None))
                
            else:

                # data
                data, actions, rewards = self.do_actions(actions=actions, num_steps=num_steps)
                
                # make sure data has two dimensions
                if data.ndim == 1:
                    data = np.array(data, ndmin=2).T 
        
                # add noisy dim
                for _ in range(noisy_dims):
                    if self.noisy_dim_dist == Noise.normal:
                        noise = self.rnd.randn(num_steps)
                    elif self.noisy_dim_dist == Noise.uniform:
                        noise = self.rnd.rand(num_steps)
                    elif self.noisy_dim_dist == Noise.binary:
                        noise = self.rnd.randint(2, size=num_steps)
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
                chunks = [(pca.execute(data), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in chunks]
            else:
                pca.train(chunks[0][0].T)
                pca.stop_training()
                U = chunks[0][0].T.dot(pca.v)
                chunks = [(data.dot(U), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in chunks]
            
        # expansion
        if expansion > 1:
            expansion_node = mdp.nodes.PolynomialExpansionNode(degree=expansion)
            chunks = [(expansion_node.execute(data), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in chunks]
            if pca_after_expansion < 1.:
                pca = mdp.nodes.PCANode(output_dim=pca_after_expansion, reduce=True)
                if chunks[0][0].shape[1] <= chunks[0][0].shape[0]:
                    pca.train(chunks[0][0])
                    chunks = [(pca.execute(data), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in chunks]
                else:
                    pca.train(chunks[0][0].T)
                    pca.stop_training()
                    U = chunks[0][0].T.dot(pca.v)
                    chunks = [(data.dot(U), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in chunks]

        # whitening
        if whitening:
            whitening_node = mdp.nodes.WhiteningNode(reduce=True)
            whitening_node.train(chunks[0][0])
            chunks = [(whitening_node.execute(data), actions, rewards) if data is not None else (None, None, None) for (data, actions, rewards) in chunks]
            
        # replace (None, None, None) tuples by single None value
        chunks = [(data, actions, rewards) if data is not None else None for (data, actions, rewards) in chunks]
        assert len(chunks) == 3
        return chunks
