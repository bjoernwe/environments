# Classes and Functions marked with "MarioAI class" or "MarioAI function" are adopted from the MarioAI benchmark
# and are authored by Sergey Karakovskiy - sergey [at] idsia [fullstop] ch

import ctypes
import environment
import numpy as np
import os
import scipy.misc
import sys

import matplotlib.pyplot as plt
from matplotlib import animation



class EnvMarioRGB(environment.Environment):
    """An environment realizing an interface to the Mario AI Challenge.
    
    http://www.marioai.org
    
    To get this example running the variable 'path_to_mario' has to point to the
    directory that contains the MarioAI challenge. It expects the build script 
    in src/amico/python/PyJava to have run successfully and have placed its
    output in bin/AmiCoBuild/PyJava.
    
    You may also need to set the environment variable LD_LIBRARY_PATH to include
    libraries that are not found otherwise. For instance, with our current Linux
    Mint configuration we have to point to /usr/lib/jvm/java-7-openjdk-amd64/jre/lib/amd64/server
    to be able to find the libjvm library.
    """
    path_to_mario = '../../../../MarioAI Java'
    
    def __init__(self, seed=None, grayscale=False, scaling=1.):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        seed:         int
        greyscale:    boolean - indicates if image data is converted to RGB or greyscale values
        """
        
        self.grayscale = grayscale
        self.scaling = scaling
        self.image_width = 320  # may be updated when scaling
        self.image_height = 240 # may be updated when scaling
        
        # Initialization of the AmiCo Simulation is adopted from the MarioAI benchmark
        print "Py: AmiCo Simulation Started:"
        print "library found: "
        print "Platform: ", sys.platform
        ##########################################
        # find_library on Linux could only be used if your libAmiCoPyJava.so is
        # on system search path or path to the library is added in to LD_LIBRARY_PATH
        #
        ##########################################
        if (sys.platform == 'linux2'):
            loadName = self.path_to_mario + '/bin/AmiCoBuild/PyJava/libAmiCoPyJava.so'
            self.libamico = ctypes.CDLL(loadName)
        else: #else if OS is a Mac OS X (libAmiCo.dylib is searched for) or Windows (AmiCo.dll)
            name =  'AmiCoPyJava'
            loadName = ctypes.util.find_library(name)
            print loadName
            self.libamico = ctypes.CDLL(loadName)
    
        # create environment
        javaClass = "ch/idsia/benchmark/mario/environments/MarioEnvironment"
        self.libamico.amicoInitialize(1, "-Djava.class.path=" + self.path_to_mario + os.path.sep + "bin" + os.path.sep + ":jdom.jar")
        self.libamico.createMarioEnvironment(javaClass)
    
        # specify necessary cfuncs
        self.reset                  = cfunc('reset', self.libamico, None, ('list', ListPOINTER(ctypes.c_int), 1))
        self.isLevelFinished        = cfunc('isLevelFinished', self.libamico, ctypes.c_bool)
        self.getMarioStatus         = cfunc('getMarioStatus', self.libamico, ctypes.c_int)
        self.getEntireObservation   = cfunc('getEntireObservation', self.libamico, ctypes.py_object,
                                           ('list', ctypes.c_int, 1),
                                           ('zEnemies', ctypes.c_int, 1))
        self.performAction          = cfunc('performAction', self.libamico, None, ('list', ListPOINTER(ctypes.c_int), 1))
        self.getObservationDetails  = cfunc('getObservationDetails', self.libamico, ctypes.py_object)
        self.getVisualRGB           = cfunc('getVisualRGB', self.libamico, ctypes.py_object)
        self.getIntermediateReward  = cfunc('getIntermediateReward', self.libamico, ctypes.c_int)

        # initial state        
        self.reset('')
        self.libamico.tick()
        initial_state = self._transformImageToRGB()
        ndim = initial_state.shape[0]

        super(EnvMarioRGB, self).__init__(ndim=ndim,
                                          initial_state = initial_state, 
                                          actions_dict = {0: 'LEFT', 
                                                          1: 'RIGHT', 
                                                          2: 'JUMP', 
                                                          3: 'STAND', 
                                                          4: 'DUCK', 
                                                          5: 'RUN_LEFT',
                                                          6: 'RUN_RIGHT', 
                                                          7: 'JUMP_LEFT', 
                                                          8: 'JUMP_RIGHT', 
                                                          9: 'RUN_SHOOT', 
                                                          10: 'JUMP/RUN'}, 
                                          noisy_dim_dist=environment.Noise.uniform, 
                                          seed=seed)
        return

    
    def _translate_action(self, action):
        """Translate a given action into a list representation
        --------------------------------------
        Parameters:
        action:         str - action to be translated
        --------------------------------------
        Return:
        self.action:    list - translated action
        """
        
        if action == 0: #'LEFT':
            return [1, 0, 0, 0, 0, 0] 
        elif action == 1: #'RIGHT':
            return [0, 1, 0, 0, 0, 0] 
        elif action == 2: #'JUMP':
            return [0, 0, 0, 1, 0, 0] 
        elif action == 3: #'STAND':
            return [0, 0, 0, 0, 0, 1] 
        elif action == 4: #'DUCK':
            return [0, 0, 1, 0, 0, 0] 
        elif action == 5: #'RUN_LEFT':
            return [1, 0, 0, 0, 1, 0] 
        elif action == 6: #'RUN_RIGHT':
            return [0, 1, 0, 0, 1, 0] 
        elif action == 7: #'JUMP_LEFT':
            return [1, 0, 0, 1, 0, 0] 
        elif action == 8: #'JUMP_RIGHT':
            return [0, 1, 0, 1, 0, 0] 
        elif action == 9: #'RUN/SHOOT':
            return [0, 0, 0, 0, 1, 0] 
        elif action == 10: #'JUMP/RUN':
            return [0, 0, 0, 1, 1, 0] 
        
        assert False
    

    
    def _do_action(self, action):
        """Perform the given action and return it as well as the resulting state of the
        environment. If no action is given, a random action is performed.
        --------------------------------------
        Parameters:
        action:        str - action to be performed
        --------------------------------------
        Return:
        self.current_state:    np.ndarray - stores the RGB values of the current snapshot
        self.last_reward:      int - difference between current and previous reward
        """

        # reset Mario to start position if level is finished or Mario is dead
        if (self.isLevelFinished() == True or self.getMarioStatus() == 0):
            self.reset('-ls')

        # translate action into list which can be processed by self.performAction,
        # perform the translated action and advance the environment by 1 tick
        translatedAction = self._translate_action(action)
        self.performAction(translatedAction)
        self.libamico.tick()
        current_state = self._transformImageToRGB()
        #reward = self.getIntermediateReward()
        reward = self.getEntireObservation(1, 0)[2][1] # y-position
        return (current_state, reward)
    
    
    def _transformImageToRGB(self):
        """Return numpy array with RGB or greyscale values of current screen
        --------------------------------------
        Return:
        self.current_state:    np.ndarray - stores the RGB values of the current snapshot
        """
        RGB = np.array(self.getVisualRGB(), dtype=int)
        
        if self.grayscale == True:
            current_state = 0.21 * ((RGB >> 16) & 0xFF) \
                               + 0.71 * ((RGB >>  8) & 0xFF) \
                               + 0.08 * ((RGB >>  0) & 0xFF)
            if self.scaling != 1.:
                current_state = current_state.reshape((240,320))
                current_state = scipy.misc.imresize(current_state, size=self.scaling)
                self.image_height, self.image_width = current_state.shape
                current_state = current_state.flatten()
        else:
            N = len(RGB)
            current_state = np.zeros(3*N, dtype=int)
            current_state[0::3] = ((RGB >> 16) & 0xFF)
            current_state[1::3] = ((RGB >>  8) & 0xFF)
            current_state[2::3] = ((RGB)       & 0xFF)
            if self.scaling != 1.:
                current_state = current_state.reshape((240,320,3))
                current_state = scipy.misc.imresize(current_state, size=self.scaling)
                self.image_height, self.image_width, _ = current_state.shape
                current_state = current_state.flatten()
            
        return current_state
    

# MarioAI class        
class ListPOINTER(object):
    """Just like a POINTER but accept a list of ctype as an argument
    """
    def __init__(self, etype):
        self.etype = etype
 
    def from_param(self, param):
        if isinstance(param, (list, tuple)):
            return (self.etype * len(param))(*param)
         
        else:
            return param
 

# MarioAI function
def cfunc(name, dll, result, * args):
    '''build and apply a ctypes prototype complete with parameter flags'''
    atypes = []
    aflags = []
    
    for arg in args:
        atypes.append(arg[1])
        aflags.append((arg[2], arg[0]) + arg[3:])
        
    return ctypes.CFUNCTYPE(result, * atypes)((name, dll), tuple(aflags))


if __name__ == '__main__':

    env = EnvMarioRGB(grayscale=False, scaling=.5)
    nx, ny = env.image_height, env.image_width

    def transform(img, grayscale=False):
        if grayscale: 
            result = img.reshape((nx, ny))
        else:
            result = np.zeros((nx, ny, 3), dtype=float)
            result[:,:,0] = img[0::3].reshape((nx, ny)) / 255.
            result[:,:,1] = img[1::3].reshape((nx, ny)) / 255.
            result[:,:,2] = img[2::3].reshape((nx, ny)) / 255.
        return result

    fig = plt.figure()
    data = transform(env.current_state)
    im = plt.imshow(data)
    #im = plt.imshow(data, cmap='gist_gray', vmin=0, vmax=255)

    def init():
        im.set_data(np.zeros((nx, ny)))
    
    def animate(i):
        data = transform(env.do_action([6,8,9,10])[0])
        #print env.getEntireObservation(1,0)[2][1] # y-position
        #print env.getEntireObservation(1,0)[4] # mario status
        im.set_data(data)
        return im
    
    _ = animation.FuncAnimation(fig, animate, init_func=init, frames=nx*ny, interval=25*5)
    plt.show()
