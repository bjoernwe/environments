# Classes and Functions marked with "MarioAI class" or "MarioAI function" are adopted from the MarioAI benchmark
# and are authored by Sergey Karakovskiy - sergey [at] idsia [fullstop] ch

import sys
import os
import numpy as np
import random
import environment
import itertools
import ctypes

import matplotlib.pyplot as plt



class EnvMarioRGB(environment.Environment):
    """An environment realizing an interface to the Mario AI Challenge.
    
    http://www.marioai.org
    
    To get this example running the variable 'path_to_mario' has to point to the
    directory that contains the MarioAI challenge. It expects the build script 
    in src/amico/python/PyJava to have run successfully and have placed its
    output in bin/AmiCoBuild/PyJava.
    
    You may also need to set the environment variable LD_LIBRARY_PATH to include
    libraries that are not found otherwise. For instance, with our current Linux
    Mint configuration we have to point to /usr/lib/jvm/default-java/jre/lib/amd64/server
    to be able to find the libjvm library.
    """
    path_to_mario = '../../../../MarioAI Java'
    
    def __init__(self, seed=None, background=True, greyscale=False):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        seed:         int
        greyscale:    boolean - indicates if image data is converted to RGB or greyscale values
        """
        
        self.greyscale              = greyscale
        
        super(EnvMarioRGB, self).__init__(ndim=320*240,
                                          initial_state = None, 
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
        
        # not yet initialized cfuncs
#         self.getObservationDetails  = None
#         self.getEntireObservation   = None
#         self.performAction          = None
        
        self.libamico               = None
        
        # Status information about the agent
#         self.isEpisodeOver          = False
#         self.marioFloats            = None 
#         self.enemiesFloats          = None
#         self.mayMarioJump           = None
#         self.isMarioOnGround        = None
#         self.marioState             = None
#         self.levelScene             = None
#         self.enemiesScene           = None
        
#         self.receptiveFieldHeight   = 19
#         self.receptiveFieldWidth    = 19
#         self.action                 = [0, 0, 0, 0, 0, 0]
#         self.current_reward         = 0
#         self.previous_reward        = 0
#         self.last_reward            = 0
        
        #self.actions               = ['RIGHT', 'LEFT', 'RUN_RIGHT', 'RUN_LEFT', 'JUMP', 'STAND', 'DUCK', 'JUMP_LEFT' 
        #                              'JUMP_RIGHT', 'RUN/SHOOT', 'JUMP/RUN']
        #self.actions                = ['RIGHT', 'JUMP', 'STAND', 'DUCK', 'RUN_RIGHT', 'JUMP_RIGHT', 'RUN/SHOOT', 'JUMP/RUN']
        
        
#         self.current_state          = []
#         self.mario_labels           = []
#         self.levelScene_labels      = []
#         self.enemies_labels         = []
        
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
            #print libamico
        else: #else if OS is a Mac OS X (libAmiCo.dylib is searched for) or Windows (AmiCo.dll)
            name =  'AmiCoPyJava'
            loadName = ctypes.util.find_library(name)
            print loadName
            self.libamico = ctypes.CDLL(loadName)
            #print libamico
            
        #self.libamico = libamico
    
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
        
        self.last_reward            = self.getIntermediateReward()
    
#         options = ""
#         if len(sys.argv) > 1:
#             options = sys.argv[1]
# 
#         if options.startswith('"') and options.endswith('"'):
#             options = options[1:-1]
#             
#         self.options1               = options + " -ls "
#         options1                    = self.options1 #+ str(self.seed)
# 
#         print options1
        self.reset('')
        # make first step to complete initialization
        #obsDetails = self.getObservationDetails()
        #self._setObservationDetails(obsDetails[0], obsDetails[1], obsDetails[2], obsDetails[3])
        #obs = self.getEntireObservation(1, 0)
        #self._integrateObservation(obs[0], obs[1], obs[2], obs[3], obs[4])
        self.current_state = self._transformImageToRGB()
        self.ndim = self.current_state.shape[0]
        print self.current_state.shape
        
        self.libamico.tick()
            
        return

    
    def _translate(self, action):
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
    
    
    def _calculateReward(self):
        """Return difference between current and previous reward
        --------------------------------------
        Return:
        self.last_reward:    int
        """
        
        return self.getIntermediateReward()
        self.current_reward     = self.getIntermediateReward()
        self.last_reward        = self.current_reward - self.previous_reward
        self.previous_reward    = self.current_reward
        
        return self.last_reward
            
    
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
            options1 = self.options1# + str(seed)
            self.reset(options1)

        assert action is not None
        #if action == None:
        #    choice = random.randint(0,len(self.actions)-1)
        #    action = self.actions[choice]
        
        # translate action into list which can be processed by self.performAction,
        # perform the translated action and advance the environment by 1 tick
        translatedAction = self._translate(action)
        self.performAction(translatedAction)
        
        self.libamico.tick()
        #obsDetails = self.getObservationDetails()
        #self._setObservationDetails(obsDetails[0], obsDetails[1], obsDetails[2], obsDetails[3])
        #obs = self.getEntireObservation(1, 0)
        
        # self._integrateObservation also stores the data of the observation in self.current_state
        #self._integrateObservation(obs[0], obs[1], obs[2], obs[3], obs[4])
        self.current_state = self._transformImageToRGB()
        reward = self.getIntermediateReward()
        return (self.current_state, reward)
    
    
    def _transformImageToRGB(self):
        """Return numpy array with RGB or greyscale values of current screen
        --------------------------------------
        Return:
        self.current_state:    np.ndarray - stores the RGB values of the current snapshot
        """
        RGB = self.getVisualRGB()
        N = len(RGB)
        
        if self.greyscale == True:
            self.current_state = np.zeros(N)
            for i in range(N):
                self.current_state[i] = (int(0.21 * ((RGB[i] >> 16) & 0xFF)
                                           + 0.71 * ((RGB[i] >>  8) & 0xFF)
                                           + 0.07 * ((RGB[i] >>  0) & 0xFF)))
   
        else:
            self.current_state = np.zeros(3*N)
            for i in range(N):
                j = 3*i
                self.current_state[j+0] = ((RGB[i] >> 16) & 0xFF)
                self.current_state[j+1] = ((RGB[i] >>  8) & 0xFF)
                self.current_state[j+2] = ((RGB[i] <<  0) & 0xFF)
            
        return self.current_state
    

    # MarioAI function
    #def _reset(self):
    #    self.action = [0, 0, 0, 0, 0, 0]
    #    self.isEpisodeOver = False
        
        
    # MarioAI function; adapted to suit the needs of the project
#     def _integrateObservation(self, squashedObservation, squashedEnemies, marioPos, enemiesPos, marioState):
#         """Store given observation in self.marioState as well as summarize all meaningful data in one
#         numpy array (self.current_state) fit for further processing
#         --------------------------------------
#         Parameters:
#         squashedObservation:    tuple
#         squashedEnemies:        tuple 
#         marioPos:               tuple - the agent's position in the environment
#         enemiesPos:             tuple - the nearest enemies' position and velocity in the environment
#         marioState:             tuple - summarization of important data about the agent
#         """
#         row = self.receptiveFieldHeight
#         col = self.receptiveFieldWidth
#         
#         levelScene     = []
#         enemiesScene   = []
#         
#         for i in range(row):
#             levelScene.append(squashedObservation[i*col:i*col+col])
#             enemiesScene.append(squashedEnemies[i*col:i*col+col])
#                    
#         self.marioFloats        = marioPos
#         self.enemiesFloats      = enemiesPos
#         self.mayMarioJump       = marioState[3]
#         self.isMarioOnGround    = marioState[2]
#         self.marioState         = marioState[1]
#         self.levelScene         = levelScene
#         self.levelScene         = list(itertools.chain(*self.levelScene))
#         self.enemiesScene       = enemiesScene
#         self.enemiesScene       = list(itertools.chain(*self.enemiesScene))
#         
#         if self.ndim is not None:
#             self.mario_labels.append(self.marioFloats)
#             self.levelScene_labels.append(self.levelScene)
#             self.enemies_labels.append(self.enemiesScene)
#         
#         self.current_state = self._transformImageToRGB()
# 
# 
#     # MarioAI function 
#     def _setObservationDetails(self, rfWidth, rfHeight, egoRow, egoCol):
#         self.receptiveFieldWidth    = rfWidth
#         self.receptiveFieldHeight   = rfHeight
#         self.marioEgoRow            = egoRow
#         self.marioEgoCol            = egoCol
#         
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
 
# # MarioAI class
# class ListByRef(object):
#     """An argument that converts a list/tuple of ctype elements into a pointer to an array of pointers to the elements
#     """
#     def __init__(self, etype):
#         self.etype = etype
#         self.etype_p = ctypes.POINTER(etype)
# 
#     def from_param(self, param):
#         
#         if isinstance(param, (list, tuple)):
#             val = (self.etype_p * len(param))()
#             
#             for i, v in enumerate(param):
#                 if isinstance(v, self.etype):
#                     val[i] = self.etype_p(v)
#                 else:
#                     val[i] = v
#                     
#             return val
#         
#         else:
#             return param

# MarioAI class
# class Inspectable(object):
#     """ All derived classes gains the ability to print the names and values of all their fields"""
#     def __repr__(self):
#         return '<%s: %s>' % (self.__class__.__name__,
#             dict([(x,y) for (x,y) in self.__dict__.items() if not x.startswith('_')]) )

# MarioAI class
# class EvaluationInfo(Inspectable):
#     def __init__(self, evInfo):
#         print "widthCells = ", evInfo[0]
#         print "widthPhys  = ", evInfo[1]
#         print "flowersDevoured = ", evInfo[2]
#         print "killsByFire = ", evInfo[3]
#         print "killsByShell = ", evInfo[4]
#         print "killsByStomp = ",  evInfo[5]
#         print "killsTotal = ", evInfo[6]
#         print "marioMode = ", evInfo[7]
#         print "marioStatus = ", evInfo[8]
#         print "mushroomsDevoured = ", evInfo[9]
#         print "marioCoinsGained = ", evInfo[10]
#         print "timeLeft = ", evInfo[11]
#         print "timeSpent = ", evInfo[12]
#         print "hiddenBlocksFound = ", evInfo[13]
# 
#         self.widthCells         = evInfo[0]
#         self.widthPhys          = evInfo[1]
#         self.flowersDevoured    = evInfo[2]
#         self.killsByFire        = evInfo[3]
#         self.killsByShell       = evInfo[4]
#         self.killsByStomp       = evInfo[5]
#         self.killsTotal         = evInfo[6]
#         self.marioMode          = evInfo[7]
#         self.marioStatus        = evInfo[8]
#         self.mushroomsDevoured  = evInfo[9]
#         self.marioCoinsGained   = evInfo[10]
#         self.timeLeft           = evInfo[11]
#         self.timeSpent          = evInfo[12]
#         self.hiddenBlocksFound  = evInfo[13]

# MarioAI function
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
    env = EnvMarioRGB(background=True, greyscale=True)
    img = env.generate_training_data(num_steps=10, whitening=False)[0][0][-1].reshape((240, 320))
    plt.imshow(img)
    plt.show()
    
