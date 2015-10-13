# Classes and Functions marked with "MarioAI class" or "MarioAI function" are adopted from the MarioAI benchmark
# and are authored by Sergey Karakovskiy - sergey [at] idsia [fullstop] ch

import sys
import os
import numpy as np
import random
import environment
import itertools
import ctypes


class EnvMario(environment.Environment):
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
    path_to_mario = '../../../MarioAI Java'
    
    def __init__(self, seed=None):
        """Initialize the environment.
        --------------------------------------
        Parameters:
        seed:         int
        """
        
        if seed == None:
            seed = random.randint(0,999999)
        random.seed(seed)

        super(EnvMario, self).__init__(seed=seed)
        
        # not yet initialized cfuncs
        self.getObservationDetails = None
        self.getEntireObservation = None
        self.performAction = None
        
        self.libamico = None
        
        # Status information about the agent
        self.isEpisodeOver = False
        self.marioFloats = None 
        self.enemiesFloats = None
        self.mayMarioJump = None
        self.isMarioOnGround = None
        self.marioState = None
        self.levelScene = None
        
        self.receptiveFieldHeight = 19
        self.receptiveFieldWidth = 19
        self.action = [0, 0, 0, 0, 0, 0]
        self.actions = ['RIGHT', 'LEFT', 'JUMP', 'STAND', 'DUCK', 'RUN_RIGHT', 
                        'RUN_LEFT', 'JUMP_RIGHT', 'JUMP_LEFT', 'RUN/SHOOT', 'JUMP/RUN']

        self.current_state = []
        
        # Initialization of the AmiCo Simulation is adopted from the MarioAI benchmark
        print "Py: AmiCo Simulation Started:"
        print "library found: "
        print "Platform: ", sys.platform
        if (sys.platform == 'linux2'):
        ##########################################
        # find_library on Linux could only be used if your libAmiCoPyJava.so is
        # on system search path or path to the library is added in to LD_LIBRARY_PATH
        #
        ##########################################
            loadName = self.path_to_mario + '/bin/AmiCoBuild/PyJava/libAmiCoPyJava.so'
            libamico = ctypes.CDLL(loadName)
            print libamico
        else: #else if OS is a Mac OS X (libAmiCo.dylib is searched for) or Windows (AmiCo.dll)
            name =  'AmiCoPyJava'
            loadName = ctypes.util.find_library(name)
            print loadName
            libamico = ctypes.CDLL(loadName)
            print libamico
            
        self.libamico = libamico
    
        # create environment
        javaClass = "ch/idsia/benchmark/mario/environments/MarioEnvironment"
        libamico.amicoInitialize(1, "-Djava.class.path=" + self.path_to_mario + os.path.sep + "bin" + os.path.sep + ":jdom.jar")
        libamico.createMarioEnvironment(javaClass)
    
        # specify necessary cfuncs
        reset = cfunc('reset', libamico, None, ('list', ListPOINTER(ctypes.c_int), 1))
        getEntireObservation = cfunc('getEntireObservation', libamico, ctypes.py_object,
                                     ('list', ctypes.c_int, 1),
                                     ('zEnemies', ctypes.c_int, 1))
        self.getEntireObservation = getEntireObservation
        performAction = cfunc('performAction', libamico, None, ('list', ListPOINTER(ctypes.c_int), 1))
        self.performAction = performAction
        
        getObservationDetails = cfunc('getObservationDetails', libamico, ctypes.py_object)
        self.getObservationDetails = getObservationDetails

        getVisualRGB = cfunc('getVisualRGB', libamico, ctypes.py_object)
        self.getVisualRGB = getVisualRGB
    
        options = ""
        if len(sys.argv) > 1:
            options = sys.argv[1]

        if options.startswith('"') and options.endswith('"'):
            options = options[1:-1]
            
        options1 = options + " -ls " + str(seed)

        reset(options1)
        
        # make first step to complete initialization
        obsDetails = self.getObservationDetails()
        self._setObservationDetails(obsDetails[0], obsDetails[1], obsDetails[2], obsDetails[3])
        obs = self.getEntireObservation(1, 0)
        self._integrateObservation(obs[0], obs[1], obs[2], obs[3], obs[4])
        self.ndim = self.current_state.shape[0]
        
        libamico.tick()
            
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
        
        # self.action: list of boolean values
        # self.action[0]: no movement if 0, move left if 1
        # self.action[1]: no movement if 0, move right if 1
        # self.action[2]: no movement if 0, duck if 1
        # self.action[3]: no movement if 0, jump if 1
        # self.action[4]: no movement if 0, run/shoot if 1
        # self.action[5]: no movement if 0, stand up if 1
        
        if action == 'RIGHT':
            #move right
            self.action = [0, 1, 0, 0, 0, 0] 
        elif action == 'LEFT':
            #move left
            self.action = [1, 0, 0, 0, 0, 0] 
        elif action == 'JUMP':
            #jump
            self.action = [0, 0, 0, 1, 0, 0] 
        elif action == 'STAND':
            #stand up
            self.action = [0, 0, 0, 0, 0, 1] 
        elif action == 'DUCK':
            #duck
            self.action = [0, 0, 1, 0, 0, 0] 
        elif action == 'RUN_RIGHT':
            #run right
            self.action = [0, 1, 0, 0, 1, 0] 
        elif action == 'RUN_LEFT':
            #run left
            self.action = [1, 0, 0, 0, 1, 0] 
        elif action == 'JUMP_RIGHT':
            #jump right
            self.action = [0, 1, 0, 1, 0, 0] 
        elif action == 'JUMP_LEFT':
            #jump left
            self.action = [1, 0, 0, 1, 0, 0] 
        elif action == 'RUN/SHOOT':
            #run/shoot
            self.action = [0, 0, 0, 0, 1, 0] 
        elif action == 'JUMP/RUN':
            #jump/run
            self.action = [0, 0, 0, 1, 1, 0] 
        elif action == 'END_ACTION':
            self.action = [1, 1, 1, 1, 1, 1]
        
        return self.action
            
    
    def _do_action(self, action=None):
        """Perform the given action and return it as well as the resulting state of the
        environment. If no action is given, a random action is performed.
        --------------------------------------
        Parameters:
        action:        str - action to be performed
        --------------------------------------
        Return:
        self.current_state:    np.ndarray - the agent's current position in the world, 
                                            information about the current incarnation of the agent
                                            and a representation of the agent's current perception
        action:                str - performed action
        """

        if (self.isEpisodeOver):
            action = 'END_ACTION'
            
        if action == None:
            choice = random.randint(0,len(self.actions)-1)
            action = self.actions[choice]
        
        # translate action into list which can be processed by self.performAction,
        # perform the translated action and advance the environment by 1 tick
        translatedAction = self._translate(action)
        self.performAction(translatedAction)
        self.libamico.tick()

        obsDetails = self.getObservationDetails()
        self._setObservationDetails(obsDetails[0], obsDetails[1], obsDetails[2], obsDetails[3])
        obs = self.getEntireObservation(1, 0)
        
        # self._integrateObservation also stores the data of the observation in self.current_state
        self._integrateObservation(obs[0], obs[1], obs[2], obs[3], obs[4])
        print translatedAction, '\t', action
        
        #TODO: return reward value
        return (self.current_state, 0.0)

    # MarioAI function
    def _reset(self):
        self.action = [0, 0, 0, 0, 0, 0]
        self.isEpisodeOver = False
        
    # MarioAI function; adapted to suit the needs of the project
    def _integrateObservation(self, squashedObservation, squashedEnemies, marioPos, enemiesPos, marioState):
        """Store given observation in self.marioState as well as summarize all meaningful data in one
        numpy array (self.current_state) fit for further processing
        --------------------------------------
        Parameters:
        squashedObservation:    tuple
        squashedEnemies:        tuple 
        marioPos:               tuple - the agent's position in the environment
        enemiesPos:             tuple - the nearest enemies' position and velocity in the environment
        marioState:             tuple - summarization of important data about the agent
        """
        row = self.receptiveFieldHeight
        col = self.receptiveFieldWidth
        
        levelScene=[]
        
        for i in range(row):
            levelScene.append(squashedObservation[i*col:i*col+col])

        #print squashedObservation
        #print squashedEnemies
        #print marioPos
        #print enemiesPos
        print marioState        
        self.marioFloats = marioPos
        self.enemiesFloats = enemiesPos
        self.mayMarioJump = marioState[3]
        self.isMarioOnGround = marioState[2]
        self.marioState = marioState[1]
        self.levelScene = levelScene
        self.levelScene = list(itertools.chain(*self.levelScene))
        
        # combine all meaningful data in one vector, composed of the agent's current position in the world,
        # information about the current incarnation of the agent and a representation of the agent's
        # current perception

        self.current_state = []
        self.current_state.append(self.marioFloats)
        
        # marioState[1]:
        # marioState[2]: boolean; 1 if agent is on ground, 0 otherwise
        # marioState[3]: boolean; 1 if agent may jump, 0 otherwise
        # marioState[4]: boolean; 1 if agent may shoot/jump, 0 otherwise
        # marioState[5]: boolean; 1 if agent is carrying a shell, 0 otherwise
        # marioState[6]: int, total kills
        # marioState[7]: int, kills by fire
        # marioState[8]: int, kills by stomping
        # marioState[9]: int, kills by shell
        # marioState[10]: integer; representing the remaining time

        self.current_state.append(marioState)
        self.current_state.append(self.levelScene)
        self.current_state = list(itertools.chain(*self.current_state))
        self.current_state = np.array(self.current_state)
        

    # MarioAI function 
    def _setObservationDetails(self, rfWidth, rfHeight, egoRow, egoCol):
        self.receptiveFieldWidth = rfWidth
        self.receptiveFieldHeight = rfHeight
        self.marioEgoRow = egoRow
        self.marioEgoCol = egoCol
        
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

# MarioAI class
class ListByRef(object):
    """An argument that converts a list/tuple of ctype elements into a pointer to an array of pointers to the elements
    """
    def __init__(self, etype):
        self.etype = etype
        self.etype_p = ctypes.POINTER(etype)

    def from_param(self, param):
        if isinstance(param, (list, tuple)):
            val = (self.etype_p * len(param))()
            for i, v in enumerate(param):
                if isinstance(v, self.etype):
                    val[i] = self.etype_p(v)
                else:
                    val[i] = v
            return val
        else:
            return param

# MarioAI class
class Inspectable(object):
    """ All derived classes gains the ability to print the names and values of all their fields"""
    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__,
            dict([(x,y) for (x,y) in self.__dict__.items() if not x.startswith('_')]) )

# MarioAI class
class EvaluationInfo(Inspectable):
    def __init__(self, evInfo):
        print "widthCells = ", evInfo[0]
        print "widthPhys  = ", evInfo[1]
        print "flowersDevoured = ", evInfo[2]
        print "killsByFire = ", evInfo[3]
        print "killsByShell = ", evInfo[4]
        print "killsByStomp = ",  evInfo[5]
        print "killsTotal = ", evInfo[6]
        print "marioMode = ", evInfo[7]
        print "marioStatus = ", evInfo[8]
        print "mushroomsDevoured = ", evInfo[9]
        print "marioCoinsGained = ", evInfo[10]
        print "timeLeft = ", evInfo[11]
        print "timeSpent = ", evInfo[12]
        print "hiddenBlocksFound = ", evInfo[13]

        self.widthCells = evInfo[0]
        self.widthPhys = evInfo[1]
        self.flowersDevoured = evInfo[2]
        self.killsByFire = evInfo[3]
        self.killsByShell = evInfo[4]
        self.killsByStomp = evInfo[5]
        self.killsTotal = evInfo[6]
        self.marioMode = evInfo[7]
        self.marioStatus = evInfo[8]
        self.mushroomsDevoured = evInfo[9]
        self.marioCoinsGained = evInfo[10]
        self.timeLeft = evInfo[11]
        self.timeSpent = evInfo[12]
        self.hiddenBlocksFound = evInfo[13]

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
    env_mario = EnvMario()
