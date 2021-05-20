# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import numpy as np
import agent as a
import tensorflow as tf
from state import State

tfk = tf.keras
ACTIONS = {'North':0,'South':1,'East':2,'West':3,'Stop':4}
ACTIONS_VALUE = {0:'North',1:'South',2:'East',3:'West',4:'Stop'}

class Counter:
  def __init__(self):
    self.counter = 0

counter = Counter()

agent_red = a.Agent(n_actions=5, gamma=0.99, epsilon=0.1, alpha=1e-3, state_dim = (18,34,7), batch_size=32,
            buffer_size=(30000,), eps_final=0.1, name='Network_red')
try:
  agent_red.NN = tfk.models.load_model('models/network_red')
  agent_red.target_NN = tfk.models.load_model('models/target_red')
  print('loaded old model red')
except:
  print("couldn't load red")

agent_blue = a.Agent(n_actions=5, gamma=0.99, epsilon=0.1, alpha=1e-3, state_dim = (18,34,7), batch_size=32,
            buffer_size=(30000,), eps_final=0.1,name='Network_blue')
try:
  agent_blue.NN = tfk.models.load_model('models/network_blue')
  agent_blue.target_NN = tfk.models.load_model('models/target_blue')
  print('loaded old model blue')
except:
  print("couldn't load blue")
SCORES = [[] for _ in range(4)]
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.red_ind = gameState.getRedTeamIndices()
    self.blue_ind = gameState.getBlueTeamIndices()
    self.counter = counter
    self.my_team = 'blue' if self.index in self.blue_ind else 'red'

    if self.my_team == 'blue':
      self.team = self.blue_ind
      self.opp = self.red_ind
      self.agent = agent_blue
    else:
      self.team = self.red_ind
      self.opp = self.blue_ind
      self.agent = agent_red


    self.hist = SCORES[self.index]
    self.saved = False
    self.states = State(self.index)
    self.ourFoodLastStep = self.getFood(gameState)
    self.width = self.ourFoodLastStep.width  # width of the board (32)
    self.height = self.ourFoodLastStep.height
    self.ourFoodLastStep = self.getOurFood(gameState)
    self.old_gameState = gameState
    self.old_state = self.states.dataInput(gameState, self)
    self.actions_ohc = np.zeros(5)

  def chooseAction(self, gameState):
    
    c_state = self.states.dataInput(gameState, self)

    reward = self.get_reward(self.old_gameState, gameState)
    self.agent.add_to_buffer(self.old_state, self.actions_ohc, reward, c_state)
    self.agent.update_step()
    self.agent.learn()
    self.agent.update_network()
    
    if gameState.data.timeleft <= 3:
      self.agent.NN.save(f'models/network_{self.my_team}')
      self.agent.target_NN.save(f'models/target_{self.my_team}')
      #self.hist.append(gameState.getAgentState(self.index).numReturned)
      #np.save(f"models/hist_{self.index}",self.hist,allow_pickle=True)
      #self.agent.update_epsilon()
      #self.agent.update_reward_annealing()
      print('saving ... ')
      #print(self.agent.epsilon)
    
    
    actions = gameState.getLegalActions(self.index)
    possible_actions = [ACTIONS[key] for key in actions]
    
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    # action = self.get_NN_action(c_state, possible_actions)
    action = self.get_base_action(gameState, actions)


    action_num = ACTIONS[action]
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    self.actions_ohc = np.zeros(5)
    self.actions_ohc[action_num] = 1.0
    self.old_state = c_state
    self.old_gameState = gameState

    return action


  def get_base_action(self, gameState, actions, epsilon=0.0):
    if np.random.rand() < epsilon:
      action = np.random.choice(actions)
    else:
      values = [self.evaluate(gameState, a) for a in actions]
      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]
      action = bestActions[0]
    
    if gameState.getAgentState(self.index).numCarrying >= 5 :
      bestDist = 9999
      for action in actions:
          successor = self.getSuccessor(gameState, action)
          pos2 = successor.getAgentPosition(self.index)
          dist = self.getMazeDistance(self.start, pos2)
          if dist < bestDist:
              bestAction = action
              bestDist = dist
      action = bestAction
    
    return action


  def get_NN_action(self, c_state, possible_actions):
    action = self.agent.get_action(c_state, possible_actions, self.index)
    self.actions_ohc = np.zeros(5)
    self.actions_ohc[action] = 1.0

    if action not in possible_actions:
      action = np.random.choice(possible_actions)
    action = ACTIONS_VALUE[action]
    return action
    

  def get_reward(self, old_gameState, gameState):
    score_reward = gameState.getAgentState(self.index).numReturned - old_gameState.getAgentState(self.index).numReturned
    food_reward = (gameState.getAgentState(self.index).numCarrying - old_gameState.getAgentState(self.index).numCarrying)*0.1
    opp_score = 0
    opp_food = 0
    for ind in self.opp:
      opp_score -= gameState.getAgentState(ind).numReturned - old_gameState.getAgentState(ind).numReturned
      opp_food -= (gameState.getAgentState(ind).numCarrying - old_gameState.getAgentState(ind).numCarrying)*0.1

    ## to make sure the agent moves from the origin
    # dist_reward = 0
    # pos1 = gameState.getAgentPosition(self.index)
    # dist1 = self.getMazeDistance(self.start, pos1)
    # pos2 = old_gameState.getAgentPosition(self.index)
    # dist2 = self.getMazeDistance(self.start, pos2)
    # if dist2 >= dist1:
    #   dist_reward = -0.0001 

    final_reward = score_reward + food_reward + opp_score + opp_food 
    return final_reward



  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}
  
  def getOurFood(self, gameState):
          if self.my_team is 'red':  # agent is red, wants to keep an eye on its food
              food = np.array([[int(gameState.getRedFood()[i][j]) for i in range(self.width)]
                              for j in range(self.height)])
              self.states.reorderMatrixLikeDisplay(gameState,food)
              self.states.invertMatrixForRed(gameState,food)
          else:
              food = np.array([[int(gameState.getBlueFood()[i][j]) for i in range(self.width)]
                              for j in range(self.height)])
              self.states.reorderMatrixLikeDisplay(gameState,food)
          return food


  def setNewFoodLastStep(self, gameState):
      self.ourFoodLastStep = self.getOurFood(gameState)

  def positionEatenFood(self, gameState):
        newFood = self.getOurFood(gameState)
        res = []
        for i in range(self.height):
            for j in range(self.width):
                # food eaten by opponent
                if self.ourFoodLastStep[i][j] == 1 and newFood[i][j] == 0:
                    res.append((i, j))
        return res

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


