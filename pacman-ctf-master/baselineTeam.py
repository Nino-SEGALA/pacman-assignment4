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

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    dataPreProcessed = self.dataInput(gameState)
                                                        
    # info_mask = np.zeros(walls.shape)
    # info_mask[int(team_mate_x)][int(team_mate_y)] = team_mate_pac 
    print("dist = ",gameState.getAgentDistances())
    # print("food = \n",food)
    # print("walls = \n",walls)
    # print('info_mask = \n',info_mask)
    


    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

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


  def ourColor(self, gameState):
    blue = gameState.getBlueTeamIndices()
    return 'blue' if self.index in blue else 'red'


  def teamMateAndOpponentState(self, gameState, color):
    red = gameState.getRedTeamIndices()
    blue = gameState.getBlueTeamIndices()
    team_mate = []
    opponent = []
    for i in range(len(blue)):
        if color == 'blue':
          if blue[i] != self.index:
              team_mate.append(gameState.getAgentState(blue[i]))
          opponent.append(gameState.getAgentState(red[i]))
        elif color == 'red':
            if red[i] != self.index:
                team_mate.append(gameState.getAgentState(red[i]))
            opponent.append(gameState.getAgentState(blue[i]))
    # team_mate_x, team_mate_y = team_mate.getPosition()
    # team_mate_pac = team_mate.isPacman * 2 - 1
    return team_mate[0], opponent  # one team_mate

  def invert(self, u):
      (i, j) = u
      return (int(j), int(i))

  def dataInput(self, gameState):
    """
    Returns the preprocessed data for the neural network
    The matrices are flipped by 180° for the Red team: the output must be flipped also!
    1. Walls
    2. Food
    3. PowerCapsule
    4. PacManFriend
    5. ScaredGhostFriend
    6. PacManOpponent
    7. ScaredGhostOpponent
    8. Sides ?
    """

    color = self.ourColor(gameState)
    agent_state = gameState.getAgentState(self.index)
    team_mate_state, opponent_state = self.teamMateAndOpponentState(gameState, color)
    width = gameState.getWalls().width  # width of the board (32)
    height = gameState.getWalls().height  # height of the board (16)

    print((width, height))

    # 1
    walls = np.array([[int(gameState.getWalls()[i][j]) for i in range(width)]
                      for j in range(height)])
    print()
    print("walls")
    print(walls)

    # 2
    food_red = np.array([[int(gameState.getRedFood()[i][j]) for i in range(width)]
                         for j in range(height)])
    food_blue = np.array([[int(gameState.getBlueFood()[i][j]) for i in range(width)]
                          for j in range(height)])

    # 3
    print(gameState.getRedCapsules())
    power_capsule_red = np.zeros((height, width), dtype=int)
    for (i, j) in gameState.getRedCapsules():
        power_capsule_red[j][i] = 1  # invert w and h
    power_capsule_blue = np.zeros((height, width), dtype=int)
    for (i, j) in gameState.getBlueCapsules():
        power_capsule_blue[j][i] = 1  # invert w and h

    if color == 'blue':
        food = food_blue - food_red
        power_capsule = power_capsule_blue - power_capsule_red
    else:
        food = food_red - food_blue
        power_capsule = power_capsule_red - power_capsule_blue
    print("food")
    print(food)
    print("power_capsule")
    print(power_capsule)

    # 4
    pacman_friend = np.zeros((height, width), dtype=int)
    if agent_state.scaredTimer == 0:
        pacman_friend[self.invert(gameState.getAgentPosition(self.index))] = 1  # TODO: check superPacMan
    if team_mate_state.scaredTimer == 0:
        pacman_friend[self.invert(team_mate_state.getPosition())] = 1
    print("pacman_friend")
    print(pacman_friend)

    # 5
    scared_ghost_friend = np.zeros((height, width), dtype=int)
    if agent_state.scaredTimer > 0:
        scared_ghost_friend[self.invert(gameState.getAgentPosition(self.index))] = 1
    if team_mate_state.scaredTimer > 0:
        scared_ghost_friend[self.invert(team_mate_state.getPosition())] = 1
    print("scared_ghost_friend")
    print(scared_ghost_friend)

    # 6
    pacman_opponent = np.zeros((height, width), dtype=int)
    # TODO : calculate where opponents are
    """if opponent_state[0].scaredTimer == 0:
        pacman_opponent[self.invert(opponent_state[0].getPosition())] = 1  # TODO: check superPacMan
    if opponent_state[1].scaredTimer == 0:
        pacman_opponent[self.invert(opponent_state[1].getPosition())] = 1"""
    print("pacman_opponent")
    print(pacman_opponent)

    # 7
    scared_ghost_opponent = np.zeros((height, width), dtype=int)
    """if opponent_state[0].scaredTimer > 0:
        scared_ghost_opponent[self.invert(opponent_state[0].getPosition())] = 1
    if opponent_state[1].scaredTimer > 0:
        scared_ghost_opponent[self.invert(opponent_state[1].getPosition())] = 1"""
    print("scared_ghost_opponent")
    print(scared_ghost_opponent)

    return walls, food, power_capsule, pacman_friend, scared_ghost_friend, pacman_opponent, scared_ghost_opponent

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
