from captureAgents import CaptureAgent
# from game import Directions
import numpy as np


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='AgentAlphaBeta', second='AgentAlphaBeta'):
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

class AgentAlphaBeta(CaptureAgent):
    #def registerInitialState(self, gameState):
        #return None

    # returns our color
    def ourColor(self, gameState):
        blue = gameState.getBlueTeamIndices()
        return 'blue' if self.index in blue else 'red'

    def teammateIndex(self):
        return 2 - self.index + 2 * (self.index % 2 == 1)

    # returns our team_mate state and a list of our opponents' state
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
        # team_mate_pac = team_mate.isPacman * 2 - 1  # pacman or ghost
        return team_mate[0], opponent  # one team_mate

    # invert i and j to have matrices like the display
    def invert(self, u):
        (i, j) = u
        return (int(j), int(i))

    def correctPosition(self, pos, col, height, width):
        pos = self.invert(pos)
        pos = height - pos[0] - 1, pos[1]
        if col is 'red':
            pos = height - pos[0] - 1, width - pos[1] - 1
        return pos

    def createStateFromGameState(self, gameState):
        """(self, color, score, wall, food, main_index, teammate_index, main_position, teammate_position,
                 main_collected, teammate_collected, team_scared, opponent_scared, opponent_index1,
                 opponent_index2, opponent_position1=None, opponent_position2=None, opponent_collected=None)"""
        width = self.getWalls().width  # width of the board (32)
        height = self.getWalls().height  # height of the board (16)
        team_mate_state, opponent_state = self.teamMateAndOpponentState(agent, color)

        color = self.ourColor(gameState)
        score = gameState.getScore()
        walls = np.array([[int(self.getWalls()[i][j]) for i in range(width)]
                          for j in range(height)])
        self.reorderMatrixLikeDisplay(walls)
        if color is 'red':
            self.invertMatrixForRed(walls)
        food_red = np.array([[int(self.getRedFood()[i][j]) for i in range(width)]
                             for j in range(height)])
        food_blue = np.array([[int(self.getBlueFood()[i][j]) for i in range(width)]
                              for j in range(height)])
        if color is 'blue':
            food = food_blue - food_red
        else:
            food = food_red - food_blue
        main_index = self.index
        teammate_index = self.teammateIndex

        main_position = self.getAgentPosition(main_index)
        main_position = self.correctPosition(main_position, color, height, width)  # corrected position
        teammate_position = self.getAgentPosition(teammate_index)
        teammate_position = self.correctPosition(teammate_position, color, height, width)  # corrected position

        main_collected = self.numCarrying
        teammate_collected = team_mate_state.numCarrying

        team_scared = self.scaredTimer > 0
        opponent_scared = opponent_state[0].scaredTimer > 0

        opponent_index1 = (self.index + 1) % 4
        opponent_index2 = (self.index + 3) % 4
        opponent1_state = gameState.getAgentState(opponent_index1)
        opponent2_state = gameState.getAgentState(opponent_index2)

        opponent_position1 = self.getAgentPosition(opponent_index1)
        opponent_position1 = self.correctPosition(opponent_position1, color, height, width)  # corrected position
        opponent_position2 = self.getAgentPosition(opponent_index2)
        opponent_position2 = self.correctPosition(opponent_position2, color, height, width)  # corrected position

        opponent_collected = opponent1_state.numCarrying + opponent1_state.numCarrying


    def chooseAction(self, gameState):
        """
        Picks the best action given to alphabeta
        """

        actions = gameState.getLegalActions(self.index)

        print("alpha beta agent : ", actions[0])
        return actions[0]
