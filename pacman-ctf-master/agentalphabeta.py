from captureAgents import CaptureAgent
# from game import Directions
import numpy as np
import state
import alphabeta

TIME_LIMIT = 0.95
DEPTH = 2

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
    """def teamMateState(self, gameState, color):
        red = gameState.getRedTeamIndices()
        blue = gameState.getBlueTeamIndices()
        team_mate = []
        for i in range(len(blue)):
            if color == 'blue':
                if blue[i] != self.index:
                    team_mate.append(gameState.getAgentState(blue[i]))
            elif color == 'red':
                if red[i] != self.index:
                    team_mate.append(gameState.getAgentState(red[i]))
        return team_mate[0]  # one team_mate"""

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

    # invert top and down to look like the display
    def reorderMatrixLikeDisplay(self, mat):
        height, width = mat.shape
        for i in range(height // 2):
            for j in range(width):
                first_element = mat[i][j]
                mat[i][j], mat[height - i - 1][j] = mat[height - i - 1][j], first_element  # inversion

    def createStateFromGameState(self, gameState):
        """(self, color, score, wall, food, main_index, teammate_index, main_position, teammate_position,
                 main_collected, teammate_collected, team_scared, opponent_scared, opponent_index1,
                 opponent_index2, opponent_position1=None, opponent_position2=None, opponent_collected=None)"""
        width = gameState.getWalls().width  # width of the board (32)
        height = gameState.getWalls().height  # height of the board (16)

        color = self.ourColor(gameState)
        score = gameState.getScore()
        wall = np.array([[int(gameState.getWalls()[i][j]) for i in range(width)]
                          for j in range(height)])
        self.reorderMatrixLikeDisplay(wall)
        if color is 'red':
            self.invertMatrixForRed(wall)
        food_red = np.array([[int(gameState.getRedFood()[i][j]) for i in range(width)]
                             for j in range(height)])
        food_blue = np.array([[int(gameState.getBlueFood()[i][j]) for i in range(width)]
                              for j in range(height)])
        if color is 'red':
            food = food_blue - food_red
        else:
            food = food_red - food_blue
        self.reorderMatrixLikeDisplay(food)
        if color is 'red':
            self.invertMatrixForRed(food)
        main_index = self.index
        teammate_index = self.teammateIndex()

        main_position = gameState.getAgentPosition(main_index)
        main_position = self.correctPosition(main_position, color, height, width)  # corrected position
        teammate_position = gameState.getAgentPosition(teammate_index)
        teammate_position = self.correctPosition(teammate_position, color, height, width)  # corrected position

        main_state = gameState.getAgentState(main_index)
        team_mate_state = gameState.getAgentState(teammate_index)
        main_collected = main_state.numCarrying
        teammate_collected = team_mate_state.numCarrying

        team_scared = main_state.scaredTimer > 0

        opponent_index1 = (self.index + 1) % 4
        opponent_index2 = (self.index + 3) % 4
        opponent1_state = gameState.getAgentState(opponent_index1)
        opponent2_state = gameState.getAgentState(opponent_index2)

        opponent_scared = opponent1_state.scaredTimer > 0

        opponent_position1 = gameState.getAgentPosition(opponent_index1)
        if opponent_position1:  # not None
            opponent_position1 = self.correctPosition(opponent_position1, color, height, width)  # corrected position
        opponent_position2 = gameState.getAgentPosition(opponent_index2)
        if opponent_position2:  # not None
            opponent_position2 = self.correctPosition(opponent_position2, color, height, width)  # corrected position

        opponent_collected = opponent1_state.numCarrying + opponent2_state.numCarrying

        return state.State(color, score, wall, food, main_index, teammate_index, main_position, teammate_position,
                           main_collected, teammate_collected, team_scared, opponent_scared, opponent_index1,
                           opponent_index2, opponent_position1, opponent_position2, opponent_collected)



    def chooseAction(self, gameState):
        """
        Picks the best action given to alphabeta
        """

        print()
        print("- chooseAction -")

        actions = gameState.getLegalActions(self.index)

        state = self.createStateFromGameState(gameState)
        #state.printState()
        #children = state.next_states(self.index)

        depth = DEPTH
        alpha = - np.inf
        beta = np.inf
        player = self.index
        bestMove = alphabeta.alphabeta(state, depth, alpha, beta, player, getBestMove=True)
        print("chooseAction : agent =", player, "position =", state.mainPosition, "bestMove = ", bestMove)

        #print("alpha beta agent : ", bestMove)
        return bestMove
