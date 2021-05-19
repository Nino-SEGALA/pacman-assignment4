import numpy as np
from captureAgents import CaptureAgent
 
 ### Preprocess the data for the network
    # returns our color


class State(CaptureAgent):
    def ourColor(self, gameState):
        blue = gameState.getBlueTeamIndices()
        return 'blue' if self.index in blue else 'red'

    # returns our team_mate state and a list of our opponents' state
    def teamMateAndOpponentState(self,gameState, color):
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
    def invert(self,gameState, u):
        (i, j) = u
        return (int(j), int(i))

    # invert top and down to look like the display
    def reorderMatrixLikeDisplay(self,gameState, mat):
        height, width = mat.shape
        for i in range(height // 2):
            for j in range(width):
                first_element = mat[i][j]
                mat[i][j], mat[height - i - 1][j] = mat[height - i - 1][j], first_element  # inversion

    # invert right/left and top/down to change red and blue players
    def invertMatrixForRed(self,gameState, mat):
        height, width = mat.shape
        for i in range(height):
            for j in range(width // 2):
                first_element = mat[i][j]
                # inversion
                mat[i][j], mat[height - i - 1][width - j - 1] = mat[height - i - 1][width - j - 1], first_element

    # free neighbours of a box in the matrix
    def freeNeighbours(self, gameState, mat, pos):
        nghb = []
        for dir in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            new_pos = (pos[0] + dir[0], pos[1] + dir[1])
            if mat[new_pos[0]][new_pos[1]] == -2:  # not a wall and unvisited for bfs
                nghb.append(new_pos)
        return nghb

    # bfs to find the boxes at the right distance of the agent
    def bfsOnBoard(self, gameState, color, agent_pos, distance):
        res = []
        # preparation
        width = gameState.getWalls().width  # width of the board (32)
        height = gameState.getWalls().height  # height of the board (16)
        board = np.array([[-int(gameState.getWalls()[i][j]) for i in range(width)]
                        for j in range(height)])
        for i in range(height):
            for j in range(width):
                if board[i][j] == 0:
                    board[i][j] = -2
        agent_pos = height - agent_pos[0] - 1, agent_pos[1]  # like display
        self.reorderMatrixLikeDisplay(gameState,board)
        if color is 'red':
            self.invertMatrixForRed(gameState,board)
            agent_pos = height - agent_pos[0] - 1, width - agent_pos[1] - 1
        board[agent_pos[0]][agent_pos[1]] = 0  # agent's position

        # BFS
        lookAt = [agent_pos]  # initialize with agent's position
        while lookAt:
            (u, v) = lookAt.pop(0)  # remove first element of lookAt
            depth = board[u][v]
            if depth == distance:
                res.append((u, v))
            if depth < distance:  # don't look further than distance
                neighbours = self.freeNeighbours(gameState,board, (u, v))
                for (i, j) in neighbours:
                    board[i][j] = depth + 1
                    lookAt.append((i, j))

        return res

    # compare two list to find a common position
    def findCommonPosition(self, gameState, pos1, pos2):
        for pos in pos1:
            if pos in pos2:
                return pos

    # get distances to the opponents
    def distancesToOpponents(self, gameState):
        dist = gameState.getAgentDistances()
        if self.index % 2 == 0:
            return [dist[1], dist[3]]
        return [dist[0], dist[2]]

    def dataInput(self, gameState, agent):
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
        # needs the functions: ourColor, teamMateAndOpponentState, invert, reorderMatrixLikeDisplay, invertMatrixForRed

    
        color = self.ourColor(gameState)
    
        agent_state = gameState.getAgentState(self.index)
        team_mate_state, opponent_state = self.teamMateAndOpponentState(gameState, color)
        width = gameState.getWalls().width  # width of the board (32)
        height = gameState.getWalls().height  # height of the board (16)

        # 1
        walls = np.array([[int(gameState.getWalls()[i][j]) for i in range(width)]
                            for j in range(height)])
        self.reorderMatrixLikeDisplay(gameState,walls)
        if color is 'red':
            self.invertMatrixForRed(gameState,walls)
   
        # 2
        food_red = np.array([[int(gameState.getRedFood()[i][j]) for i in range(width)]
                                for j in range(height)])
        food_blue = np.array([[int(gameState.getBlueFood()[i][j]) for i in range(width)]
                                for j in range(height)])

        # 3
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
        self.reorderMatrixLikeDisplay(gameState,food)
        self.reorderMatrixLikeDisplay(gameState,power_capsule)
        if color is 'red':
            self.invertMatrixForRed(gameState,food)
            self.invertMatrixForRed(gameState,power_capsule)


        # 4
        pacman_friend = np.zeros((height, width), dtype=int)
        if agent_state.scaredTimer == 0:
            pacman_friend[self.invert(gameState,gameState.getAgentPosition(self.index))] = 1  # TODO: check superPacMan
        if team_mate_state.scaredTimer == 0:
            pacman_friend[self.invert(gameState,team_mate_state.getPosition())] = -1  # TODO
        self.reorderMatrixLikeDisplay(gameState,pacman_friend)
        if color is 'red':
            self.invertMatrixForRed(gameState,pacman_friend)


        # 5
        scared_ghost_friend = np.zeros((height, width), dtype=int)
        if agent_state.scaredTimer > 0:
            scared_ghost_friend[self.invert(gameState,gameState.getAgentPosition(self.index))] = 1
        if team_mate_state.scaredTimer > 0:
            scared_ghost_friend[self.invert(gameState,team_mate_state.getPosition())] = 1
        if color is 'red':
            self.invertMatrixForRed(gameState,scared_ghost_friend)
        self.reorderMatrixLikeDisplay(gameState,scared_ghost_friend)

        # 6
        pacman_opponent = np.zeros((height, width), dtype=int)
        # TODO : calculate where opponents are
        if opponent_state[0].scaredTimer == 0:
            pos1 = opponent_state[0].getPosition()
            if pos1:
                pacman_opponent[self.invert(gameState,pos1)] = 1  # TODO: check superPacMan
        if opponent_state[1].scaredTimer == 0:
            pos2 = opponent_state[1].getPosition()
            if pos2:
                pacman_opponent[self.invert(gameState,pos2)] = 1
        self.reorderMatrixLikeDisplay(gameState,pacman_opponent)
        if color is 'red':
            self.invertMatrixForRed(gameState,pacman_opponent)
        eatenFood = agent.positionEatenFood(gameState)
        for pos in eatenFood:  # already the right positions
            pacman_opponent[pos] = 1

        # SIMPLE TEST CASE
        '''distToOp = gameState.distancesToOpponents(agent)
        # TODO: team_mate distances ?!
        distaaa = abs(gameState.invert(gameState.getAgentPosition(agent.index))[0] - gameState.invert(team_mate_state.getPosition())[0])
        distbbb = abs(gameState.invert(gameState.getAgentPosition(agent.index))[1] - gameState.invert(team_mate_state.getPosition())[1])
        distToTeamMate = distaaa + distbbb
        distToOpTeamMate = [distToOp[0] + distToTeamMate, distToOp[1] - distToTeamMate]
        agent_pos = gameState.invert(gameState.getAgentPosition(agent.index))
        team_mate_pos = gameState.invert(team_mate_state.getPosition())
        boxes1 = gameState.bfsOnBoard(color, agent_pos, distToOp[0])
        boxes2 = gameState.bfsOnBoard(color, team_mate_pos, distToOpTeamMate[0])
        boxes3 = gameState.bfsOnBoard(color, agent_pos, distToOp[1])
        boxes4 = gameState.bfsOnBoard(color, team_mate_pos, distToOpTeamMate[1])'''
        # SIMPLE TEST CASE



        # 7
        scared_ghost_opponent = np.zeros((height, width), dtype=int)
        """if opponent_state[0].scaredTimer > 0:
        scared_ghost_opponent[gameState.invert(opponent_state[0].getPosition())] = 1
        if opponent_state[1].scaredTimer > 0:
        scared_ghost_opponent[gameState.invert(opponent_state[1].getPosition())] = 1"""
        self.reorderMatrixLikeDisplay(gameState,scared_ghost_opponent)
        if color is 'red':
            self.invertMatrixForRed(gameState,scared_ghost_opponent)

        matrices = [walls, food, power_capsule, pacman_friend, scared_ghost_friend, pacman_opponent, scared_ghost_opponent]
        output = np.stack(matrices,axis=-1)
        output = np.expand_dims(output, axis=0)
        output = output.astype(dtype=np.float32)
        return output


