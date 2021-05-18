import copy

def new_position(pos1, pos2):
    return pos1[0] + pos2[0], pos1[1] + pos2[1]

def same_team(index1, index2):
    if (index1 % 2 == 0 and index2 % 2 == 0) or (index1 % 2 == 1 and index2 % 2 == 1):
        return True
    return False

class State:
    def __init__(self, color, score, wall, food, main_index, teammate_index, main_position, teammate_position,
                 main_collected, teammate_collected, team_scared, opponent_scared, opponent_index1,
                 opponent_index2, opponent_position1=None, opponent_position2=None, opponent_collected=None):
        self.color = color
        self.score = score
        self.wall = wall
        self.food = food
        self.moves = [(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)]  # all moves
        # Index
        self.mainIndex = main_index  # position of agent who plays
        self.teammateIndex = teammate_index  # position of teammate
        self.opponentIndex1 = opponent_index1  # position of opponent if known
        self.opponentIndex2 = opponent_index2  # position of other opponent if known
        # Position
        self.mainPosition = main_position  # position of agent who plays
        self.teammatePosition = teammate_position  # position of teammate
        self.opponentPosition1 = opponent_position1  # position of opponent if known
        self.opponentPosition2 = opponent_position2  # position of other opponent if known
        # Collected food
        self.mainCollected = main_collected  # collected food of the agent who plays
        self.teammateCollected = teammate_collected  # collected food of teammate
        self.opponentCollected = opponent_collected  # collected food of opponent
        # scared
        self.teamScared = team_scared
        self.opponentScared = opponent_scared

    def printState(self):
        print()
        print("printState")
        print(self.wall)
        print(self.color, self.score)
        print(self.mainIndex, self.teammateIndex, self.mainPosition, self.teammatePosition)
        print(self.mainCollected, self.teammateCollected, self.teamScared, self.opponentScared)
        print(self.opponentIndex1, self.opponentIndex2, self.opponentPosition1, self.opponentPosition2, self.opponentCollected)
        print()

    def position_index(self, index):
        correspond = [(self.mainIndex, self.mainPosition), (self.teammateIndex, self.teammatePosition),
                      (self.opponentIndex1, self.opponentPosition1), (self.opponentIndex2, self.opponentPosition2)]
        for (ind, pos) in correspond:
            if ind == index:
                return pos
        print("positionIndex | Wrong index given")
        return None

    def possible_moves(self, index):
        res = []
        position = self.position_index(index)
        for move in self.moves:
            new_pos = new_position(position, move)
            if self.wall[new_pos[0]][new_pos[1]] == 0:  # possible to go there
                res.append(move)
        return res

    def food_after_move(self, position_after_move):
        i, j = position_after_move
        if self.food[i][j] != 0:
            new_food = copy.deepcopy(self.food)
            new_food[i][j] = 0
            return True, new_food
        return False, self.food

    def update_food(self, index):
        if index == self.mainIndex:
            return self.mainCollected + 1, self.teammateCollected, self.opponentCollected
        elif index == self.teammateIndex:
            return self.mainCollected, self.teammateCollected + 1, self.opponentCollected
        else:
            return self.mainCollected, self.teammateCollected, self.opponentCollected + 1

    def update_position(self, index, new_pos):
        if index == self.mainIndex:
            return new_pos, self.teammatePosition, self.opponentPosition1, self.opponentPosition2
        elif index == self.teammateIndex:
            return self.mainPosition, new_pos, self.opponentPosition1,  self.opponentPosition2
        elif index == self.opponentIndex1:
            return self.mainPosition, self.teammatePosition, new_pos,  self.opponentPosition2
        else:
            return self.mainPosition, self.teammatePosition,  self.opponentPosition1, new_pos

    def is_red(self):
        return 1 if self.color == 'red' else -1

    def new_score(self, index, new_pos):
        vertical = new_pos[1]
        limit = len(self.wall[0])
        new_score = self.score
        if index == self.mainIndex:
            if vertical == limit and self.mainCollected > 0:  # returns home with collected food
                new_score += self.is_red() * self.mainCollected
                self.mainCollected = 0
        elif index == self.teammateIndex:
            if vertical == limit and self.teammateCollected > 0:  # returns home with collected food
                new_score += self.is_red() * self.teammateCollected
                self.teammateCollected = 0
        else:
            if vertical == limit - 1 and self.opponentCollected > 0:  # returns home with collected food
                new_score += self.is_red() * self.opponentCollected  # "1 opponent care all the collected food"
                self.opponentCollected = 0
        return new_score

    def in_homebase(self, index, new_pos):
        limit = len(self.wall[0])
        if same_team(index, self.mainIndex):  # right part of the field
            if new_pos[1] >= limit:  # right side
                return True
        else:  # left part of field
            if new_pos[1] < limit:  # left side
                return True
        return False

    """def pacman_eaten(self, index, new_pos):
        # TODO: scared?
        correspond = [(self.mainIndex, self.mainPosition), (self.teammateIndex, self.teammatePosition),
                      (self.opponentIndex1, self.opponentPosition1), (self.opponentIndex2, self.opponentPosition2)]
        for (ind, pos) in correspond:
            if ind != index:  # not ourself
                if pos == new_pos:  # 2 agents, same position
                    if not same_team(index, ind):  # different team
                        if self.in_homebase(index, new_pos):  # index, new_pos in homebase: ghost
                            if same_team(index, self.mainIndex):"""

    def next_states(self, index):
        children = []
        position = self.position_index(index)
        for move in self.possible_moves(index):
            new_pos = new_position(position, move)  # positions
            main_pos, teammate_pos, opp_pos1, opp_pos2 = self.update_position(index, new_pos)
            food_cond, food = self.food_after_move(new_pos)  # food matrix
            if food_cond:
                main_coll, teammate_coll, opponent_coll = self.update_food(index)  # collected food
            else:
                main_coll, teammate_coll, opponent_coll = self.mainCollected, self.teammateCollected, \
                                                          self.opponentCollected
            new_score = self.new_score(index, new_pos)
            new_state = State(self.color, new_score, self.wall, food, self.mainIndex, self.teammateIndex, main_pos, teammate_pos,
                              main_coll, teammate_coll, self.teamScared, self.opponentScared, self.opponentIndex1,
                              self.opponentIndex2, opp_pos1, opp_pos2, opponent_coll)

            #print("new state : ", move)
            #new_state.printState()
            children.append((move, new_state))
        #print("CHILDREN :")
        #for (move, child) in children:
            #print(move, children)
        #print("End children")
        return children
