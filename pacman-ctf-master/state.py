import copy

def new_position(pos1, pos2):
    return pos1[0] + pos2[0], pos1[1] + pos2[1]

class State:
    def __init__(self, score, wall, food, main_index, teammate_index, main_position, teammate_position, main_collected,
                 teammate_collected, opponent_index1=None, opponent_index2=None, opponent_position1=None,
                 opponent_position2=None, opponent_collected=None):
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
            return self.main_collected + 1, self.teammate_collected, self.opponent_collected
        elif index == self.teammateIndex:
            return self.main_collected, self.teammate_collected + 1, self.opponent_collected
        else:
            return self.main_collected, self.teammate_collected, self.opponent_collected + 1

    def update_position(self, index, new_pos):
        if index == self.mainIndex:
            return new_pos, self.teammate_position, self.opponent_position1, self.opponent_position2
        elif index == self.teammateIndex:
            return self.main_position, new_pos, self.opponent_position1,  self.opponent_position2
        elif index == self.opponent_index1:
            return self.main_position, self.teammate_position, new_pos,  self.opponent_position2
        else
            return self.main_position, self.teammate_position,  self.opponent_position1, new_pos

    def next_states(self, index):
        children = []
        position = self.position_index(index)
        for move in self.possible_moves(index):
            new_pos = new_position(position, move)  # positions
            main_pos, teammate_pos, opp_pos1, opp_pos2 = self.update_position(index, new_pos)
            food_cond, food = self.food_after_move(new_pos)  # food matrix
            if food_cond:
                main_coll, teammate_coll, opponent_coll = self.update_food(index)  # collected food
            # score
            # PacMan eaten
            # TODO: calculate what changes
            new_state = State(self.score, self.wall, food, self.main_index, self.teammate_index, main_pos, teammate_pos,
                              main_coll, teammate_coll, self.opponent_index1, self.opponent_index2, opp_pos1, opp_pos2,
                              opponent_coll)

        return children
