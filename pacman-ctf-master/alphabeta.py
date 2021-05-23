import multiprocessing
import time
from draft import choosenAction
import numpy as np
import state as stt


# PARAMETERS
MAX_SCORE = 20
MAX_COLLECTED = 20
DISTANCE_PENALIZED_OUR_AGENTS = 6
NUMBER_FOOD_COLLECT = 5
MAX_DISTANCE_FOOD = 1  # h * w or w
MAX_DISTANCE_HOMEBASE = 1  # h * w/2 or w/2
MAX_DISTANCE_OPPONENT = 5  # position unknown above

COEF_SCORE = 40
COEF_COLL = 30
COEF_COLL_OPP = 5
COEF_DISTANCE_FOOD1 = 1
COEF_DISTANCE_FOOD2 = 0.5
COEF_DISTANCE_OUR_AGENTS = 0.5  # 0.1
COEF_DISTANCE_HOMEBASE1 = 1
COEF_DISTANCE_HOMEBASE2 = 0.5
COEF_DISTANCE_OPPONENT = 1


# Your foo function
def foo(n, action):
    for i in range(5 * n):
        action.value += 1
        print("Tick", action.value)
        time.sleep(1)


def main():
    # Start foo as a process
    #global choosenAction
    foo(1, choosenAction)
    print("0 :", choosenAction.value)
    foo(2, choosenAction)
    print("1 :", choosenAction.value)
    p = multiprocessing.Process(target=foo, name="Foo", args=(10, choosenAction))
    p.start()

    # Wait 10 seconds for foo
    time.sleep(3)

    # Terminate foo
    p.terminate()

    # Cleanup
    p.join()

    print("2 :", choosenAction.value)

    return choosenAction.value


# get food position
def get_food_positions(food):
    res = []
    h, w = food.shape
    for i in range(h):
        for j in range(w):
            if food[i][j] == 1:
                res.append((i, j))
    return res


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


# get distance between our two agents | [-1, 0]
def get_distance_between_our_agents(state):
    pos1 = state.mainPosition
    pos2 = state.teammatePosition
    if manhattan_distance(pos1, pos2) <= DISTANCE_PENALIZED_OUR_AGENTS:
        dist = distance_closest_bfs(state.wall, [pos2], pos1)  # bfs distance
        return min(0, (dist - DISTANCE_PENALIZED_OUR_AGENTS) / DISTANCE_PENALIZED_OUR_AGENTS)
    return 0


# get homebase position
def get_homebase_positions(wall):
    res = []
    h, w = wall.shape
    j = w//2  # vertical coordinate of our homebase
    for i in range(h):
        if wall[i][j] == 0:
            res.append((i, j))
    return res


# get distance to our home_base
def get_distance_homebase(state, index):
    pos = state.mainPosition
    if index == state.teammateIndex:
        pos = state.teammatePosition
    homebase_position = get_homebase_positions(state.wall)
    return distance_closest_bfs(state.wall, homebase_position, pos)  # bfs distance


# calculate penalty for distance to homebase
def calculate_homebase_penalty(state, index, dist_food):
    col = state.mainCollected
    if index == state.teammateIndex:
        col = state.teammateCollected
    if col == 0:  # nothing collected
        return 0

    MAX_DISTANCE_HOMEBASE = state.wall.shape[1] / 2  # w / 2
    dist_homebase = get_distance_homebase(state, index)
    if dist_homebase < dist_food or col > NUMBER_FOOD_COLLECT:  # closest to come back than to catch next food
        return dist_homebase / MAX_DISTANCE_HOMEBASE

    return 0  # no penalty if food closer


# calculate distance to the enemy (<0 when enemy can eat us)
def distance_opponent(state, pos, dist_hb):
    opponent = []
    side_limit = state.wall.shape[1] // 2  # w / 2
    min_distance = np.inf
    if state.opponentPosition1:
        opponent.append(state.opponentPosition1)
    if state.opponentPosition2:
        opponent.append(state.opponentPosition2)

    if not opponent:  # opponents' position unknown
        return MAX_DISTANCE_OPPONENT

    for opp_pos in opponent:
        dist = distance_closest_bfs(state.wall, [opp_pos], pos)  # distance agent - opponent

        if opp_pos[1] < side_limit and pos[1] < side_limit:  # opponent is ghost and we are pacman
            if not state.opponentScared:  # opponent not scared
                if dist > 1 and dist_hb > 0:
                    dist += 17 * (1-dist_hb)  # avoid deadlocks
                    print("dist_opp:", 16.7 * (1-dist_hb))
                dist = - dist  # negative distance
        if opp_pos[1] >= side_limit and pos[1] >= side_limit:  # opponent is pacman and we are ghost
            if state.teamScared:  # we are scared
                if dist > 1 and dist_hb > 0:
                    dist += 17 * (1-dist_hb)  # avoid deadlocks
                    print("dist_opp:", 16.7 * (1 - dist_hb))
                dist = - dist  # negative distance

        if abs(dist) < abs(min_distance):
            min_distance = dist

    """if dist == 0:  # eaten
        return (2 * (pos[1] > side_limit) - 1) * 100"""

    #print("distance_opponent: pos=", pos, "opp=", opponent, min_distance)

    min_distance = min_distance / MAX_DISTANCE_OPPONENT
    if min_distance < 0:  # enemy can eat us
        min_distance = -1.5 - min_distance
        #print("dist_opp < 0: ", dist)
    else:
        min_distance = 1.5 - min_distance
        #print("dist_opp > 0: ", dist)

    #print("distance_opponent :", dist)
    return min_distance


# bfs to find the boxes at the right distance of the agent
def distance_closest_bfs(wall, goals, pos):
    """Calculate with bfs the minimum distance between pos to reach a goal"""
    # preparation
    board = np.full(wall.shape, -1)
    board[pos[0]][pos[1]] = 0  # agent's position

    # BFS
    lookAt = [pos]  # initialize with agent's position
    while lookAt:
        (u, v) = lookAt.pop(0)  # remove first element of lookAt
        distance = board[u][v]
        if (u, v) in goals:  # opponent's food
            return distance

        for dir in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            new_pos_i, new_pos_j = (u + dir[0], v + dir[1])
            if wall[new_pos_i][new_pos_j] == 0 and board[new_pos_i][new_pos_j] == -1:  # accessible and unvisited
                board[new_pos_i][new_pos_j] = distance + 1
                lookAt.append((new_pos_i, new_pos_j))

    print("distance_closest_food | No food found")
    #print(board)
    #exit()  # REMOVE THIS !!!
    return 1000  # no food found


# invert the output for red
def invertOutput(move):
    cor = [((1, 0), (-1, 0)), ((0, 1), (0, -1)), ((-1, 0), (1, 0)), ((0, -1), (0, 1)), ((0, 0), (0, 0))]
    for m, m_cor in cor:
        if move == m:
            return m_cor


def convert_move(color, move):
    conv = [((1, 0), "South"), ((0, 1), "East"), ((-1, 0), "North"), ((0, -1), "West"), ((0, 0), "Stop")]
    if color is 'red':
        move = invertOutput(move)

    for (direction, command) in conv:
        if move == direction:
            return command


# an agent is eaten by another one
def superposition(state):
    team = [state.mainPosition, state.teammatePosition]
    opponent = [state.opponentPosition1, state.opponentPosition2]
    for pos in team:
        for pos_opp in opponent:
            if pos == pos_opp:  # there is a superposition
                return True
    return False


def heuristic_superposition(state):
    print("h_s :", state.opponentScared)

    team = [state.mainPosition, state.teammatePosition]
    opponent = []
    side_limit = state.wall.shape[1] // 2  # w / 2
    if state.opponentPosition1:
        opponent.append(state.opponentPosition1)
    if state.opponentPosition2:
        opponent.append(state.opponentPosition2)

    for i in range(len(team)):
        pos = team[i]
        for opp_pos in opponent:
            if pos == opp_pos:
                if pos[1] < side_limit:  # opponent is ghost and we are pacman
                    if not state.opponentScared:  # opponent not scared
                        if i == 2:  # teammate
                            return -10
                        return -100
                    else:  # opponent scared
                        if i == 2:  # teammate
                            return 10
                        return 100
                if pos[1] >= side_limit:  # opponent is pacman and we are ghost
                    if state.teamScared:  # we are scared
                        if i == 2:  # teammate
                            return -10
                        return -100
                    else:  # we are not scared
                        if i == 2:  # teammate
                            return 10
                        return 100

    print("heu_sup, shouldn't come here")
    return None


def heuristic(state):
    h, w = state.wall.shape
    MAX_DISTANCE_FOOD = w

    score = COEF_SCORE * (state.is_red() * state.score / MAX_SCORE)

    coll = COEF_COLL * (state.mainCollected / MAX_COLLECTED + state.teammateCollected / MAX_COLLECTED)
    coll_opp = COEF_COLL_OPP * (state.opponentCollected / MAX_COLLECTED)
    collected = coll - coll_opp

    foodPosition = get_food_positions(state.food)
    real_distance_food1 = distance_closest_bfs(state.wall, foodPosition, state.mainPosition)
    real_distance_food2 = distance_closest_bfs(state.wall, foodPosition, state.teammatePosition)
    distance_food1 = COEF_DISTANCE_FOOD1 * (real_distance_food1 / MAX_DISTANCE_FOOD)
    distance_food2 = COEF_DISTANCE_FOOD2 * (real_distance_food2 / MAX_DISTANCE_FOOD)
    distance_food = distance_food1 + distance_food2

    distance_our_agents = COEF_DISTANCE_OUR_AGENTS * get_distance_between_our_agents(state)

    distance_homebase1 = COEF_DISTANCE_HOMEBASE1 * calculate_homebase_penalty(state, state.mainIndex, distance_food1)
    distance_homebase2 = COEF_DISTANCE_HOMEBASE2 * calculate_homebase_penalty(state, state.teammateIndex, distance_food2)
    distance_homebase = distance_homebase1 + distance_homebase2

    distance_enemy = COEF_DISTANCE_OPPONENT * distance_opponent(state, state.mainPosition, distance_homebase1)

    res = score + collected - distance_food + distance_our_agents - distance_homebase + distance_enemy

    if state.color is 'blue':
        print("heuristic : pos=", state.mainPosition, "pos2=", state.teammatePosition, "pos3=", state.opponentPosition1,
              "pos4=", state.opponentPosition2, "|| sc=", score, "col=", collected, "df1=", distance_food1,
              "da=", distance_our_agents, "dh1=", distance_homebase1, "do=", distance_enemy, "||", "res=", res)
    # "df2=", distance_food2, "dh2=", distance_homebase2
    return res


# alpha beta algorithm
def alphabeta(state, depth, alpha, beta, player, getBestMove=False):
    """simulate the different possibilities for depth moves and use minimax logic to find the best option"""
    #print("alphabeta : depth=", depth, "player=", player)

    if superposition(state):
        hs = heuristic_superposition(state)
        print("heuristic_superposition:", hs)
        return hs

    if depth == 0:  # or end of game
        #print("aB | depth 0")
        return heuristic(state)

    elif stt.same_team(player, state.mainIndex):  # team MAX
        #print("aB | MAX")
        v = - np.inf
        bestMove = None
        children = state.next_states(player)
        for (move, child) in children:
            v_old = v
            next_player = (player+1) % 4
            v = max(v, alphabeta(child, depth-1, alpha, beta, next_player))
            if getBestMove and v != v_old:
                bestMove = move
            alpha = max(alpha, v)
            if beta <= alpha:  # beta pruning
                break

        print("aB | MAX", player, v)

        if getBestMove:
            return convert_move(state.color, bestMove)

    else:  # team MIN
        #print("aB | MIN")
        v = np.inf

        # Opponent's position unknown
        if not state.position_index(player):  # None : unknown
            print("aB | Opp_pos unknown")
            next_player = (player + 1) % 4
            v = min(v, alphabeta(state, depth, alpha, beta, next_player))  # depth-1 (?) : no simulation here
            """beta = min(beta, v)
            if beta <= alpha:  # alpha pruning
                break"""

        else:
            children = state.next_states(player)
            for (move, child) in children:
                next_player = (player+1) % 4
                v = min(v, alphabeta(child, depth-1, alpha, beta, next_player))
                beta = min(beta, v)
                if beta <= alpha:  # alpha pruning
                    break

        print("aB | MIN", player, v)

    return v
