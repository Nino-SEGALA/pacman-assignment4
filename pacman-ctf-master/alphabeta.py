import multiprocessing
import time
from draft import choosenAction
import numpy as np
import state as stt


# PARAMETERS
MAX_SCORE = 20
MAX_COLLECTED = 20
DISTANCE_PENALIZED_OUR_AGENTS = 6
MAX_DISTANCE_FOOD = 1  # h * w or w

COEF_SCORE = 10
COEF_COLL1 = 5
COEF_COLL2 = 3
COEF_DISTANCE_FOOD = 1
COEF_DISTANCE_OUR_AGENTS = 1



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
    print(board)
    exit()  # REMOVE THIS !!!
    return 1000  # no food found


def convert_move(move):
    conv = [((1, 0), "South"), ((0, 1), "East"), ((-1, 0), "North"), ((0, -1), "West"), ((0, 0), "Stop")]
    for (direction, command) in conv:
        if move == direction:
            return command


# return the value of a state for a player
def heuristic(state):
    h, w = state.wall.shape
    MAX_DISTANCE_FOOD = w
    score = COEF_SCORE * (state.score / MAX_SCORE)
    coll1 = COEF_COLL1 * (state.mainCollected / MAX_COLLECTED + state.teammateCollected / MAX_COLLECTED)
    coll2 = COEF_COLL2 * (state.opponentCollected / MAX_COLLECTED)
    collected = coll1 - coll2
    foodPosition = get_food_positions(state.food)
    real_distance_food = distance_closest_bfs(state.wall, foodPosition, state.mainPosition)
    distance_food = COEF_DISTANCE_FOOD * (real_distance_food / MAX_DISTANCE_FOOD)
    distance_our_agents = COEF_DISTANCE_OUR_AGENTS * get_distance_between_our_agents(state)
    # distance home
    res = score + collected - distance_food + distance_our_agents
    print("h :", coll1, coll2)
    print("heuristic :", state.mainPosition, " |", score, collected, distance_food, distance_our_agents, " ||", res)
    return res


# alpha beta algorithm
def alphabeta(state, depth, alpha, beta, player, getBestMove=False):
    """simulate the different possibilities for depth moves and use minimax logic to find the best option"""
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

        print("aB | MAX", v)

        if getBestMove:
            return convert_move(bestMove)

    else:  # team MIN
        #print("aB | MIN")
        v = np.inf

        # Opponent's position unknown
        if not state.position_index(player):  # None : unknown
            #print("aB | Opp_pos unknown")
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

    return v
