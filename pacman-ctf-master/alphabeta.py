import multiprocessing
import time
from draft import choosenAction
import numpy as np
import state as stt


# PARAMETERS



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


# bfs to find the boxes at the right distance of the agent
def distance_closest_food(wall, food, pos):
    """print()
    print("distance_closest_food")
    print(wall)
    print(food)"""
    res = []
    # preparation
    board = np.full(wall.shape, -1)
    board[pos[0]][pos[1]] = 0  # agent's position

    # BFS
    lookAt = [pos]  # initialize with agent's position
    while lookAt:
        (u, v) = lookAt.pop(0)  # remove first element of lookAt
        distance = board[u][v]
        #print(lookAt, (u, v), distance)
        if food[u][v] == 1:  # opponent's food
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
    score = state.score
    collected = (state.mainCollected + state.teammateCollected) - state.opponentCollected
    distance_coins = distance_closest_food(state.wall, state.food, state.mainPosition)
    # distance home
    print("heuristic :", score, collected, distance_coins)
    res = score + collected - distance_coins
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
