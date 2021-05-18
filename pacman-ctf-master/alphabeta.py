import multiprocessing
import time
from draft import choosenAction
import numpy as np
import state as stt


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


def convert_move(move):
    conv = [((1, 0), "South"), ((0, 1), "East"), ((-1, 0), "North"), ((0, -1), "West"), ((0, 0), "Stop")]
    for (direction, command) in conv:
        if move == direction:
            return command


# return the value of a state for a player
def heuristic(state):
    # score
    # collected
    # distance coins
    # distance home
    return 0


# alpha beta algorithm
def alphabeta(state, depth, alpha, beta, player, getBestMove=False):
    """simulate the different possibilities for depth moves and use minimax logic to find the best option"""
    if depth == 0:  # or end of game
        print("aB | depth 0")
        return heuristic(state)

    elif stt.same_team(player, state.mainIndex):  # team MAX
        print("aB | MAX")
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
        print("aB | MIN")
        v = np.inf
        # TODO: opponent's position known ?
        if not state.position_index(player):  # None : unknown
            print("aB | Opp_pos unknown")
            next_player = (player + 1) % 4
            v = min(v, alphabeta(state, depth - 1, alpha, beta, next_player))
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
