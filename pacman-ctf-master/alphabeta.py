import multiprocessing
import time
from draft import choosenAction
import numpy as np


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


# return the value of a state for a player
def heuristic(state, player):
    return 0


# return the next states after the different possible moves for a state and a player
def nextStates(state, player):
    return []


# TODO : player 1,2,3,4 or index?!
# alpha beta algorithm
def alphabeta(state, depth, alpha, beta, player):
    """simulate the different possibilities for depth moves and use minimax logic to find the best option"""
    if depth == 0:  # or end of game
        return heuristic(state, player)

    elif player is 'red':
        v = - np.inf
        children = nextStates(state, player)
        for child in children:
            v = max(v, alphabeta(child, depth-1, alpha, beta, 'blue'))
            alpha = max(alpha, v)
            if beta <= alpha:  # beta pruning
                break

    else:  # 'blue'
        v = np.inf
        children = nextStates(state, player)
        for child in children:
            v = min(v, alphabeta(child, depth-1, alpha, beta, 'red'))
            beta = min(beta, v)
            if beta <= alpha:  # alpha pruning
                break

    return v
