import alphabeta
from multiprocessing import Value

choosenAction = Value('i', 0)

if __name__ == '__main__':
    print("hello")
    choosenAction.value = alphabeta.main()
    print("draft :", choosenAction.value)
