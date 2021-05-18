from captureAgents import CaptureAgent
from game import Directions


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

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        # test detection of opponents' positions with eaten food
        '''if self.color is 'red':
            dataPreProcessed = gameState.dataInput(self)
            self.setNewFoodLastStep(gameState)
            return "Stop"
        '''

        actions = gameState.getLegalActions(self.index)

        print("alpha beta agent : ", actions[0])
        return actions[0]
