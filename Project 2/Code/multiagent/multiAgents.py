# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #chosenIndex = bestIndices[-1]

        "Add more of your code here if you want to"
        #print "best score: {}".format(bestScore)
        #print "bestIndices: {}".format(bestIndices)


        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        foodList = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newCapsules = successorGameState.getCapsules()

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        #score = scoreEvaluationFunction(successorGameState)
        ghostPositions = successorGameState.getGhostPositions()

        for food in foodList:
            score += 1.0/(manhattanDistance(newPos, food))
            #print "foodpart: {}".format(1.0/(manhattanDistance(newPos, food))



        if newFood[newPos[0] - 1][newPos[1]] == True:
            score += 2
           #print "yay111!"

        if newFood[newPos[0]][newPos[1] + 1] == True:
            score += 2
            #print "yay222!"

        if newFood[newPos[0] + 1][newPos[1]] == True:
            score += 2
            #print "yay333!"

        if newFood[newPos[0]][newPos[1] - 1] == True:
            score += 2
            #print "yay4444!"

        for capsule in newCapsules:
            score += 1.0/(manhattanDistance(newPos, capsule))

        if newScaredTimes == 0:
             for ghost in newGhostStates:
                score += 0.2 * manhattanDistance(newPos,ghost.getPosition())
        else:
            score += 0

        if newPos in ghostPositions:
            score -= 9999
        if (newPos[0]-1,newPos[1]) in ghostPositions:
            score -= 9999
        if (newPos[0]+1,newPos[1]) in ghostPositions:
            score -= 9999
        if (newPos[0],newPos[1]-1) in ghostPositions:
            score -= 9999
        if (newPos[0],newPos[1]+1) in ghostPositions:
            score -= 9999

        score += sum(newScaredTimes)
        #for state in newGhostStates:
        #print successorGameState.getGhostPositions()
        return score

def manhattanDistance( xy1, xy2 ):
    "Returns the Manhattan distance between points xy1 and xy2"
    return abs( xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1] )


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        return self.minimaxAction(gameState)


    def minimaxAction(self, state):
        """
        Select the minimax action for pacman
        """
        actions = state.getLegalActions(0)
        maxVal = float("-inf")
        maxAction = None
        for action in actions:
            val = self.minValue(state.generateSuccessor(0,action),1,0)
            if val > maxVal:
                maxVal = val
                maxAction = action

        return maxAction

    def isTerminalState(self, state, currentDepth):
        """
        Checks if the state currently examined is a terminal state,
        ie. state is a win/lose state or maximum search tree depth reached.
        """
        if(state.isWin() or state.isLose()):
            return True
        elif self.depth == currentDepth:
            return True
        else:
            return False

    def maxValue(self, state, currentDepth):
        """
        Recursive search function that returns the max value in minimax search
        """
        if self.isTerminalState(state, currentDepth):
            return self.evaluationFunction(state) # returns the eval value if terminal state is reached
        maxVal = float("-inf")
        actions = state.getLegalActions(0)
        for action in actions:
            successor = state.generateSuccessor(0, action)
            maxVal = max(maxVal, self.minValue(successor, 1, currentDepth))

        return maxVal

    def minValue(self, state, idx, currentDepth):
        """
        Recursice search function that returns the min value in minimax search
        """
        if self.isTerminalState(state, currentDepth):
            return self.evaluationFunction(state) # returns the eval value if terminal state is reached
        minVal = float("inf")
        actions = state.getLegalActions(idx)

        for action in actions:
            successor = state.generateSuccessor(idx, action)

            if idx == state.getNumAgents() -1 :
                # All min players have had their turn --> move to next ply
                minVal = min(minVal, self.maxValue(successor, currentDepth + 1))
            else:
                # Min player turns remaining --> move to next min player
                minVal = min(minVal, self.minValue(successor, idx + 1, currentDepth))

        return minVal

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimaxABAction(gameState)

    def minimaxABAction(self, state):
        """
        Select the minimax action for pacman with alpha-beta pruning
        """

        # Initialize alpha and beta values
        alpha = float("-inf")
        beta = float("inf")

        actions = state.getLegalActions(0)
        maxVal = float("-inf")
        maxAction = None
        for action in actions:
            val = self.minValueAB(state.generateSuccessor(0,action),alpha,beta,1,0)
            if val > maxVal:
                maxVal = val
                maxAction = action
            alpha = max(alpha, val)
        return maxAction

    def isTerminalState(self, state, currentDepth):
        """
        Checks if the state currently examined is a terminal state,
        ie. state is a win/lose state or maximum search tree depth reached.
        """
        if(state.isWin() or state.isLose()):
            return True
        elif self.depth == currentDepth:
            return True
        else:
            return False

    def maxValueAB(self, state,alpha, beta, currentDepth):
        """
        Recursive search function that returns the max value in minimax search with alpha beta pruning
        """

        if self.isTerminalState(state, currentDepth):
           return self.evaluationFunction(state) # returns the eval value if terminal state is reached

        maxVal = float("-inf")
        actions = state.getLegalActions(0)

        for action in actions:
            successor = state.generateSuccessor(0, action)
            maxVal = max(maxVal, self.minValueAB(successor,alpha, beta, 1, currentDepth))
            if maxVal > beta:
               # MIN has already found a better path
               # --> Rest of the successors can be pruned since this node will not be picked by MAX.
               return maxVal
            # Update alpha value if maxVal is better
            alpha = max(alpha, maxVal)

        return maxVal

    def minValueAB(self, state,alpha, beta, idx, currentDepth):
        """
        Recursive search function that returns the min value in minimax search with alpha-beta pruning
        """

        if self.isTerminalState(state, currentDepth):
            return self.evaluationFunction(state) # returns the eval value if terminal state is reached
        minVal = float("inf")
        actions = state.getLegalActions(idx)

        for action in actions:
            successor = state.generateSuccessor(idx, action)
            if idx == state.getNumAgents() -1 :
                # All min players have had their turn --> move to next ply
                minVal = min(minVal, self.maxValueAB(successor,alpha, beta, currentDepth + 1))
                if minVal < alpha:
                    # MAX has already found a beter path
                    # --> Rest of the successors can be pruned since this node will not be picked by MAX
                    return minVal
                beta = min(beta,minVal)

            else:
                # Min player turns remaining --> move to next min player
                minVal = min(minVal, self.minValueAB(successor,alpha, beta,  idx + 1, currentDepth))
                if minVal < alpha:
                   return minVal
                # Update beta if minVal is better
                beta = min(beta,minVal)
        return minVal


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimaxAction(gameState)


    def expectimaxAction(self, state):
        """
        Select the expectimax action for pacman
        """
        actions = state.getLegalActions(0)
        maxVal = float("-inf")
        maxAction = None
        for action in actions:
            val = self.expValue(state.generateSuccessor(0,action),1,0)
            if val > maxVal:
                maxVal = val
                maxAction = action

        return maxAction

    def isTerminalState(self, state, currentDepth):
        """
        Checks if the state currently examined is a terminal state,
        ie. state is a win/lose state or maximum search tree depth reached.
        """
        if(state.isWin() or state.isLose()):
            return True
        elif self.depth == currentDepth:
            return True
        else:
            return False

    def maxValue(self, state, currentDepth):
        """
        Recursive search function that returns the max value in expectimax search
        """
        if self.isTerminalState(state, currentDepth):
            return self.evaluationFunction(state) # returns the eval value if terminal state is reached
        maxVal = float("-inf")
        actions = state.getLegalActions(0)
        successors = []
        for action in actions:
            successors.append(state.generateSuccessor(0, action))

        for successor in successors:
            maxVal = max(maxVal, self.expValue(successor, 1, currentDepth))

        return maxVal

    def expValue(self, state, idx, currentDepth):
        """
        Recursive search function that returns the expected value in expectimax search
        """
        if self.isTerminalState(state, currentDepth):
            return self.evaluationFunction(state) # returns the eval value if terminal state is reached

        expVal = float(0)
        actions = state.getLegalActions(idx)
        successors = []


        for action in actions:
            successors.append(state.generateSuccessor(idx, action))

        for successor in successors:
            #print state.getNumAgents()
            if idx == state.getNumAgents() -1 :
                # All exp players have had their turn --> move to next ply
                expVal = expVal + float(float(float(1/float(len(actions))) * self.maxValue(successor, currentDepth + 1)))
            else:
                # exp player turns remaining --> move to next exp player
                expVal = expVal + float(float(1/float(len(actions))) * self.expValue(successor, idx + 1, currentDepth))

        return expVal


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foodGrid= currentGameState.getFood()
    foodList = foodGrid.asList()
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = currentGameState.getGhostPositions()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()


    # Get base score from game state
    score = currentGameState.getScore()

    for food in foodList:
        score += 1.0/(manhattanDistance(pos, food))

    if foodGrid[pos[0] - 1][pos[1]] == True:
        score += 2

    if foodGrid[pos[0]][pos[1] + 1] == True:
        score += 2

    if foodGrid[pos[0] + 1][pos[1]] == True:
        score += 2

    if foodGrid[pos[0]][pos[1] - 1] == True:
        score += 2

    for capsule in capsules:
        score += 1.0/(manhattanDistance(pos, capsule))

    if scaredTimers == 0:
         for ghost in newGhostStates:
            score += 0.2 * manhattanDistance(pos,ghost.getPosition())
    else:
        score += 0

    if pos in ghostPositions:
        score -= 9999
    if (pos[0]-1,pos[1]) in ghostPositions:
        score -= 9999
    if (pos[0]+1,pos[1]) in ghostPositions:
        score -= 9999
    if (pos[0],pos[1]-1) in ghostPositions:
        score -= 9999
    if (pos[0],pos[1]+1) in ghostPositions:
        score -= 9999

    score += sum(scaredTimers)
    return score

# Abbreviation
better = betterEvaluationFunction

