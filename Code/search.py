# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    stack = util.Stack() # LIFO-stack for storing fringe nodes
    startState = problem.getStartState()
    exploredSet = set() # Set for storing all explored squares
    stack.push(TreeNode(startState, None, None, None))
    while True:
        if(stack.isEmpty()):
            # Return failure if fringe is empty
            return None

        # Get top node from stack
        current = stack.pop()
        if problem.isGoalState(current.state):
            # If current leaf is goal, return the corresponding path
            route = []
            while current.parent != None:
                route.append(current.action)
                current = current.parent
            route.reverse()
            return route
        # Add current node to explored set
        exploredSet.add(hash(current.state))

        # Expand fringe
        successors = problem.getSuccessors(current.state)
        for successor in successors:
            newNode = TreeNode(successor[0],successor[1],successor[2], current)
            if hash(newNode.state) not in exploredSet :
                stack.push(newNode)

    # Should not reach this point, raise exception
    util.raiseNotDefined()

class TreeNode:
    """
    Search-tree node data structure.
    Attributes:
        .state: state of this node
        .action: action to reach state
        .cost: cost of the action
        .parent: Parent-node
    """
    def __init__(self, state, action, cost, parent):
        self.state= state
        self.action = action
        self.cost = cost
        self.parent = parent

    def __hash__(self):
        hsh =  hash(self.state)
        return hsh

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue() # FIFO-queue for storing fringe nodes
    startState = problem.getStartState()
    queue.push(TreeNode(startState, None, None, None))
    exploredSet = set()
    while True:
        if(queue.isEmpty()):
            # Return failure if fringe is empty
            return None

        # Get top node from queue
        current = queue.pop()
        if(problem.isGoalState(current.state)):
            # If current leaf is goal, return the corresponding path
            route = []
            while current.parent != None:
                route.append(current.action)
                current = current.parent
            route.reverse()
            return route

        # Expand fringe
        if(hash(current.state) not in exploredSet):
            successors = problem.getSuccessors(current.state)
            for successor in successors:
                newNode = TreeNode(successor[0],successor[1],successor[2], current)
                exploredSet.add(hash(current.state))
                queue.push(TreeNode(successor[0],successor[1],successor[2], current))

    # Should not reach this point, raise exception
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    queue = util.PriorityQueue() # Priority queue for storing fringe nodes
    startState = problem.getStartState()
    queue.push(TreeNode(startState, None, None, None),0)
    exploredSet = set()
    while True:
        if(queue.isEmpty()):
            # Return failure if fringe is empty
            return None

        # Get top node from queue
        current = queue.pop()
        if(problem.isGoalState(current.state)):
            # If current leaf is goal, return the corresponding path
            route = []
            while current.parent != None:
                route.append(current.action)
                current = current.parent
            route.reverse()
            return route

        # Expand fringe
        if(hash(current.state) not in exploredSet):
            successors = problem.getSuccessors(current.state)
            for successor in successors:
                newNode = TreeNode(successor[0],successor[1],successor[2], current)
                exploredSet.add(hash(current.state))
                queue.push(newNode,getCost(newNode))

    # Should not reach this point, raise exception
    util.raiseNotDefined()

def getCost(treeNode):
    current = treeNode
    cost = current.cost
    while current.parent != None and current.parent.cost != None:
       cost += current.parent.cost
       current = current.parent
    return cost

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    queue = util.PriorityQueue() # Priority queue for storing fringe nodes
    startState = problem.getStartState()
    queue.push(TreeNode(startState, None, None, None),0)
    exploredSet = set()
    while True:
        if(queue.isEmpty()):
            # Return failure if fringe is empty
            return None

        # Get top node from queue
        current = queue.pop()
        if(problem.isGoalState(current.state)):
            # If current leaf is goal, return the corresponding path
            route = []
            while current.parent != None:
                route.append(current.action)
                current = current.parent
            route.reverse()
            return route

        # Expand fringe
        if(hash(current.state) not in exploredSet):
            successors = problem.getSuccessors(current.state)
            for successor in successors:
                newNode = TreeNode(successor[0],successor[1],successor[2], current)
                exploredSet.add(hash(current.state))
                #Calculate combined cost
                combinedCost = getCost(newNode) + heuristic(newNode.state, problem)
                queue.push(newNode,combinedCost)

    # Should not reach this point, raise exception
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
