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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # initialize the closest food as very, very far
        closest_food = 9999
        for food in successorGameState.getFood().asList():
            # and find the actually closest food
            closest_food = min(closest_food, manhattanDistance(newPos, food))

        # if a ghost is
        for ghost in successorGameState.getGhostPositions():
            # uncomfortably close
            if manhattanDistance(newPos, ghost) < 2:
                # run
                return -9999

        return successorGameState.getScore() + 1.0 / closest_food


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        # used by pacman
        def max_finder(gameState, depth):
            # if we need to return
            # depth is always +1 to find depth for max
            if gameState.isWin() or gameState.isLose() or depth + 1 == self.depth:
                return self.evaluationFunction(gameState)
            # initialize maxvalue
            largest = -9999
            # replace if there's any better value
            # actions for 0 because it's asking for pacman's actions
            for action in gameState.getLegalActions(0):
                # 0 because it's asking for pacman's successors
                successor = gameState.generateSuccessor(0, action)
                largest = max(largest, min_finder(successor, depth + 1, 1))
            return largest

        # used by ghosts
        def min_finder(gameState, depth, agentIndex):
            # if we need to return
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            smallest = 9999
            # traversing the tree
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    smallest = min(smallest, max_finder(successor, depth))
                else:
                    smallest = min(smallest, min_finder(successor, depth, agentIndex + 1))
            return smallest

        # initialize score and action
        currentScore = -999999
        returnAction = ''
        for action in gameState.getLegalActions(0):
            # traverse the tree - agent 1 is the ghost
            score = min_finder(gameState.generateSuccessor(0, action), 0, 1)
            # if there's a better action (max successors - based on score) choose that
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        # Used by pacman
        def max_finder(gameState, depth, alpha, beta):
            # if we need to return (and depth is +1 to find depth for max)
            if gameState.isWin() or gameState.isLose() or depth + 1 == self.depth:
                return self.evaluationFunction(gameState)
            # initialize maxvalue
            largest = -9999
            temp_alpha = alpha
            # actions for - because it's asking for pacman's actions
            for action in gameState.getLegalActions(0):
                # 0 because it's asking for pacman's successors
                successor = gameState.generateSuccessor(0, action)
                largest = max(largest, min_finder(successor, depth + 1, 1, temp_alpha, beta))
                # prune: not largest - equal, since stated in the project guide
                if largest > beta:
                    return largest
                temp_alpha = max(temp_alpha, largest)
            return largest

        # used by ghosts
        def min_finder(gameState, depth, agentIndex, alpha, beta):
            # if we need to return
            if gameState.isWin() or gameState.isLose():  # Terminal Test
                return self.evaluationFunction(gameState)
            minvalue = 9999
            temp_beta = beta
            # traversing the tree
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    minvalue = min(minvalue, max_finder(successor, depth, alpha, temp_beta))
                    # prune
                    if minvalue < alpha:
                        return minvalue
                    temp_beta = min(temp_beta, minvalue)
                else:
                    minvalue = min(minvalue, min_finder(successor, depth, agentIndex + 1, alpha, temp_beta))
                    # prune
                    if minvalue < alpha:
                        return minvalue
                    temp_beta = min(temp_beta, minvalue)
            return minvalue

        # initialize score and action
        currentScore = -9999
        returnAction = ''
        alpha = -9999
        beta = 9999
        for action in gameState.getLegalActions(0):
            # Next level is a min level. Hence calling min for successors of the root.
            score = min_finder(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            # if there's a better action (max successors - based on score) choose that
            if score > currentScore:
                returnAction = action
                currentScore = score
            # Updating alpha
            if score > beta:
                return returnAction
            alpha = max(alpha, score)
        return returnAction


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

        # used by pacman
        def max_finder(gameState, depth):
            # if we need to return (and depth is +1 to find depth for max)
            if gameState.isWin() or gameState.isLose() or depth + 1 == self.depth:
                return self.evaluationFunction(gameState)
            # initialize maxvalue
            maxvalue = -9999
            # replace if there's any better value
            # action for 0 because it's asking for pacman's actions
            for action in gameState.getLegalActions(0):
                # 0 because it's asking for pacman's successors
                successor = gameState.generateSuccessor(0, action)
                maxvalue = max(maxvalue, random_min(successor, depth + 1, 1))
            return maxvalue

        # used by ghosts
        def random_min(gameState, depth, agentIndex):
            # if we need to return
            if gameState.isWin() or gameState.isLose():  # Terminal Test
                return self.evaluationFunction(gameState)
            # get actions here
            all_actions = gameState.getLegalActions(agentIndex)
            expectation = 0
            actions_len = len(all_actions)
            # traversing the tree
            for action in all_actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # one agent works in pacman's favor and one doesn't
                if agentIndex == (gameState.getNumAgents() - 1):
                    expectedvalue = max_finder(successor, depth)
                else:
                    expectedvalue = random_min(successor, depth, agentIndex + 1)
                expectation = expectation + expectedvalue
            if actions_len == 0:
                return 0
            return float(expectation) / float(actions_len)

        # initialize score and action
        currentScore = -9999
        returnAction = ''
        for action in gameState.getLegalActions(0):
            # traverse tree
            score = random_min(gameState.generateSuccessor(0, action), 0, 1)
            # if there's a better action (max successors - based on score) choose that
            if score > currentScore:
                returnAction = action
                currentScore = score
        return returnAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    current_position = currentGameState.getPacmanPosition()
    current_food = currentGameState.getFood().asList()
    remaining_food = currentGameState.getNumFood()
    remaining_caps = len(currentGameState.getCapsules())

    closest_food = 9999
    for food in current_food:
        closest_food = min(closest_food, manhattanDistance(current_position, food))

    closest_ghost = 0
    for ghost in currentGameState.getGhostPositions():
        closest_ghost = manhattanDistance(current_position, ghost)
        if closest_ghost < 2:
            return -9999

    return 1.0 / (remaining_food + 1) * 900000 + closest_ghost + \
           1.0 / (closest_food + 1) * 900 + \
           1.0 / (remaining_caps + 1) * 9000 + \
           50000 if currentGameState.isWin() else (-50000 if currentGameState.isLose() else 0)


# Abbreviation
better = betterEvaluationFunction
