# myTeam.py
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

import numpy as np
from game import Actions
from util import PriorityQueue
from game import Directions
from captureAgents import CaptureAgent
import util,random,time
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Agent', second = 'Agent'):
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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class Agent(CaptureAgent):

    def registerInitialState(self, gameState):
        '''
        Initialize includinig state define and calculation
        '''
        CaptureAgent.registerInitialState(self, gameState)
        self.state = gameState
        self.teamMate_index=[x for x in self.getTeam(gameState) if x!=(self.index)][0]
        
    def MCTS(self, root, iterations=100):
        '''
        Use MCTS
        '''
        for _ in range(iterations):
            leaf = root.expand()
            reward = leaf.reward()
            leaf.backpropagate(reward)
        return root.mcts_search() 
    
    def chooseAction(self, gameState):
        '''
        Determines the best action for the agent based on the current game state.
    
        Strategy:
        1. Check if the agent is being chased by an enemy.
        2. Decide if the agent should chase an opponent.
        3. If the agent is set to chase an opponent, use a Monte Carlo Tree Search (MCTS) strategy.
        4. If the agent is being chased, decide on an escape strategy.
        5. If neither, proceed to consume dots or return home based on the game situation.
    
        Returns:
            The best action for the agent to take based on the current game state.
        '''
        agent_position = gameState.getAgentState(self.index).getPosition()

        # Determine if chased
        being_chased = False
        ChaseOpponent = False
        for current_enemy, previous_enemy in zip(self.getEnemyGhostIndices(self.state), self.getEnemyGhostIndices(gameState)):
            if current_enemy == previous_enemy:
                current_distance = self.getMazeDistance(agent_position, gameState.getAgentState(current_enemy).getPosition())
                previous_distance = self.getMazeDistance(agent_position, gameState.getAgentState(previous_enemy).getPosition())
                if current_distance <= previous_distance <= 4: # If the enemy is closer than before and within a distance of 4
                    being_chased = True
                    break
        # Determine if chase enemy 
        if gameState.getAgentState(self.index).scaredTimer > 0:
            ChaseOpponent = False
        else:
            my_positions = [self.getMazeDistance(agent_position, gameState.getAgentState(i).getPosition()) 
                            for i in self.getEnemyPacmanIndices(gameState)]
            teammate_positions = [self.getMazeDistance(gameState.getAgentState(self.teamMate_index).getPosition(), 
                                                       gameState.getAgentState(i).getPosition()) 
                                  for i in self.getEnemyPacmanIndices(gameState)]
            if sum(my_positions) < sum(teammate_positions):
                ChaseOpponent = True
        # using MCTS to chase enemy
        if ChaseOpponent:
            root = MCTSNode(gameState, self, None, None, [gameState.getAgentState(i).getPosition() for i in self.getEnemyGhostIndices(gameState)], getBoundaryLine(gameState, self.red))
            return self.MCTS(root)
        # If chased, run away, if not, eat food.
        else:
            agentState = gameState.getAgentState(self.index)
            if not agentState.isPacman:
                resultAction = self.consumeDots(gameState)
            else:
                if (len(self.getFood(gameState).asList()) <= 2):
                    resultAction = Directions.STOP if len(self.wayToOurSide(gameState)) == 0 else self.wayToOurSide(gameState)[0]
                elif being_chased:
                    #go to our side or go to capsule
                    way = self.wayToOurSide(gameState)
                    if way and len(self.getFood(gameState).asList()) <= 2: return way[0]
                    paths = [(self.unifiedSearch(gameState, agent_position, [c], self.getDangerousPositions(gameState),strategy='a_star')) for c in self.getCapsules(gameState)]
                    paths = [p for p in paths if p]
                    resultAction = min(paths, key=len)[0] if paths else (Directions.STOP if not way else way[0])
                else:
                    resultAction = self.consumeDots(gameState)   
        
        self.state = gameState
        return resultAction
    """
    def chase_or_not(self, gameState):
        mindist = 9999
        if len(self.getCapsulesYouAreDefending(gameState)) == 0:
            return True
        for i in self.getEnemyPacmanIndices(gameState):
            for j in self.getCapsulesYouAreDefending(gameState):
                if mindist > self.getMazeDistance(gameState.getAgentState(i).getPosition(), j) and self.getMazeDistance(gameState.getAgentState(i).getPosition(), j) < 5:
                    mindist = self.getMazeDistance(gameState.getAgentState(i).getPosition(), j)
        print(mindist)
        teammindist = 999
        for j in self.getCapsulesYouAreDefending(gameState):
            if self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), j) < teammindist and self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), j) < 5:
                teammindist = self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), j)
        if teammindist > mindist:
            return False
        return True;
    """
    def get_dot_distances(self, gameState, position, dots):
        """Calculate the path length to each dot for the given position."""
        danger_positions = self.getDangerousPositions(gameState)
        return {dot: len(path) for dot in dots if (path := self.unifiedSearch(gameState, position, [dot], danger_positions, strategy='greedy'))}

    def consumeDots(self, gameState):
        '''
        Returns an action to target and consume dots in the game.
    
        Strategy:
        1. Determine the current position of this agent and its teammate.
        2. Sort the dots based on proximity to this agent.
        3. Calculate the maze distance from this agent and its teammate to each dot.
        4. If there are no available dots, initiate a return to the home side.
        5. If the closest dot is closer to the teammate, opt for the second closest dot (if available).
        6. If none of the above conditions apply, use a greedy search approach to target the closest dot.
    
        Returns:
        action (str): The best action for the agent to take in order to consume dots.
        '''
 
        # Identify positions of all dots and current positions of the agents.
        my_position = gameState.getAgentState(self.index).getPosition()
        teammate_position = gameState.getAgentState(self.teamMate_index).getPosition()

        # Sort the dots based on their proximity to this agent.
        nearest_dots = sorted(self.getFood(gameState).asList(), key=lambda dot: self.getMazeDistance(my_position, dot))

        # Calculate distances from both agents to each dot.
        my_dot_distances = self.get_dot_distances(gameState, my_position, nearest_dots)
        teammate_dot_distances = self.get_dot_distances(gameState, teammate_position, nearest_dots)

        # If no dots are available, return to the home side.
        if not my_dot_distances:
            return Directions.STOP if len(self.wayToOurSide(gameState)) == 0 else self.wayToOurSide(gameState)[0]

        # Identify the closest dot to this agent.
        sorted_dots = sorted(my_dot_distances.items(), key=lambda x: x[1])
        closest_dot = sorted_dots[0][0]

        # If the teammate is closer to the closest dot, choose the second closest dot (if available).
        if teammate_dot_distances.get(closest_dot, float('inf')) < my_dot_distances[closest_dot]:
            closest_dot = sorted_dots[1][0] if len(sorted_dots) > 1 else sorted_dots[0][0]

        # Default strategy: Use a greedy search to target the closest dot.
        return self.unifiedSearch(gameState, my_position, [closest_dot], self.getDangerousPositions(gameState), strategy='greedy')[0]

    def get_successors(self, gameState, node, lis):
        """Return the successors of a node which aren't walls or in the list of dangerous places"""
        actions = [Directions.WEST, Directions.NORTH, Directions.EAST, Directions.SOUTH]
        x, y = node

        successors = []
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            next_node = (int(x + dx), int(y + dy))
            if not gameState.getWalls()[next_node[0]][next_node[1]] and next_node not in lis:
                successors.append((next_node, action))

        return successors

    def unifiedSearch(self, gameState, start, goal, lis, strategy='greedy'):
        """Perform a greedy search and astar using Maze distance as a heuristic"""
        explored = set()
        frontier = PriorityQueue() if strategy == 'a_star' else []

        # Initial node, path, and cost
        if strategy == 'a_star':
            frontier.push((start, [], 0), 0)
        else:
            frontier.append((start, [], 0))

        while not frontier.isEmpty() if strategy == 'a_star' else frontier:
            # Get the next node based on the strategy
            node, path, cost = frontier.pop() if strategy == 'a_star' else frontier.pop(0)

            if node in explored:
                continue

            explored.add(node)
            if node in goal:
                return path  

            successors = self.get_successors(gameState, node, lis)
            for next_node, action in successors:
                new_cost = cost + 1

                if strategy == 'a_star':
                    heuristic = max(self.getMazeDistance(next_node, g) for g in goal)
                    total_cost = new_cost + heuristic  # Total cost for A*
                    frontier.push((next_node, path + [action], new_cost), total_cost)
                else:  # greedy
                    heuristic = self.getMazeDistance(next_node, goal[0])
                    # Append and sort for greedy
                    frontier.append((next_node, path + [action], heuristic))
                    frontier.sort(key=lambda x: x[2])

        return None  # Return None if no path is found

    def getEnemyGhostIndices(self, gameState):
        """Get the enemy agents' indices which are in ghost mode and not scared"""
        enemy_positions = [i for i in self.getOpponents(gameState) if gameState.getAgentPosition(i) is not None]
        return [i for i in enemy_positions if not gameState.getAgentState(i).isPacman and gameState.getAgentState(i).scaredTimer <= 5]

    def getDangerousPositions(self, gameState):
        """Get a list of positions which are regarded dangerous for my agent"""
        dangerous_positions = []
        enemyGhostPositions = [gameState.getAgentState(i).getPosition() for i in self.getEnemyGhostIndices(gameState)]
        EnemyPacmanPositions = [gameState.getAgentState(i).getPosition() for i in self.getEnemyPacmanIndices(gameState)]
        dangerous_positions.extend(enemyGhostPositions)

        # When scared, get away from enemy pacman
        if gameState.getAgentState(self.index).scaredTimer > 0:  
            dangerous_positions.extend(EnemyPacmanPositions + [pos for enemyPacPos in EnemyPacmanPositions for pos in AdjPos(gameState, enemyPacPos)])

        # Do not go to dead ends
        for enemyPos in enemyGhostPositions:
            dangerous_positions.extend(AdjPos(gameState, enemyPos))

        return dangerous_positions 
    
    def wayToOurSide(self, gameState):
        """Find the shortest path to our side of maze"""
        return min([(path, len(path)) for boundary in getBoundaryLine(gameState, self.red) 
                             if (path := self.unifiedSearch(gameState, gameState.getAgentState(self.index).getPosition(), [boundary], self.getDangerousPositions(gameState),strategy='a_star'))],key=lambda x: x[1], default=(None,))[0] or []

    def getEnemyPacmanIndices(self, gameState):
        """Get the enemy agents' indices which are in Pacman mode"""
        enemy_positions = [i for i in self.getOpponents(gameState) if gameState.getAgentPosition(i) is not None]
        return [i for i in enemy_positions if gameState.getAgentState(i).isPacman]
     
# other functions #

def getBoundaryLine(gameState, red, mySide=True):
    #return boundaries positions for my side and opponent side except walls 
    a = gameState.data.layout.width // 2 - 1 if (red and mySide) or (not red and not mySide) else gameState.data.layout.width // 2
    return [(a, y) for y in range(1, gameState.data.layout.height) if not gameState.hasWall(a, y)]

def AdjPos(gameState, pos):
    #return adjacent positions 1 step away from north,west,east,south except walls
    x, y = int(pos[0]), int(pos[1])
    return {(a, b) for a, b in [(x-1, y), (x+1, y), (x, y+1), (x, y-1)] if not gameState.hasWall(a, b)}

class MCTSNode:
    def __init__(self, gameState, agent, action, parent, enemy_pos, borderline):
        self.initialize_node(gameState, agent, action, parent, enemy_pos, borderline)

    def initialize_node(self, gameState, agent, action, parent, enemy_pos, borderline):
        """Initializes the MCTS node with necessary attributes."""
        self.parent = parent
        self.children = []     
        self.q_value = 0.0
        self.enemy_pos = enemy_pos
        self.legalActions = [act for act in gameState.getLegalActions(agent.index) if act != 'Stop']
        self.depth = parent.depth + 1 if parent else 0
        self.gameState = gameState.deepCopy()
        self.unexploredActions = self.legalActions[:]
        self.borderline = borderline
        self.visits = 1
        self.action = action
        self.agent = agent
        self.rewards = 0

    def getEnemyPacmanPositions(self):
        """Returns the positions of enemy Pacman agents."""
        enemy_pacman_positions = [
            self.gameState.getAgentState(i).getPosition()
            for i in self.agent.getOpponents(self.gameState)
            if self.gameState.getAgentState(i).isPacman
        ]
        return enemy_pacman_positions

    def expand(self):
        """Expands the node by adding a child or returning itself if max depth is reached."""
        if self.depth >= 15:  # max depth
            return self

        if self.unexploredActions:
            return self.create_child()

        next_best_node = self.select_best_child() if util.flipCoin(1) else random.choice(self.children)
        return next_best_node.expand()

    def create_child(self):
        """Creates and returns a new child node."""
        action = self.unexploredActions.pop()
        child_node = MCTSNode(
            self.gameState.deepCopy().generateSuccessor(self.agent.index, action),
            self.agent, action, self, self.enemy_pos, self.borderline
        )
        self.children.append(child_node)
        return child_node

    def mcts_search(self):
        """Performs the MCTS search and returns the best action found."""
        end_time = time.time() + 0.5  

        while time.time() < end_time:
            selected_node = self.expand()
            selected_node.backpropagate(selected_node.reward())  

        return self.select_best_child().action

    def select_best_child(self, c=1.0):
        """Selects and returns the best child node based on the UCT value."""
        return max(self.children, key=lambda child: (
            child.q_value / child.visits + c * np.sqrt(2 * np.log(self.visits) / child.visits)
        ))

    def backpropagate(self, reward):
        """Backpropagates the reward up to the root node."""
        self.visits += 1
        self.q_value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def reward(self):
        """Calculates and returns the reward for the current node."""
        # substantial rewards the agent if it kills the enemy Pacman
        my_pos = self.gameState.getAgentPosition(self.agent.index)
        enemy_pacman_positions = self.getEnemyPacmanPositions()

        enemy_pacman_positions = [pos for pos in enemy_pacman_positions if pos is not None]

        if not enemy_pacman_positions:
            return -1

        distances_to_enemy_pacman = [self.agent.getMazeDistance(my_pos, pos) for pos in enemy_pacman_positions]
        closest_enemy_distance = min(distances_to_enemy_pacman)

        reward = -closest_enemy_distance

        if closest_enemy_distance == 0:
            reward += 1000
        return reward
    
 