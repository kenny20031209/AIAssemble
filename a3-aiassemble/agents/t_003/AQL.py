from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
import collections
import operator
from util import nearestPoint
from util import Queue
from util import PriorityQueue


TRAINING = False

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Agent1', second = 'Agent2'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class ApproxQLearning(CaptureAgent): 
    def registerInitialState(self, gameState): 
        self.epsilon=0.1
        self.alpha=0.2 
        self.discount=0.9
        self.weights = {'closest food': -6.8749077328451, 'bias': -22.70846720199332, 'distance to ghost': 13.28476402592656, 'distance to pacman': -25.2408978796536, 'eats food': 30.745567010663386, 'homedist': - 15}
        self.start=gameState.getAgentPosition(self.index)
        self.featureSelection=FeaturesExtractor(self)
        CaptureAgent.registerInitialState(self, gameState)
        self.enemyCarryFood = self.getFood(gameState).asList()
    def chooseAction(self, gameState):
        legalActions = gameState.getLegalActions(self.index)
        if not legalActions:
            return None
        remainingFood = len(self.getFood(gameState).asList())
        if remainingFood <= 2 or (gameState.data.timeleft < 100 and gameState.getAgentState(self.index).numCarrying > 0):
            minDist = float('inf')
            for act in legalActions:
                nextStep = gameState.generateSuccessor(self.index, act)
                nextPos = nextStep.getAgentPosition(self.index)
                currentDist = self.getMazeDistance(self.start, nextPos)
                if currentDist < minDist:
                    optimalAction = act
                    minDist = currentDist
            return optimalAction
        myPos = gameState.getAgentPosition(self.index)
        adversaries = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        activeGhosts = [foe.getPosition() for foe in adversaries if (not foe.isPacman or gameState.getAgentState(self.index).scaredTimer > 0) and foe.getPosition() is not None]
        
        if activeGhosts and len(self.getCapsules(gameState)) > 0:
            if all(self.getMazeDistance(myPos, ghost) > self.getMazeDistance(myPos, self.getCapsules(gameState)[0]) + 1 for ghost in activeGhosts):
                minDist = float('inf')
                for act in legalActions:
                    nextStep = gameState.generateSuccessor(self.index, act)
                    nextPos = nextStep.getAgentPosition(self.index)
                    currentDist = self.getMazeDistance(self.getCapsules(gameState)[0], nextPos)
                    if currentDist < minDist:
                        optimalAction = act
                        minDist = currentDist
                return optimalAction
        
        selectedAction = None
        if TRAINING:
            if util.flipCoin(self.epsilon):
                selectedAction = random.choice(legalActions)
                self.epsilon = self.epsilon*0.95
            else:
                selectedAction = self.determineBestPolicy(gameState)
                self.adjustWeights(gameState, selectedAction)
        else:
            selectedAction = self.determineBestPolicy(gameState)
        return selectedAction
        

    def getweights(self): 
        return self.weights

    def computeQValue(self, gameState, action): 
        next_state = gameState.generateSuccessor(self.index, action)
        features = self.featureSelection.getFeatures(gameState, next_state)
        return features * self.weights

    def modify(self, gameState, action, nextState, feedback):
        futureState = gameState.generateSuccessor(self.index, action)
        extractedFeatures = self.featureSelection.getFeatures(gameState, futureState)
        presentQ = self.computeQValue(gameState, action)
        futureMaxQ = self.maxQvalue(nextState)
        correction = (feedback + self.discount * futureMaxQ) - presentQ
        self.weights = {key: weight + self.alpha * correction * extractedFeatures[key] for key, weight in self.weights.items()}

    def adjustWeights(self,gameState,action):
        nextState = gameState.generateSuccessor(self.index,action) 
        reward = self.getReward(gameState,nextState)
        self.modify(gameState,action, nextState,reward)
        
    def detect_dead_ends(self,gameState):
        """
        Detects dead ends in the maze and returns their entrance positions.

        Args:
        gameState: The game object which has the getWalls method.

        Returns:
        List of tuples, each tuple represents a position (x, y) which is an entrance to a dead end.
        """
        walls = gameState.getWalls()
        height, width = walls.height, walls.width
        dead_end_entrances = []

        # Check each cell in the maze
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:  # Check only if the cell is not a wall
                    # Count how many walls surround the current cell
                    num_adj_walls = sum([walls[x + dx][y + dy] for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]])
                    # If 3 walls are surrounding the cell, it's the entrance of a dead-end
                    if num_adj_walls == 3:
                        dead_end_entrances.append((x, y))

        return dead_end_entrances
    def find_dead_end_cells_from_entrance(self, x, y, walls, visited):
        if (x, y) in visited or walls[x][y]:  # Stop if position is already visited or it's a wall
            return []
    
        visited.add((x, y))
    
        # List of possible moves
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        cells_in_dead_end = [(x, y)]
        
        num_adj_walls = sum([walls[x + dx][y + dy] for dx, dy in directions])
        if num_adj_walls < 2:
            return [] 
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < walls.width and 0 <= new_y < walls.height:  # Check boundaries
                cells_in_dead_end.extend(self.find_dead_end_cells_from_entrance(new_x, new_y, walls, visited)) 
        return cells_in_dead_end

    def findDeadCells(self, gameState):
        """
        Finds all cells inside all dead-ends in the maze.
    
        Args:
        gameState: The game state object which has the getWalls method.
        
        Returns:
        A list of positions inside all dead-ends.
        """
        walls = gameState.getWalls()
        dead_end_entrances = self.detect_dead_ends(gameState)
        visited = set()
        all_dead_end_cells = []
        
    
        for entrance in dead_end_entrances:
            all_dead_end_cells.extend(self.find_dead_end_cells_from_entrance(entrance[0], entrance[1], walls, visited))
    
        return all_dead_end_cells
    
    def getReward(self,gameState,nextState):
        reward = 0 
        agentPosition = gameState.getAgentPosition(self.index)
        
        myFoods = self.getFood(gameState).asList( )
        distToFood = min([self.getMazeDistance(agentPosition, food)
            for food in myFoods])
        if distToFood == 1:
            nextFoods = self.getFood(nextState).asList()
            if len(myFoods) - len(nextFoods) == 1:
                reward = 8

        # Reward for bringing food back to the home territory
        reward += (gameState.getScore() - nextState.getScore()) * 20
        
        deadEnds = self.detect_dead_ends(gameState)
        if agentPosition == nextState.getAgentPosition(self.index):
            reward -= 20
        
        team = [gameState.getAgentState(i) for i in self.getTeam(gameState)]
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a.getPosition()
            for a in enemies if ((not a.isPacman or team[0].scaredTimer > 0) and a.getPosition()!=None)]
        deadCell = self.findDeadCells(gameState)
        if ghosts:
            distance = min(self.getMazeDistance(gameState.getAgentPosition(self.index), i) for i in ghosts)
            if distance < 7:
                if nextState.getAgentPosition(self.index) in deadEnds:
                    reward -= 50

                        
        # Reward for eating a scared ghost
        current_enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        next_enemies = [nextState.getAgentState(i) for i in self.getOpponents(nextState)]
        current_scared = sum([1 for enemy in current_enemies if enemy.scaredTimer > 0])
        next_scared = sum([1 for enemy in next_enemies if enemy.scaredTimer > 0])
        if current_scared > next_scared:
            reward += 20
        
        if nextState.getAgentPosition(self.index) == gameState.getAgentPosition(self.index):
            reward -= 2
        # Penalty for getting eaten
        if nextState.getAgentPosition(self.index) == self.start:
            reward -= 70

        # Check if any opponent has been eaten
        for opponent in self.getOpponents(gameState):
            current_pos = gameState.getAgentPosition(opponent)
            next_pos = nextState.getAgentPosition(opponent)
            if next_pos == gameState.getInitialAgentPosition(opponent):
                reward += 20  # Assign a positive reward

            
        if nextState.isOver():
            if nextState.getScore() > 0:
                reward += 500  # Huge reward for winning
            else:
                reward -= 500  # Huge penalty for losing
        return reward
    def maxQvalue(self,gameState):
        allowedActions = gameState.getLegalActions(self.index)
        if len(allowedActions) == 0:
            return 0.0
        bestAction = self.determineBestPolicy(gameState)
        return self.computeQValue(gameState,bestAction)
        
    def determineBestPolicy(self, gameState):
        legalActions = gameState.getLegalActions(self.index)
        legalActions.remove('Stop')
        if not legalActions:
            return None
                               
        action_vals = {action: self.computeQValue(gameState, action) for action in legalActions}
        best_q_value = max(action_vals.values())
        best_actions = [action for action,
                        q_value in action_vals.items() if q_value == best_q_value]
        return random.choice(best_actions)
    
    def final(self,state):
        CaptureAgent.final(self, state)
class FeaturesExtractor:
    def __init__(self, agentInstance):
        self.agentInstance = agentInstance
    def getFeatures(self, gameState, next_state):
        # extract the grid of food and wall locations and get the ghost locations 
        width, height = gameState.data.layout.width, gameState.data.layout.height
        food = self.agentInstance.getFood(gameState)
        walls = gameState.getWalls()
        enemies = [gameState.getAgentState(i) for i in self.agentInstance.getOpponents(gameState)]
        team = [gameState.getAgentState(i) for i in self.agentInstance.getTeam(gameState)]
        agentPosition = gameState.getAgentPosition(self.agentInstance.index) 
        nextPosition = next_state.getAgentPosition(self.agentInstance.index)
        teammate = []
        
        if self.agentInstance.index == self.agentInstance.getTeam(gameState)[0]:
            teammate = gameState.getAgentPosition(self.agentInstance.getTeam(gameState)[1])
        else:
            teammate = gameState.getAgentPosition(self.agentInstance.getTeam(gameState)[0])
        features=util.Counter() 
        middle_line = gameState.data.layout.width // 2
                
        
        ghosts = [a.getPosition()
            for a in enemies if not a.isPacman and a.getPosition()!=None] 
        pacman = [a.getPosition()
            for a in enemies if (a.isPacman or a.scaredTimer > 10) and a.getPosition()!=None] 
        
        ghosts = [g for g in ghosts if g not in pacman]
                
        features["bias"]=1.0
        deadCell = self.agentInstance.findDeadCells(gameState)
        deadEnds = self.agentInstance.detect_dead_ends(gameState)

        
        if pacman:
            distance = min(self.agentInstance.getMazeDistance(nextPosition, i) for i in pacman)
            if distance < 7:
                features["distance to pacman"] = distance / (walls.width * walls.height)
        
        if food[nextPosition[0]][nextPosition[1]]: 
            features["eats food"] = 1.0
        
        lis = walls.asList()
        dist =self.closestFood(nextPosition,food,lis,teammate)  / (walls.width * walls.height)
        
        if ghosts:
            distance = min(self.agentInstance.getMazeDistance(nextPosition, i) for i in ghosts)
            if distance < 7:
                for i in deadCell:
                    lis.append(i)
                for i in deadEnds:
                    lis.append(i)   
                dist =self.closestFood(nextPosition,food,lis,teammate)  / (walls.width * walls.height)

        if dist is not None:
            features["closest food"] = float(dist) / (walls.width * walls.height)
        if ghosts:
            distance = min(self.agentInstance.getMazeDistance(nextPosition, i) for i in ghosts)
            if distance < 7:
                features["distance to ghost"] = distance / (walls.width * walls.height)
                if gameState.getAgentState(self.agentInstance.index).numCarrying > 2:
                    features["eats food"] = 0
                    features["closest food"] = 0
                    features["homedist"] = self.agentInstance.getMazeDistance(nextPosition, self.agentInstance.start) / (walls.width * walls.height)
        
        if gameState.getAgentPosition(self.agentInstance.index) in deadCell or gameState.getAgentPosition(self.agentInstance.index) in deadEnds:
            features["distance to ghost"] = 0
            
        features.divideAll(10.0)
        
        
        return features
    def closestFood(self, pos, food, walls,teammate):
        # Get the positions of all the food
        food_positions = food.asList()
    
        # If there's no food left, return None
        if not food_positions:
            return None

        # Calculate the maze distance to each food and return the minimum distance
        distances = [self.agentInstance.getMazeDistance(pos, food_pos) for food_pos in food_positions]
        teamMateDist = [self.agentInstance.getMazeDistance(teammate, food_pos) for food_pos in food_positions]
        if len(distances) > 2:
            if min(distances) > min(teamMateDist):
                distances.remove(min(distances))
                distances.remove(min(distances))
        return min(distances)

class Agent1(ApproxQLearning):
    pass

class Agent2(ApproxQLearning):
    pass