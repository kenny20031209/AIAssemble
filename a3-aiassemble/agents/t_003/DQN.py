
from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
from util import nearestPoint
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
TRAINING = True 
SAVE_FREQUENCY = 5
####################
# Neural Network   #
####################

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):  # Assuming 5 possible actions
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(1537, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#####################
# DQN Agent         #
#####################

class DQN_Agent(CaptureAgent):

    def __init__(self, index, timeForComputing = .1, epsilon=0.3, alpha=0.2, gamma=0.9):
        CaptureAgent.__init__(self, index, timeForComputing)
        self.epsilon = epsilon
        if TRAINING == False:
            epsilon = 0
        self.alpha = alpha
        self.replay_buffer = ReplayBuffer(10000)
        self.gamma = gamma
        self.q_values = util.Counter()
        self.weights = util.Counter()
        self.model = QNetwork(1537, 5)  # Assuming 5 features and 5 possible actions
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()
        self.episodes_so_far = 0
        self.ACTIONS = ['North', 'South', 'East', 'West', 'Stop']
        self.action_to_int = {"Stop": 0, "North": 1, "South": 2, "East": 3, "West": 4}
        self.int_to_action = {0: "Stop", 1: "North", 2: "South", 3: "East", 4: "West"}
        self.load_weights()
        
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        
    def save_weights(self, filename="model_weights.pth"):
        torch.save(self.model.state_dict(), filename)

    def load_weights(self, filename="model_weights.pth"):
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename))
            self.model.eval()
    
    def gameStateToTensor(self, gameState):
        # Get layout size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        # Initialize matrices to represent agent, food, and ghost positions
        agent_matrix = np.zeros((width, height))
        food_matrix = np.zeros((width, height))
        ghost_matrix = np.zeros((width, height))
    
        # Set agent position
        agent_pos = gameState.getAgentPosition(self.index)
        agent_matrix[agent_pos[0]][agent_pos[1]] = 1
    
        # Set food positions
        food_positions = self.getFood(gameState).asList()
        for pos in food_positions:
            food_matrix[pos[0]][pos[1]] = 1
    
        # Set ghost positions
        for i in self.getOpponents(gameState):
            pos = gameState.getAgentPosition(i)
            if pos:  # Might be None if agent is not observable
                ghost_matrix[pos[0]][pos[1]] = 1
    
        # Flatten matrices and append score
        state_vector = np.concatenate((agent_matrix.flatten(), food_matrix.flatten(), ghost_matrix.flatten(), [gameState.getScore()]))
    
        return state_vector


    def getAction(self, gameState):
        # Implement your DQN logic here
        # Note: For the initial version, we can use a random choice
        actions = gameState.getLegalActions(self.index)
        # Get the successor state using the chosen action
        action = self.select_action(gameState)
        successor = gameState.generateSuccessor(self.index, action)

        # Compute the reward using the current and successor states
        reward = self.compute_reward(gameState, successor)

        # Store this experience in the replay buffer
        int_action = self.action_to_int[action]
        self.replay_buffer.push(gameState, int_action, reward, successor)
        self.episodes_so_far = self.episodes_so_far + 1
        if TRAINING and self.episodes_so_far % SAVE_FREQUENCY == 0:
            self.save_weights()
        if TRAINING:
            self.learn_from_replay()
        return action
    def learn_from_replay(self, BATCH_SIZE=32):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        samples = self.replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*samples)
        
        # Convert game states to tensors
        states = [self.gameStateToTensor(s) for s in states]
        next_states = [self.gameStateToTensor(s) for s in next_states]
        
        # Convert data to PyTorch tensors
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))

        # Compute expected Q values
        predicted_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        predicted_q_values = predicted_q_values.squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            max_next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * max_next_q_values

        # Compute the loss
        loss = self.loss_fn(predicted_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def select_action(self, gameState):
        legal_actions = gameState.getLegalActions(self.index)
        legal_action_indices = [self.ACTIONS.index(action) for action in legal_actions]
    
        # Exploration: select a random action
        if np.random.rand() < self.epsilon:
            return random.choice(legal_actions)

        # Exploitation: select the action with highest Q-value
        with torch.no_grad():
            state_tensor = torch.tensor(self.gameStateToTensor(gameState), dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state_tensor)
            # We only want Q-values of legal actions
            legal_q_values = q_values[0, legal_action_indices]
            action_index = legal_action_indices[torch.argmax(legal_q_values).item()]
            return self.ACTIONS[action_index]

    def compute_reward(self, current_game_state, next_game_state):
        reward = 0

        # Reward for bringing food back to the home territory
        reward += (current_game_state.getScore() - next_game_state.getScore()) * 10

        # Reward for eating an opponent's capsule
        if len(current_game_state.getCapsules()) < len(next_game_state.getCapsules()):
            reward += 5

        # Reward for eating a scared ghost
        current_enemies = [current_game_state.getAgentState(i) for i in self.getOpponents(current_game_state)]
        next_enemies = [next_game_state.getAgentState(i) for i in self.getOpponents(next_game_state)]
        current_scared = sum([1 for enemy in current_enemies if enemy.scaredTimer > 0])
        next_scared = sum([1 for enemy in next_enemies if enemy.scaredTimer > 0])
        if current_scared > next_scared:
            reward += 8

        # Penalty for getting eaten
        if next_game_state.getAgentPosition(self.index) == self.start:
            reward -= 15

        # Small penalty for each step to encourage faster decisions
        if current_game_state.getAgentPosition(self.index) == next_game_state.getAgentPosition(self.index):
            reward -= 1000

            # Check if any opponent has been eaten
        for opponent in self.getOpponents(current_game_state):
            current_pos = current_game_state.getAgentPosition(opponent)
            next_pos = next_game_state.getAgentPosition(opponent)
            if next_pos == current_game_state.getInitialAgentPosition(opponent) and current_pos != next_pos:
                reward += 15  # Assign a positive reward

        # Check if any food has been eaten
        my_food_current = set(current_game_state.getRedFood().asList()) if self.red else set(current_game_state.getBlueFood().asList())
        my_food_next = set(next_game_state.getRedFood().asList()) if self.red else set(next_game_state.getBlueFood().asList())
        if len(my_food_current) > len(my_food_next):
            reward -= 3  # Assign a negative penalty

        return reward

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def createTeam(firstIndex, secondIndex, isRed, first='DQN_Agent', second='DQN_Agent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]
