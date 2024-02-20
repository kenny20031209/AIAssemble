# AI Method 1 - Approximate Q-learning

Your notes about this part of the project, including acknowledgement, comments, strengths and limitations, etc.

If you use Greedy Best First Search (GBFS) or Monte Carlo Tree Search (MCTS), you do not need to explain the algorithm GBFS/MCTS, tell us how you used it and how you applied it in your team.

# Table of Contents
- [Governing Strategy Tree](#governing-strategy-tree)
  * [Motivation](#motivation)
  * [Application](#application)
  * [Trade-offs](#trade-offs)     
     - [Advantages](#advantages)
     - [Disadvantages](#disadvantages)
  * [Future improvements](#future-improvements)

## Governing Strategy Tree  

### Motivation  
In this game, Classical planning such as search algorithms and PDDL do not perform well because Pacman games have very many states, and Classical planning usually cannot take into account the full range of situations.
At this point, Approximate Q-learning provides us with a robust framework for choosing the best actions in complex environments, by adjusting the size of the rewards and the weights of the features. Approximate Q-learning can take more situations into account. At the same time, Approximate Q-learning can handle larger state and action spaces within a limited state space than traditional Q-learning.


[Back to top](#table-of-contents)

### Application  
First, let us look at the formula for the approximation q.

$$
Q(s,a) = \sum_{i} f_{i}(s,a) \times w_{i}
$$

As the formula shows above, where $f_i(s, a)$ is the value of the $i^{th}$ feature for state $s$ and action $a$, and $w_i$ is the weight for that feature.

To implement approximate q-learning in code, we will use the following steps

#### 1. initialize the parameter:

```python
class ApproxQLearning(CaptureAgent): 
    def registerInitialState(self, gameState): 
        self.epsilon=0.1
        self.alpha=0.2 
        self.discount=0.9
        self.weights = {'closest food': -6.8749077328451, 'bias': -22.70846720199332, 'distance to ghost': 13.28476402592656, 'distance to pacman': -25.2408978796536, 'eats food': 30.745567010663386, 'homedist': - 15}
        self.start=gameState.getAgentPosition(self.index)
```
In the code above, self.epsilon=0.1 means there are 10% chance that agent might move randomly. self.discount represents $times%, it represents to what extent the approximate q-learning will consider future rewards instead of instant rewards. self.alpha represents the learning rate.

#### 2. Compute Qvalue and return an action:
In the function choose action, the function determineBestPolicy will be used to get legal action.
```python
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
```
By computing the q-value of each action, the action with the highest q-value will be returned.
```python
    def computeQValue(self, gameState, action): 
        next_state = gameState.generateSuccessor(self.index, action)
        features = self.featureSelection.getFeatures(gameState, next_state)
        return features * self.weights
```

#### 3. Training and adjusting weights.

The above situation demonstrates the procedure of approximate q-learning choose action, Next, I will demonstrate the training process.
```python
        if TRAINING:
            if util.flipCoin(self.epsilon):
                selectedAction = random.choice(legalActions)
                self.epsilon = self.epsilon*0.95
            else:
                selectedAction = self.determineBestPolicy(gameState)
                self.adjustWeights(gameState, selectedAction)
```
The above code shows the training logic, if it's in training, we have a certain chance to randomly select the action, at the same time, need to make some adjustments to the weights.
The code below shows how to adjust weights.

```python
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
```
The modify function can be explained by following the formula

$$
\delta = (r + \gamma \times \max_{a'} Q(s', a') - Q(s, a))
$$

where weights equal:

$$
w_i \leftarrow w_i + \alpha \times \delta \times f_i(s, a)
$$

Regarding the REWARD function, the details can be viewed in the code due to its length, which mainly encourages the AGENT to eat food and eat enemies as well as evade them. This also apply to featureExtract function.



[Back to top](#table-of-contents)

### Trade-offs  
#### *Advantages*  

Adaptable: By constantly learning and updating weights, the agent can adapt to various game scenarios.
Handles large state spaces: Approximate Q learning allows us to handle a large number of states and actions without the need to store a Q value for each state-action pair.

#### *Disadvantages*

Weight/feature selection: Choosing the right weights/features can be a challenge. Inappropriate weights may lead the agent to make poor decisions.
Here are three examples to help understand the problem:
1. feature selection: If we use score as one of the weights, approximate q-learning may not be able to understand under what circumstances the score will be increased, which may cause the
2. weight adjustment: If one weight is too high/low, it is easy to over-emphasize/neglect other weights.
3. hard to balance: When features getting more and more, there may be conflicts between features. For instance: the two features "eat food" and "escape from enemy" are easily triggered at the same time, which makes the agent choose the local optimal solution: for example, when being chased, the agent can easily enter a dead-end street to eat food, because it is far away from the enemy and can eat food at the same time. when being chased.

Training required: The agent needs sufficient training time to optimize its weights.



[Back to top](#table-of-contents)

### Future improvements  
Feature selection: We can consider introducing more features to optimize the agent's behavior.
Deep Learning Integration: By integrating deep learning techniques, such as deep Q-networks, we can further optimize the performance of the agent.
Exploration/exploitation balance: By adjusting the epsilon value, we can better balance exploration and exploitation so that the agent learns more efficiently.




[Back to top](#table-of-contents)
