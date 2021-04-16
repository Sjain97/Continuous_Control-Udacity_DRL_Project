# Report

## Implementation
The environment was solved using a deep reinforcement learning agent. The implementation can be found in the `Continuous_Control.ipynb` Jupyter Notebook.
`ddpg_agent.py` contains the rl-agent and `model.py` contains the neural networks used as the estimators. 

### Learning algorithm
[DDPG](https://arxiv.org/abs/1509.02971) which is an actor-critic approach was used as the learning algorithm for the agent.
This algorithm is quite similar to DQN, but also manages to solve tasks with continuous action spaces. As an off-policy algorithm
DDPG utilizes four neural networks: a local actor, a target actor, a local critic and a target critic
Each training step the experience (state, action, reward, next state) the 20 agents gained was stored.
Then every second training step the agent learned from a random sample from the stored experience. The actor tries to estimate the
optimal policy by using the estimated state-action values from the critic while critic tries to estimate the optimal Q-value function
and learns by using a normal Q-learning approach. Using this approach one gains the benefits of value based and policy based
methods at the same time.

### Hyperparameters
The following hyperparameters were used:
```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

### Neural network Architecture
The actor model is a simple feedforward network:
* Batch normalization
* Input layer: 33 (input) neurons (the state size)
* 1st hidden layer: 128 neurons (leaky relu)
* 2nd hidden layer: 128 neurons (leaky relu)
* output layer: 4 neurons (1 for each action) (tanh)

The critic model:
* Batch normalization
* Input layer: 33 (input) neurons (the state size)
* 1st hidden layer: 128 neurons (leaky relu)
* 2nd hidden layer: 128 neurons (leaky relu)
* output layer: 1 neuron

## Results
The agent was able to solve the environment after 164 episodes achieving an average score of 30.08 over the last 100 episodes
of the training.
<br>

<img src="/assets/training_log.jpg" width="60%" align="top-left" alt="" title="Training Logs" />
<br>

The average scores of the 20 agents during the training process: <br>
<img src="/assets/Scores_over_last_100_episodes.jpg" width="40%" align="top-left" alt="" title="Scores over last 100 episodes" />
<br>

<img src="/assets/Scores_over_all_164_episodes.jpg" width="40%" align="top-left" alt="" title="Scores over all 164 episodes" />

## Possible Future Improvements
The algorithm could be improved in many ways, such as:

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
  - The idea is to implement a Policy Gradient algorithm that determines the appropriate policy with gradient methods. However, the change in the policy from one iteration to another is very slow in the neighbourhood of the previous policy in the high dimensional space.
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
  - The idea behind using these technique for sampling from the replay buffer is that not all experiences are equal, some are more important than others in terms of reward, so naturally the agent should at least prioritize between the different experiences.
- [Asynchronous Actor Critic](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
  - The idea is to have a global network and multiple agents who all interact with the environment separately and send their gradients to the global network for optimization in an asynchronous way.
