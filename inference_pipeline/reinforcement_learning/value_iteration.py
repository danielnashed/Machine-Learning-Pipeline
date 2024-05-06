import numpy as np
import copy

"""
This module contains the ValueIteration class which is used to learn the optimal policy for an agent in a Markov Decision Process (MDP) using the value 
iteration algorithm. The value iteration algorithm is a type of dynamic programming algorithm used to learn an optimal policy for an agent by 
iteratively updating the value function of each state. It is considered model-based since it relies directly on the transition function to generate all
possible successor states we can transition to from a current state. The value function represents the expected cumulative reward that an agent can achieve 
from a given  state by following the optimal policy. The algorithm works by iterating over all possible state-action pairs and updating the value function 
using the Bellman equation. The Bellman equation allows us to update the value of a state by considering the current reward we collect in that state and the 
discounted future rewards we can collect from next state. The algorithm then selects the best action to take in each state by selecting the action that maximizes 
the value function. The algorithm then updates the value function using the value of the best action and repeats the process until convergence. For each state, we 
store the best action to take in that state in our policy so we can return the optimal policy.

The ValueIteration class contains the following attributes:
    - RL: the Reinforcement Learning (RL) environment
    - engine: the engine for the agent
    - crash_algorithm: the crash algorithm for the agent
    - start_state: the start state for the agent
    - goal_state: the goal state for the agent
    - forbidden_state: the forbidden state for the agent
    - actions: the permissable actions for the agent
    - costs: the costs for each state
    - transition: the transition probabilities for the agent
    - velocity_limit: the velocity limits for the agent
    - training_iterations: the number of training iterations for the agent
    - reward: the reward for the goal state
    - alpha: the learning rate for the agent
    - gamma: the discount factor for the agent
    - world: the world for the agent
    - transfer_learning: the directory for transfer learning
    - output: the path to save functions to
    - epsilon: the epsilon for convergence
    - learning_metrics: the learning metrics for the agent
    
The ValueIteration class contains the following methods:
    - keep_updating: check if the value function has converged
    - successors: generate all possible next states from the current state
    - stochastic_future_reward: calculate the future reward for a given state and action
    - learn: learn the optimal policy using the value iteration algorithm
"""

class ValueIteration():
    def __init__(self, env):
        self.RL = env # inherit parent RL class to use its utility functions and attributes
        self.engine = env.engine # engine for agent
        self.crash_algorithm = env.crash_algorithm # crash algorithm for agent
        self.start_state = env.start_state # start state for agent
        self.goal_state = env.goal_state # goal state for agent
        self.forbidden_state = env.forbidden_state # forbidden state for agent
        self.actions = env.actions # permissable actions
        self.costs = env.costs # costs for each state
        self.transition = env.transition # transition probabilities
        self.velocity_limit = env.velocity_limit # velocity limits
        self.training_iterations = env.training_iterations # number of training iterations
        self.reward = env.reward # reward for goal state
        self.alpha = env.alpha # learning rate
        self.gamma = env.gamma # discount factor
        self.world = env.world # world for agent
        self.transfer_learning = env.transfer_learning # directory for transfer learning
        self.output = env.output # path to save functions to 
        self.epsilon = env.convergence_epsilon # epsilon for convergence
        self.learning_metrics = []

    """
    'keep_updating' method is responsible for checking if the value function has converged or not. It does so by comparing the value function from the current
    iteration with the value function from the last iteration. If the difference between any two corresponding states in the two value functions is less 
    than epsilon, the value function has converged and the method returns False so we stop iterating. Otherwise, it returns True to keep iterating.
    Args:
        V (list): value function
        Vlast (list): value function from last iteration
    Returns:
        bool: True if value function has not converged, False otherwise
    """
    def keep_updating(self, V, Vlast):
        array1 = np.array(V)
        array2 = np.array(Vlast)
        self.learning_metrics.append(np.max(np.absolute(array1 - array2))) # log the difference between V and Vlast
        return np.any(np.abs(array1 - array2) > self.epsilon) # True if any of the elements in the array is greater than epsilon, False otherwise

    """
    'successors' method is responsible for generating all possible next states from the current state. It does so by applying the desired action 
    and the undesired action in current state using the kinematics model and collision handling function.
    Args:
        current_state (tuple): current state of the agent
        desired_action (tuple): desired action to be taken by the agent
    Returns:
        children (list): list of tuples containing next state and probability of reaching that state
    """
    def successors(self, current_state, desired_action):
        penalty = 1.5 # penalty for collision # 1.5
        children = []
        # [1] apply desired action
        probability = self.transition['success'] 
        new_state = self.RL.apply_kinematics(current_state, desired_action)
        new_state_after_collision = self.RL.handle_collision(current_state, new_state)
        if new_state_after_collision == new_state: # no collision happened
            new_state_index = self.state_to_index[new_state]
            children.append((new_state_index, probability))
        else:
            new_state_index = self.state_to_index[new_state_after_collision]
            children.append((new_state_index, probability*penalty)) # reduce reward by scaling down probability
         # [2] apply undesired action
        probability = self.transition['fail'] 
        undesired_action = (0, 0) # no acceleration happens
        new_state = self.RL.apply_kinematics(current_state, undesired_action)
        new_state_after_collision = self.RL.handle_collision(current_state, new_state)
        if new_state_after_collision == new_state: # no collision happened
            new_state_index = self.state_to_index[new_state]
            children.append((new_state_index, probability))
        else:
            new_state_index = self.state_to_index[new_state_after_collision]
            children.append((new_state_index, probability*penalty)) # reduce reward by scaling down probability
        return children
    
    """
    'stochastic_future_reward' method is responsible for calculating the future reward for a given state, action to be taken in that state and value function.
    Since the MDP is stochastic, the future reward is calculated by taking the weighted sum of the future rewards of all possible next states. The weights are
    simply the probabilities of reaching those states given by the transition model.
    Args:
        s (int): current state index
        a (int): action index
        Vlast (list): value function from last iteration
    Returns:
        future_reward (float): future reward for the given state and action
    """
    def stochastic_future_reward(self, s, a, Vlast):
        current_state = self.index_to_state[s] # convert state index to (x, y, vx, vy)
        desired_action = self.actions[a] # action is a tuple (ddx, ddy)
        new_states = self.successors(current_state, desired_action)
        future_reward = 0
        for new_state in new_states:
            future_reward += new_state[1] * Vlast[new_state[0]] # probability * future reward
        return future_reward
    
    """
    'learn' method is responsible for learning the optimal policy using value iteration algorithm. It does so by iterating over all states and actions 
    and updating the value function using the Bellman equation. The iteration continues until the value function converges or the maximum number of iterations 
    is reached. The algorithm works by looping over all possible state-action pairs and updating the auxiliary Q function values using the Bellman equation
    which is a type of dynamic programming algorithm. It allows us to update the value of a state by considering the current reward we collect in that state 
    and the discounted future rewards we can collect from the next state. The algorithm then selects the best action to take in each state by selecting the
    action that maximizes the Q value. The algorithm then updates the value function using the Q values and repeats the process until convergence. For each 
    state, we store the best action to take in that state in our policy so we can return the optimal policy.
    Args:
        S (dict): dictionary containing state index as key and state as value
        V (list): value function
        Vlast (list): value function from last iteration
        R (dict): dictionary containing state index as key and rewards for each action as value
        Q (dict): dictionary containing state index as key and Q values for each action as value
        P (dict): dictionary containing state index as key and best action to take in that state as value
    Returns:
        P (dict): dictionary containing state index as key and best action to take in that state as value
        learning_metrics (list): list containing learning metrics
    """
    def learn(self, S, V, Vlast, R, Q, P):
        t = 1
        while (self.keep_updating(V, Vlast) and t < self.training_iterations):
            Vlast = copy.deepcopy(V)  
            for s in S.keys(): # key: state_index
                print(f'{s}/{len(S)}')
                for a, _ in enumerate(self.actions): # a is action index
                    Q[s][a] = R[s][a] + self.gamma * self.stochastic_future_reward(s, a, Vlast) 
                P[s] = self.RL.get_best_action(Q, s) # return index of best action to take in state s
                V[s] = Q[s][P[s]] # return value of best action to take in state s
            print(f'        Iteration {t}, delta V: {round(self.learning_metrics[-1], 5)}')
            t += 1
        print(f'        Iteration {t}, delta V: {round(self.learning_metrics[-1], 5)}')
        return P, self.learning_metrics

