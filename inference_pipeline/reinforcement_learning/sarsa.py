import numpy as np

"""
This module contains the SARSA class which is used to learn the optimal policy for an agent in a Markov Decision Process (MDP) using the SARSA algorithm.
The SARSA algorithm is a type of reinforcement learning algorithm that is model-free and on-policy. It is model free because it does not know the full MDP
model, like the transition function T but does not attempt to approximate the model. Instead, it aims to directly search for the optimal policy. It is also
considered on-policy because it includes an exploration step and we update the Q-function with the action taken by the agent in new state s'. SARSA is a powerful
method to learn an optimal policy and has been shown to converge under certain conditions. The algorithm mainly relies on a Q function for each state-action pair.
The Q function represents the expected cumulative reward that an agent can achieve from a given state by taking a given action and following the optimal policy
thereafter. The algorithm works by selecting an action using an epsilon-greedy policy, applying the action to the current state, observing the reward and next 
state, taking the next action using an epsilon-greedy policy, and updating the Q function using the Bellman equation. The Bellman equation allows us to update 
the Q value of a state-action pair by considering the current reward we collect in that state, the discounted future rewards we can collect from next state, and the Q value we can achieve from the next state. The algorithm then selects
the best action to take in each state by selecting the action that maximizes the Q function. The algorithm then updates the Q function repeats the process until 
a certain number of episodes are run. After the algorithm runs for a specific number of episodes, we can extract the optimal policy from the Q function
by selecting the action that maximizes the Q value for each state.

The SARSA class contains the following attributes:
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
    - learning_metrics: the learning metrics for the agent

The SARSA class contains the following methods:
    - learn: learn the optimal policy using the SARSA algorithm
"""

class SARSA():
    def __init__(self, env):
        self.RL = env
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
        self.learning_metrics = None # learning metrics
    
    """
    'learn' method is responsible for learning the optimal policy using the SARSA algorithm. Learning consists of running episodes of the agent interacting 
    with the environment, selecting actions using an epsilon-greedy policy, updating the Q function using the Bellman equation, and extracting the optimal policy.
    The method returns the optimal policy and learning metrics. Within each episode, we initialize the state randomly that is inversely proportional to the 
    frequency of visiting that state in the past and select an action using an epsilon-greedy policy. We then apply the action to the current state, observe the 
    reward and next state, take the next action using an epsilon-greedy policy, and update the Q function using the Bellman equation and the new action. If an 
    episode sequence length exceeds a certain threshold, we penalize the agent for being stuck and break the loop.
    Args:
        S (list): list of states
        R (list): list of rewards
        Q (list): Q function
        visits (list): visit count for each state-action pair
    Returns:
        tuple: optimal policy and learning metrics
    """
    def learn(self, S, R, Q, visits):
        Q_history = {'Q_forbidden_max': [], 'Q_track_max': [], 'Q_forbidden_mean': [], 'Q_track_mean': []}
        visit_history = []
        for t in range(self.training_iterations + 1):
            t_inner = 0
            escape = False
            s, horizon = self.RL.choose_start_state(visits, t) # initialize state randomly close to terminal state
            a, greedy_epsilon, flag = self.RL.epsilon_greedy_choice(Q, s, visits, t) # choose action using epsilon-greedy policy
            while S[s] != self.goal_state:
                t_inner += 1
                action = self.actions[a] # action is a tuple (ddx, ddy)
                state = self.index_to_state[s] # convert state index to (x, y, vx, vy)
                random_num = np.random.uniform(0, 1)
                if random_num < self.transition['fail']:
                    action = (0, 0) # no acceleration happens
                new_state = self.RL.apply_kinematics(state, action) # apply action to get new state
                new_state = self.RL.handle_collision(state, new_state) # handle collision with walls
                new_s = self.state_to_index[new_state] # convert new state to state index
                new_a, greedy_epsilon, flag = self.RL.epsilon_greedy_choice(Q, new_s, visits, t) # choose action using epsilon-greedy policy
                alpha = self.RL.learning_rate(t) # learning rate anneahling
                # penalize agent for being stuck
                if t_inner >= 200:
                    reward = -999
                    escape = True
                else:
                    reward = R[s][a]
                Q[s][a] += alpha * (reward + (self.gamma * Q[new_s][new_a]) - Q[s][a]) # update Q function 
                visits[s][a] += 1 # update visit count
                s = new_s # update state
                a = new_a # update action
                if escape is True:
                    break
            Q_history, visit_history, not_visited, Q_forbidden_mean, Q_track_mean = self.RL.print_stats(Q, Q_history, visits, visit_history)
            print(f'        Outer Iteration {t}, horizon: {horizon:.3f}, greedy-epsilon: {greedy_epsilon:.3f}, learning rate: {alpha:.3f}, not_visited: {not_visited:.3f}%, Q_track_mean: {Q_track_mean:.3f}, inner iterations: {t_inner:0{4}d}')
        # compile learning metrics and extract policy
        learning_metrics = (Q_history, visit_history, visits, alpha, greedy_epsilon, not_visited)
        P = self.RL.extract_policy(Q) # extract policy from Q function
        self.RL.export_Q(Q) # export Q for debugging
        return P, learning_metrics
