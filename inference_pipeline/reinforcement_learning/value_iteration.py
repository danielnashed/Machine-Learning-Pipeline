import numpy as np
import copy

class ValueIteration():
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
        self.epsilon = env.convergence_epsilon # epsilon for convergence
        self.learning_metrics = []

    def keep_updating(self, V, Vlast):
        array1 = np.array(V)
        array2 = np.array(Vlast)
        self.learning_metrics.append(np.linalg.norm(array1 - array2)) # log the difference between V and Vlast
        return np.any(np.abs(array1 - array2) > self.epsilon)


    def successors(self, current_state, desired_action):
        children = []
        # [1] apply desired action
        probability = self.transition['success'] 
        new_state = self.RL.apply_kinematics(current_state, desired_action)
        new_state = self.RL.handle_collision(current_state, new_state)
        new_state_index = self.state_to_index[new_state]
        children.append((new_state_index, probability))
         # [2] apply undesired action
        probability = self.transition['fail'] 
        undesired_action = (0, 0) # no acceleration happens
        new_state = self.RL.apply_kinematics(current_state, undesired_action)
        new_state = self.RL.handle_collision(current_state, new_state)
        new_state_index = self.state_to_index[new_state]
        children.append((new_state_index, probability))
        return children
    
    def stochastic_future_reward(self, s, a, Vlast):
        current_state = self.index_to_state[s] # convert state index to (x, y, vx, vy)
        desired_action = self.actions[a] # action is a tuple (ddx, ddy)
        new_states = self.successors(current_state, desired_action)
        future_reward = 0
        for new_state in new_states:
            future_reward += new_state[1] * Vlast[new_state[0]] # probability * future reward
        return future_reward
    
    def learn(self, S, V, Vlast, R, Q, P):
        t = 1
        while (self.keep_updating(V, Vlast) and t < self.training_iterations):
            Vlast = copy.deepcopy(V)  
            for s in S.keys(): # key: state_index
                for a, _ in enumerate(self.actions): # a is action index
                    Q[s][a] = R[s][a] + self.gamma * self.stochastic_future_reward(s, a, Vlast) 
                P[s] = self.RL.get_best_action(Q, s) # return index of best action to take in state s
                V[s] = Q[s][P[s]] # return value of best action to take in state s
            print(f'        Iteration {t}, delta V: {round(self.learning_metrics[-1], 5)}')
            t += 1
        print(f'        Iteration {t}, delta V: {round(self.learning_metrics[-1], 5)}')
        return P, self.learning_metrics
