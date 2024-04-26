import pandas as pd
import numpy as np
import copy
import itertools
import time
from inference_pipeline.reinforcement_learning import state as State

class ReinforcementLearning:
    def __init__(self):
        self.hyperparameters = None # hyperparameters for the model
        # self.state = State() # current state of the agent
        self.function = None # function learned from training data
        self.engine = None # engine for agent
        self.crash_algorithm = None # crash algorithm for agent
        self.start_state = None # start state for agent
        self.goal_state = None # goal state for agent
        self.forbidden_state = None # forbidden state for agent
        self.actions = None # permissable actions
        self.costs = None # costs for each state
        self.transition = None # transition probabilities
        self.velocity_limit = None # velocity limits
        self.training_iterations = 100 # number of training iterations
        self.reward = None # reward for goal state
        self.alpha = None # learning rate
        self.gamma = None # discount factor
        self.world = None # world for agent
        self.learning_metrics = [] # logs of model metric as function of training iterations

    """
    'set_params' method is responsible for setting the hyperparameters for the model.
    Args:
        hyperparameters (dictionary): hyperparameters for the model
    Returns:
        None
    """
    def set_params(self, hyperparameters):
        self.hyperparameters = hyperparameters

    def mapping(self, world):
        x = list(range(len(world[0])))
        y = list(range(len(world)))
        x_vel = list(range(self.velocity_limit[0], self.velocity_limit[1] + 1))
        y_vel = list(range(self.velocity_limit[0], self.velocity_limit[1] + 1))
        states = list(itertools.product(x, y, x_vel, y_vel)) # create all combinations of x, y, x_vel, y_vel
        # states = self.convert_tuples_2_objects(states) # convert list of tuples to list of objects
        S = {} # key: state index, value: terrain
        self.state_to_index = {}
        self.index_to_state = {}
        for i, state in enumerate(states):
            self.state_to_index[state] = i # mapping from state to index
            self.index_to_state[i] = state  # mapping from index to state
            # S[i] = world[state.y][state.x] # get terrain of state
            S[i] = world[state[1]][state[0]] # get terrain of state
        self.S = S
        return S

    def inside_boundary(self, state):
        x, y, x_vel, y_vel = state
        if x < 0 or x >= len(self.world[0]) or y < 0 or y >= len(self.world):
            return False
        return True

    def apply_kinematics(self, current_state, action):
        x, y, x_vel, y_vel = current_state
        x_accel, y_accel = action
        lower_limit, upper_limit = self.velocity_limit
        # [1] update velocity 
        new_x_vel = x_vel + x_accel # new velocity_x = current velocity_x + acceleration_x
        new_y_vel = y_vel + y_accel # new velocity_y = current velocity_y + acceleration_y
        # [2] cap new velocity at velocity limit 
        new_x_vel =  max(lower_limit, min(new_x_vel, upper_limit))
        new_y_vel =  max(lower_limit, min(new_y_vel, upper_limit))
        # [3] update position
        new_x = x + new_x_vel # new position_x = current position_x + new velocity_x
        new_y = y + new_y_vel # new position_y = current position_y + new velocity_y
        return (new_x, new_y, new_x_vel, new_y_vel)

    def initialize_rewards(self, R, S, num_rows, num_cols):
        for s in range(len(R)): # s is a state index in S
            current_state = self.index_to_state[s] # convert state index to state (x, y, vx, vy)
            terrain = S[s]  # get terrain of the current state
            # set reward if current state is the goal regardless of action
            if terrain == self.goal_state:
                R[s]= [self.reward for action in R[s]]
            else:
                for a in range(len(R[0])): # a is an action index
                    action = self.actions[a] # action is a tuple (ddx, ddy)
                    new_state = self.apply_kinematics(current_state, action)
                    if not self.inside_boundary(new_state):
                        # R[s][a] = None
                        R[s][a] = -9999
                        continue # skip if new state is outside the world
                    new_state_index = self.state_to_index[new_state]
                    terrain = S[new_state_index] # get terrain of the new state
                    # if terrain == self.forbidden_state: # walls are forbidden 
                    #     R[s][a] = None ## FIX LATER
                    #     continue
                    R[s][a] = self.costs[terrain] # set cost to enter new state terrain
        return R

    # def reorder_states(self, S): # recursive 
    #     new_S = {}
    #     goal_states = [index for index, terrain in self.S.items() if terrain == self.goal_state]
    #     pass
    #     return S

    def initialize_vars(self, world):
        num_rows = len(world)
        num_cols = len(world[0])
        num_x_vel = len(range(self.velocity_limit[0], self.velocity_limit[1] + 1))
        num_y_vel = len(range(self.velocity_limit[0], self.velocity_limit[1] + 1))
        num_states = num_rows * num_cols * num_x_vel * num_y_vel
        num_actions = len(self.actions)
        S = self.mapping(world) # states
        V = [-9999 for _ in range(num_states)] # value function
        Vlast = [-99999999 for _ in range(num_states)] # copy of value function
        R = [[0]*num_actions for _ in range(num_states)] # reward function
        R = self.initialize_rewards(R, S, num_rows, num_cols)
        Q = [[-9999]*num_actions for _ in range(num_states)] # Q function
        P = [None for _ in range(num_states)] # policy

        # # remove invalid states from S
        # valid_S = copy.deepcopy(S)
        # for s in S.keys():
        #     is_valid = self.valid(s)
        #     if not is_valid:
        #         valid_S.pop(s)
        # S = valid_S

        return (S, V, Vlast, R, Q, P, num_rows, num_cols)
    
    def keep_updating(self, V, Vlast, epsilon):
        array1 = np.array(V)
        array2 = np.array(Vlast)
        self.learning_metrics.append(np.linalg.norm(array1 - array2)) # log the difference between V and Vlast
        return np.any(np.abs(array1 - array2) > epsilon)

    def successors(self, current_state, desired_action, S, num_rows, num_cols):
        children = []
        # [1] apply desired action
        probability = self.transition['success'] 
        new_state = self.apply_kinematics(current_state, desired_action)
        if self.inside_boundary(new_state):
            new_state_index = self.state_to_index[new_state]
            terrain = S[new_state_index] # get terrain of new state
            # if the action I take makes me bump into a wall, then reset the state to the current state and reset velocity to 0
            if terrain == self.forbidden_state: # apply crash algorithm
                new_state = (current_state[0], current_state[1], 0, 0)
                new_state_index = self.state_to_index[new_state]
            children.append((new_state_index, probability))
         # [2] apply undesired action
        probability = self.transition['fail'] 
        undesired_action = (0, 0) # no acceleration happens
        new_state = self.apply_kinematics(current_state, undesired_action)
        if self.inside_boundary(new_state):
            new_state_index = self.state_to_index[new_state]
            terrain = S[new_state_index] # get terrain of new state
            # if the action I take makes me bump into a wall, then reset the state to the current state and reset velocity to 0
            if terrain == self.forbidden_state: # apply crash algorithm
                new_state = (current_state[0], current_state[1], 0, 0)
                new_state_index = self.state_to_index[new_state]
            children.append((new_state_index, probability))
        return children
    
    def stochastic_future_reward(self, s, a, actions, S, Vlast, num_rows, num_cols):
        current_state = self.index_to_state[s] # convert state index to (x, y, vx, vy)
        desired_action = actions[a] # action is a tuple (ddx, ddy)
        new_states = self.successors(current_state, desired_action, S, num_rows, num_cols)
        future_reward = 0
        for new_state in new_states:
            future_reward += new_state[1] * Vlast[new_state[0]] # probability * future reward
        return future_reward
    
    def get_best_action(self, Q, s):
        lst = Q[s]
        # if there are no actions to take in state s, then return None
        # if all([x is None for x in lst]):
        #     return None
        # if there are actions to take in state s, then return the index of the best action
        lst = [x if x is not None else -9999 for x in lst] # replace None with -999 to use max function
        action_index = lst.index(max(lst)) # index of best action in Q[s]
        return action_index
    
    def valid(self, s):
        # check if state is on border of boundary 
        state = self.index_to_state[s] # convert state index to (x, y, vx, vy)
        x, y = state[0:2]
        on_border = False
        # [1] check if state is on border of boundary
        if x == 0 or x == len(self.world[0]) - 1 or y == 0 or y == len(self.world) - 1:
            on_border = True
        # [2] check if velocity leads to a state outside the boundary
        if on_border:
            x_vel, y_vel = state[2:]
            if x == 0 and x_vel <= -2:
                return False
            if x == len(self.world[0]) - 1 and x_vel >= 2:
                return False
            if y == 0 and y_vel <= -2:
                return False
            if y == len(self.world) - 1 and y_vel >= 2:
                return False
        return True

    def value_iteration(self, world):
        S, V, Vlast, R, Q, P, num_rows, num_cols = self.initialize_vars(world)
        # epsilon = 1e-3
        # epsilon = 0.1
        epsilon = 0.1
        t = 1
        while (self.keep_updating(V, Vlast, epsilon) and t < self.training_iterations):
            start_time = time.time()
            Vlast = copy.deepcopy(V)  
            for s, terrain in S.items(): # key: state_index, value: terrain
                # if terrain == self.forbidden_state:
                #     continue # do nothing if you are on a forbidden state since you cant be in one in first place
                for a, _ in enumerate(self.actions): # a is action index
                    # if R[s][a] is None:
                    #     Q[s][a] = None
                    #     continue
                    Q[s][a] = R[s][a] + self.gamma * self.stochastic_future_reward(s, a, self.actions, S, Vlast, num_rows, num_cols) 
                P[s] = self.get_best_action(Q, s) # return index of best action to take in state s
                # if P[s] is None:
                #     continue
                V[s] = Q[s][P[s]] # return value of best action to take in state s
            end_time = time.time()
            print(f'        Iteration {t}, delta V: {self.learning_metrics[-1]}, {end_time - start_time:.2f} s')
            t += 1
        self.training_iterations = t
        print('Policy learning complete...')
        return P


    """
    'fit' method is responsible for training the model using the training data. The method first checks if
    the model requires initial training of an autoencoder. If it does, the method trains the autoencoder. The 
    method then creates the feedforward neural network by dropping the output layer of the autoencoder and 
    clipping the remaining layers to the first hidden layer of the feedforward network. 
    
    This has the effect of placing the feedforward network in a more appropriate weights and biases space where the initial shallow
    layers have already learned valuable features so the weights in those shallow layers dont need alot of 
    updating during gradient descent. This helps compat the side-effects of the vanishing gradient phenomena
    becuase we focus our updates to those weights in the deep layers. Weights in the shallow layers only
    undergo fine-tuning. The method then trains the feedforward neural network using batch gradient descent. 
    The method returns logs of metrics as function of epochs. 
    Args:
        X (DataFrame): training data
        y (DataFrame): target variable
    Returns:
        learning_metrics (dictionary): logs of model metric as function of epochs
    """
    def fit(self, world):
        print(f'        Training RL agent using {self.engine}...')
        self.world = world
        # engine = getattr(ReinforcementLearning(), self.engine) # reference engine function to run RL algorithm
        if self.engine == 'value_iteration':
            policy = self.value_iteration(world)
        elif self.engine == 'q_learning':
            policy = self.q_learning(world) # learn optimal policy
            pass
        elif self.engine == 'sarsa':
            # policy = engine(world) # learn optimal policy
            pass
        self.function = policy
        # return self.learning_metrics
        return None
    
    def initialize_agent(self):
        # get index of all start states in world 
        start_state_indices = [index for index, terrain in self.S.items() if terrain == self.start_state]
        # randomly select a start state
        start_state_index = np.random.choice(start_state_indices)
        return start_state_index
    
    def create_path(self, start_state_index):
        policy = self.function
        path = []
        current_state_index = start_state_index
        current_state = self.index_to_state[current_state_index]
        action_index = policy[current_state_index]
        if action_index is None:
            print('         Policy contains a None action. Failed to reach goal state.')
            return path
        action = self.actions[action_index]
        new_state = self.apply_kinematics(current_state, action)
        if self.inside_boundary(new_state):
            new_state_index = self.state_to_index[new_state]
        else:
            print('         Policy contains loops. Failed to reach goal state.')
            return path
        path.append((current_state, action, new_state))
        terrain = self.S[new_state_index]
        while terrain != self.goal_state:
            current_state_index = new_state_index
            current_state = self.index_to_state[current_state_index]
            action_index = policy[current_state_index]
            if action_index is None:
                print('         Policy contains a None action. Failed to reach goal state.')
                return path
            action = self.actions[action_index]
            new_state = self.apply_kinematics(current_state, action)
            if self.inside_boundary(new_state):
                new_state_index = self.state_to_index[new_state]
            else:
                print('         Policy contains loops. Failed to reach goal state.')
                return path
            path.append((current_state, action, new_state))
            terrain = self.S[new_state_index]
        return path

    """
    'predict' method is responsible for making predictions using the trained model. The method first
    calculates the activations of the neurons in all layers of the neural network using the forward
    propagation method. The method then returns the predicted values. For classification, the method
    returns the class with the highest probability. For regression, the method returns the predicted
    values.
    Args:
        X (DataFrame): input values
    Returns:
        y_pred (DataFrame): predicted values
    """
    def predict(self):
        start_state_index = self.initialize_agent() # place agent at a start state
        path = self.create_path(start_state_index) # create path from optimal policy
        path_metrics = {'length': len(path), 'cost': 0, 'training_iters': self.training_iterations} # calculate path metrics
        return path, path_metrics