import pandas as pd
import numpy as np
import copy
import itertools
import time
import random
import csv
import os
from inference_pipeline.reinforcement_learning import value_iteration as __value_iteration__
from inference_pipeline.reinforcement_learning import bresenham_line as BresenhamLineAlgorithm

class ReinforcementLearning:
    def __init__(self):
        self.hyperparameters = None # hyperparameters for the model
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
        self.convergence_epsilon = None # epsilon for convergence for value iteration
        self.world = None # world for agent
        self.visit_history = None # history of visits to states
        self.transfer_learning = None # directory for transfer learning
        self.output = None # path to save functions to 
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
        S = {} # key: state index, value: terrain
        self.state_to_index = {}
        self.index_to_state = {}
        for i, state in enumerate(states):
            self.state_to_index[state] = i # mapping from state to index
            self.index_to_state[i] = state  # mapping from index to state
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

    def initialize_rewards(self, R, S):
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
                        R[s][a] = -9999
                        continue # skip if new state is outside the world
                    new_state_index = self.state_to_index[new_state]
                    terrain = S[new_state_index] # get terrain of the new state
                    # if terrain == self.forbidden_state: # walls are forbidden 
                    #     R[s][a] = None ## FIX LATER
                    #     continue
                    R[s][a] = self.costs[terrain] # set cost to enter new state terrain
        return R

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
        R = self.initialize_rewards(R, S)
        if self.transfer_learning is None:
            Q = [[-9]*num_actions for _ in range(num_states)] # Q function # -9999
        else:
            Q = self.initialize_via_transfer() # initialize Q function using transfer learning 
        for s in range(len(Q)):
            if S[s] == self.goal_state:
                Q[s] = [1000]*num_actions
            elif S[s] == self.forbidden_state:
                Q[s] = [-9999]*num_actions
        P = [None for _ in range(num_states)] # policy
        visits = np.zeros((len(Q), len(Q[0]))) # visit history
        return (S, V, Vlast, R, Q, P, visits)
    
    def initialize_via_transfer(self):
        with open(self.transfer_learning, 'r') as f:
            reader = csv.reader(f)
            Q = list(reader)
        return Q
    
    # def keep_updating(self, V, Vlast, epsilon):
    #     array1 = np.array(V)
    #     array2 = np.array(Vlast)
    #     self.learning_metrics.append(np.linalg.norm(array1 - array2)) # log the difference between V and Vlast
    #     return np.any(np.abs(array1 - array2) > epsilon)

    # def successors(self, current_state, desired_action, S, num_rows, num_cols):
    #     children = []
    #     # [1] apply desired action
    #     probability = self.transition['success'] 
    #     new_state = self.apply_kinematics(current_state, desired_action)
    #     if self.inside_boundary(new_state):
    #         new_state_index = self.state_to_index[new_state]
    #         terrain = S[new_state_index] # get terrain of new state
    #         # if the action I take makes me bump into a wall, then reset the state to the current state and reset velocity to 0
    #         if terrain == self.forbidden_state: # apply crash algorithm
    #             new_state = (current_state[0], current_state[1], 0, 0)
    #             new_state_index = self.state_to_index[new_state]
    #         children.append((new_state_index, probability))
    #      # [2] apply undesired action
    #     probability = self.transition['fail'] 
    #     undesired_action = (0, 0) # no acceleration happens
    #     new_state = self.apply_kinematics(current_state, undesired_action)
    #     if self.inside_boundary(new_state):
    #         new_state_index = self.state_to_index[new_state]
    #         terrain = S[new_state_index] # get terrain of new state
    #         # if the action I take makes me bump into a wall, then reset the state to the current state and reset velocity to 0
    #         if terrain == self.forbidden_state: # apply crash algorithm
    #             new_state = (current_state[0], current_state[1], 0, 0)
    #             new_state_index = self.state_to_index[new_state]
    #         children.append((new_state_index, probability))
    #     return children
    
    # def stochastic_future_reward(self, s, a, actions, S, Vlast, num_rows, num_cols):
    #     current_state = self.index_to_state[s] # convert state index to (x, y, vx, vy)
    #     desired_action = actions[a] # action is a tuple (ddx, ddy)
    #     new_states = self.successors(current_state, desired_action, S, num_rows, num_cols)
    #     future_reward = 0
    #     for new_state in new_states:
    #         future_reward += new_state[1] * Vlast[new_state[0]] # probability * future reward
    #     return future_reward

    def get_best_action(self, Q, s):
        lst = np.array(Q[s])
        # if all actions are of equal value, choose an action randomly 
        if len(np.unique(lst)) == 1:
            return np.random.choice(range(len(lst)))
        action_index = np.argmax(lst) # index of best action in Q[s]
        return action_index
    
    def reorder_states(self, S):
        track_state_indices = np.array([index for index, terrain in S.items() if terrain != self.forbidden_state and terrain != self.goal_state])
        track_states = np.array([self.index_to_state[index][:2] for index in track_state_indices]) # extract states as (x, y)
        goal_state_indices = [index for index, terrain in self.S.items() if terrain == self.goal_state] # extract all goal indices
        goal_states = np.array(list(set([self.index_to_state[index][:2] for index in goal_state_indices]))) # extract states as (x, y)
        goal_states_sorted = goal_states[np.lexsort((goal_states[:, 1], goal_states[:, 0]))] # sort all states by x then by y
        goal_state =  goal_states_sorted[len(goal_states_sorted) // 2] # pick middle goal state 
        track_states_dists = np.linalg.norm(track_states - goal_state, axis=1) # calculate euclidian distance from all track states to goal state
        track_state_indices_sorted = track_state_indices[np.argsort(track_states_dists)] # sort track states by decreasing distance to goal state
        self.track_state_indices = track_state_indices_sorted
        return None

    # # expanding horizon
    # def choose_start_state(self, visits, t):
    #     initial_horizon = 2000 # start with 10 states at start near terminal state #10, 1500
    #     peak_time = 0.8 # time when horizon covers entire space of indices
    #     growth_factor = np.log(len(self.track_state_indices) / initial_horizon) / peak_time # control growth rate
    #     horizon = min(len(self.track_state_indices) ,initial_horizon * np.exp(t * growth_factor / self.training_iterations)) # exponential growth of horizon from 10 states to all track states
    #     indices = self.track_state_indices[:int(horizon)] # expanding horizon of indices
    #     freqs = np.sum(visits[indices], axis=1)
    #     probs = 1 / (1 + (10 * freqs)) # probability inversely proportional to frequency of visit
    #     normalized_probs = probs / sum(probs) # normalize probabilities so all sum up to 1
    #     s = random.choices(list(indices), weights=list(normalized_probs), k=1)[0] # choose a state according to its probability
    #     return s, horizon/len(self.track_state_indices)

    # sliding receptive field: this works better than expanding horizon
    def choose_start_state(self, visits, t):
        peak_time_1 = 0.0 # time when horizon covers entire space of indices, 0.8
        peak_time_2 = 0.0
        receptive_field = 4000 # max number of states to look at for choosing a state
        # limited receptive field that moves from terminal states to start states
        if t < peak_time_1 * self.training_iterations:
            initial_frontier_pos = 2000 # start with 10 states at start near terminal state #10, 1500
            velocity = len(self.track_state_indices) / (peak_time_1 * self.training_iterations) # velocity of horizon frontier
            frontier_pos = initial_frontier_pos + (velocity * t)  # linear shift in horizon frontier 
            frontier_pos = int(min(frontier_pos, len(self.track_state_indices))) # clip horizon to maximum number of states
            tail_pos = int(max(0, frontier_pos - receptive_field))
        elif t < peak_time_2 * self.training_iterations:
            tail_pos = len(self.track_state_indices) - receptive_field
            frontier_pos = len(self.track_state_indices)
        # after peak time, horizon expands to cover all states
        else:
            tail_pos = 0
            frontier_pos = len(self.track_state_indices)
        indices = self.track_state_indices[tail_pos:frontier_pos] # expanding horizon of indices
        freqs = np.sum(visits[indices], axis=1)
        probs = 1 / (1 + (10 * freqs)) # probability inversely proportional to frequency of visit
        normalized_probs = probs / sum(probs) # normalize probabilities so all sum up to 1
        s = random.choices(list(indices), weights=list(normalized_probs), k=1)[0] # choose a state according to its probability
        return s, frontier_pos/len(self.track_state_indices)
    

    def epsilon_greedy_choice(self, Q, s, visits, t):
        # epsilon = 0.25 * (1 / (t + 1)) # epsilon decreases over time (exploration-exploitation tradeoff)
        # epsilon = 1.0 * np.exp(-t / (0.8 * self.training_iterations)) # exponential decay of epsilon, #0.5
        # freq = np.sum(visits[s])
        # epsilon = 0.9 * np.exp(- 0.1 * freq)
        flag = None
        epsilon = 1 - (0.5* t /self.training_iterations) # linear decay from 1.0 to 0.8 up to 80% of iterations #0.2, 0.5, 0.8
        if t > 0.8 * self.training_iterations:
            epsilon = 0.1 # 0.2
        if t == 0:
            self.initial_greedy_epsilon = epsilon
        values = list(Q[s])
        random_num = np.random.uniform(0, 1)
        # exploration choice has probability epsilon of choosing a suboptimal action
        if random_num < epsilon:
            flag = 'explore'
            # when exploring, dont just pick an action randomly, but pick one that has not been visited frequently in the past in current state
            freqs = visits[s]
            probs = 1 / (1 + (10 * freqs)) # probability inversely proportional to frequency of visit
            normalized_probs = probs / sum(probs) # normalize probabilities so all sum up to 1
            a = random.choices(list(range(len(values))), weights=list(normalized_probs), k=1)[0] # choose an action according to its probability
        # exploitation choice (greedy) has probability 1 - epsilon of choosing the best action
        else:
            flag = 'exploit'
            a = self.get_best_action(Q, s)
        # print(flag)
        return a, epsilon, flag
    
    def pretty_grid(self, grid):
        # convert the grid to emojis for visualization
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == 'S':
                    grid[row][col] = '🟥'
                elif grid[row][col] == 'F':
                    grid[row][col] = '🏁'
                elif grid[row][col] == '#':
                    grid[row][col] = '⬛'
                else:
                    grid[row][col] = '⬜'
        return grid
    
    def export_line(self, points, start, end):
        time_stamp = time.time()
        grid = copy.deepcopy(self.world) # copy the world grid
        grid = self.pretty_grid(grid) # convert the grid to emojis for visualization
        for point in points:
            x, y = point
            if self.inside_boundary((x, y, 0, 0)): # check if point is outside boundary
                grid[y][x] = '🟦'
        grid[start[1]][start[0]] = '🔴'
        if self.inside_boundary((end[0], end[1], 0, 0)):
            grid[end[1]][end[0]] = '🟢'
        grid_string = ''
        for row in grid:
            grid_string += '    ' + ''.join(row) + '\n'
        # export to txt file for visualization
        with open('line' + str(time_stamp) + '.txt', 'w') as f:
            f.write(grid_string)
        return None
        
    def handle_collision(self, state, new_state):
        start = state[0:2] # start position as (x, y)
        end = new_state[0:2] # end position as (x, y)
        bresenham_line = BresenhamLineAlgorithm.BresenhamLine(start, end) # rasterize line between start and end
        points = bresenham_line.draw_line() # get all grid points on line
        prev_point = start
        # self.export_line(points, start, end) # for debugging
        for point in points:
            x, y = point
            current_point = point # pointer reference
            if not self.inside_boundary((x, y, 0, 0)): # check if point is outside boundary
                continue # you will get to a point that is a wall before you get to a point that is outside the boundary, so it is safe to skip
            if self.world[y][x] == self.forbidden_state:
                # soft version --> reduce velocity to 0 and place agent near crash site
                if self.crash_algorithm == 'soft':
                    # nearby_state = self.find_nearby_state(x, y) # find a nearby state to place agent
                    # return (nearby_state[0], nearby_state[1], 0, 0) # reset velocity to 0
                    return (prev_point[0], prev_point[1], 0, 0) # last point on trajectory that is not a wall
                # hard version --> reset agent to start state with velocity 0
                elif self.crash_algorithm == 'harsh':
                    s = self.initialize_agent() # pick a start state index at random
                    start_state = self.index_to_state[s] # convert state index to (x, y, vx, vy)
                    return (start_state[0], start_state[1], 0, 0) # reset velocity to 0
            prev_point = current_point
        return new_state
    

    def learning_rate(self, t):
        alpha = self.alpha * np.exp(-t / (0.5 * self.training_iterations)) # exponential decay of learning rate
        return alpha
    
    def extract_policy(self, Q):
        P = [None for _ in range(len(Q))]
        for s in range(len(Q)):
            P[s] = self.get_best_action(Q, s)
        return P
    
    def export_Q(self, Q):
        csv_file_path = 'Q.csv' # file path to save CSV file
        csv_file_path = os.path.join(self.output, 'Q.csv') # directory to save file 
        # open the CSV file in write mode
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file) # create CSV writer object
            for row in Q:
                csv_writer.writerow(row) # write each row (list) to the CSV file
        return None
    
    def print_stats(self, world, S, Q, Q_history, visits, visit_history):
        Q_forbidden = [Q[s] for s, terrain in S.items() if terrain == self.forbidden_state] # values for forbidden states
        Q_forbidden_max = np.max(Q_forbidden)
        Q_forbidden_mean = np.mean(Q_forbidden)
        Q_track = [Q[s] for s, terrain in S.items() if terrain != self.forbidden_state] # values for track states
        Q_track_max = np.max(Q_track)
        Q_track_mean = np.mean(Q_track)
        Q_history['Q_forbidden_max'].append(Q_forbidden_max)
        Q_history['Q_forbidden_mean'].append(Q_forbidden_mean)
        Q_history['Q_track_max'].append(Q_track_max)
        Q_history['Q_track_mean'].append(Q_track_mean)
        Q_track_visits = {}
        for s in range(len(visits)):
            state = self.index_to_state[s]
            if world[state[1]][state[0]] != self.forbidden_state:
                Q_track_visits[state] = np.array(visits[s]) # store visit count for all track states
        Q_track_visits = np.stack(list(Q_track_visits.values())) # array manipulation
        not_visited = np.sum(Q_track_visits == 0) * 100 / (len(Q_track_visits) * len(Q_track_visits[0])) # find state-action pairs not visited
        visit_history.append(not_visited)
        return Q_history, visit_history, not_visited,  Q_forbidden_mean, Q_track_mean
    
    def q_learning(self, world):
        S, _, _, R, Q, _, _, _ = self.initialize_vars(world)
        for t in range(self.training_iterations + 1):
            t_inner = 0
            start_time = time.time()
            s, horizon = self.choose_start_state(S, t) # initialize state randomly close to terminal state
            while S[s] != self.goal_state:
                t_inner += 1
                a = self.epsilon_greedy_choice(Q, s, t) # choose action using epsilon-greedy policy
                action = self.actions[a] # action is a tuple (ddx, ddy)
                state = self.index_to_state[s] # convert state index to (x, y, vx, vy)
                # if t_inner % 1 == 0:
                #     print(f'            Inner Iteration {t_inner} --> random state: {state}, terrain: {S[s]}')
                new_state = self.apply_kinematics(state, action) # apply action to get new state
                if not self.inside_boundary(new_state):
                    # print('            Agent outside boundary')
                    Q[s][a] = -99999
                    s, horizon = self.choose_start_state(S, t) # initialize state randomly close to terminal state
                    continue
                new_s = self.state_to_index[new_state] # convert new state to state index
                Q[s][a] += self.alpha * (R[s][a] + (self.gamma * max(Q[new_s])) - Q[s][a]) # update Q function 
                s = new_s # update state
            print(f'                Inner Iteration {t_inner+1} --> random state: {self.index_to_state[s]}, terrain: {S[s]}')
            end_time = time.time()
            print(f'        Outer Iteration {t}, horizon: {horizon:.5f}, {end_time - start_time:.2f} s')
        P = self.extract_policy(Q) # extract policy from Q function
        print('Policy learning complete...')
        return P

    def sarsa(self, world):
        S, _, _, R, Q, _, _, _ = self.initialize_vars(world)
        self.reorder_states(S) # reorder track states in order of distance from goal states
        Q_history = {'Q_forbidden_max': [], 'Q_track_max': [], 'Q_forbidden_mean': [], 'Q_track_mean': []}
        visit_history = []
        visits = np.zeros((len(Q), len(Q[0])))
        outer_start_time = time.time()
        for t in range(self.training_iterations + 1):
            t_inner = 0
            escape = False
            s, horizon = self.choose_start_state(visits, t) # initialize state randomly close to terminal state
            a, greedy_epsilon, flag = self.epsilon_greedy_choice(Q, s, visits, t) # choose action using epsilon-greedy policy
            while S[s] != self.goal_state:
                t_inner += 1
                action = self.actions[a] # action is a tuple (ddx, ddy)
                state = self.index_to_state[s] # convert state index to (x, y, vx, vy)
                random_num = np.random.uniform(0, 1)
                if random_num < self.transition['fail']:
                    action = (0, 0) # no acceleration happens
                new_state = self.apply_kinematics(state, action) # apply action to get new state
                new_state = self.handle_collision(state, new_state) # handle collision with walls
                new_s = self.state_to_index[new_state] # convert new state to state index
                new_a, greedy_epsilon, flag = self.epsilon_greedy_choice(Q, new_s, visits, t) # choose action using epsilon-greedy policy
                alpha = self.learning_rate(t)
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
            Q_history, visit_history, not_visited,  Q_forbidden_mean, Q_track_mean = self.print_stats(world, S, Q, Q_history, visits, visit_history)
            print(f'        Outer Iteration {t}, horizon: {horizon:.3f}, greedy-epsilon: {greedy_epsilon:.3f}, learning rate: {alpha:.3f}, not_visited: {not_visited:.3f}%, Q_track_mean: {Q_track_mean:.3f}, inner iterations: {t_inner:0{4}d}')
        outer_end_time = time.time()
        self.learning_metrics = (Q_history, visit_history)
        self.visit_history = visits
        self.final_alpha = alpha 
        self.final_greedy_epsilon = greedy_epsilon
        self.final_not_visited = not_visited
        P = self.extract_policy(Q) # extract policy from Q function
        self.export_Q(Q) # export Q for debugging
        print('\nPolicy learning complete...')
        print(f'Training iterations: {t} --> {outer_end_time-outer_start_time:.3f}s --- {not_visited:.3f}% of position state-action pairs were not visited')
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
        if self.engine == 'value_iteration':
            engine = __value_iteration__.ValueIteration(self)
            # policy = self.value_iteration(world)
        elif self.engine == 'q_learning':
            policy = self.q_learning(world) # learn optimal policy
        elif self.engine == 'sarsa':
            policy = self.sarsa(world) # learn optimal policy
        policy = self.simulator(engine, world)
        self.function = policy
        return self.learning_metrics
    
    def initialize_agent(self):
        # get index of all start states in world 
        start_state_indices = [index for index, terrain in self.S.items() if terrain == self.start_state]
        # randomly select a start state
        start_state_index = np.random.choice(start_state_indices)
        return start_state_index

    def create_path(self, start_state_index):
        policy = self.function
        # get index of all start states in world 
        goal_state_index = [index for index, terrain in self.S.items() if terrain == self.goal_state][0]
        goal_state = self.index_to_state[goal_state_index]
        path = []
        current_state_index = start_state_index
        new_state_index = start_state_index
        count = 0 # count of self loops
        collisions = 0 # count of collisions
        status = True
        terrain = ''
        while terrain != self.goal_state:
            prev_state_index = current_state_index # pointer to prev state to detect self loop
            current_state_index = new_state_index
            current_state = self.index_to_state[current_state_index]
            action_index = policy[current_state_index]
            if action_index is None:
                # print('         Policy contains a None action. Failed to reach goal state.')
                status = False
                break
            action = self.actions[action_index]
            random_num = np.random.uniform(0, 1)
            if random_num < self.transition['fail']:
                action = (0, 0) # no acceleration happens
            new_state = self.apply_kinematics(current_state, action)
            new_state_after_collision = self.handle_collision(current_state, new_state)
            if new_state_after_collision != new_state:
                collisions += 1
            new_state = new_state_after_collision
            # path.append((current_state, action, new_state))
            if self.inside_boundary(new_state):
                new_state_index = self.state_to_index[new_state]
                if new_state_index == prev_state_index:
                    count += 1
                    if count > 10:
                        # print('         Policy generates self loop. Failed to reach goal state.')
                        status = False
                        break
            else:
                # print('         Policy generates path outside boundary. Failed to reach goal state.')
                status = False
                break
            path.append((current_state, action, new_state))
            terrain = self.S[new_state_index]
        dist_to_goal = np.linalg.norm(np.array(new_state[:2]) - np.array(goal_state[:2]))
        return path, status, dist_to_goal, collisions

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
        path, status, dist_to_goal, collisions = self.create_path(start_state_index) # create path from optimal policy
        path_metrics = {'length': len(path), 'cost': 0, 'training_iters': self.training_iterations} # calculate path metrics
        return path, path_metrics, status, dist_to_goal, collisions
    




    def simulator(self, engine, world):
        S, V, Vlast, R, Q, P, visits = self.initialize_vars(world)
        engine.state_to_index = self.state_to_index
        engine.index_to_state = self.index_to_state
        self.reorder_states(S) # reorder track states in order of distance from goal states
        outer_start_time = time.time()
        if self.engine == 'value_iteration':
            P, learning_metrics = engine.learn(S, V, Vlast, R, Q, P)
            iterations = len(learning_metrics)
            not_visited = 0
            self.learning_metrics = (learning_metrics) # delta V over iterations
        else:
            value_history = {'Q_forbidden_max': [], 'Q_track_max': [], 'Q_forbidden_mean': [], 'Q_track_mean': []}
            visit_history = []
            P, learning_metrics = engine.learn(S, V, Vlast, R, Q, P)
            pass
        outer_end_time = time.time()
        # self.learning_metrics = (value_history, visit_history)
        # self.visit_history = visits
        # self.final_alpha = alpha 
        # self.final_greedy_epsilon = greedy_epsilon
        # self.final_not_visited = not_visited
        # P = self.extract_policy(Q) # extract policy from Q function
        # self.export_Q(Q) # export Q for debugging
        print('\nPolicy learning complete...')
        print(f'Training iterations: {iterations} --> {outer_end_time-outer_start_time:.3f}s --- {not_visited:.3f}% of position state-action pairs were not visited')
        return P