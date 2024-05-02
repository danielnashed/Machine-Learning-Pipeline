import pandas as pd
import numpy as np
import copy
import itertools
import time
import random
import csv
import os
from inference_pipeline.reinforcement_learning import state as State
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
        R = self.initialize_rewards(R, S, num_rows, num_cols)
        if self.transfer_learning is None:
            Q = [[-9999]*num_actions for _ in range(num_states)] # Q function
        else:
            Q = self.initialize_via_transfer() # initialize Q function using transfer learning 
        P = [None for _ in range(num_states)] # policy

        # # remove invalid states from S
        # valid_S = copy.deepcopy(S)
        # for s in S.keys():
        #     is_valid = self.valid(s)
        #     if not is_valid:
        #         valid_S.pop(s)
        # S = valid_S

        return (S, V, Vlast, R, Q, P, num_rows, num_cols)
    
    def initialize_via_transfer(self):
        with open(self.transfer_learning, 'r') as f:
            reader = csv.reader(f)
            Q = list(reader)
            # Q = [[None]*len(data[0]) for _ in range(len(data))] # Q function
            # for s in range(len(data)):
            #     for a in range(len(data[0])):
            #         Q[s][a] = data[s][a]
        return Q
    
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
        # if all actions are equally of equal value, choose an action randomly 
        if all([x == -9999 for x in lst]):
            return np.random.choice(range(len(lst)))
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
    
    # def choose_start_state(self, S, t): # another solution: maybe choose states with probability to inverse of their frequency of visit
    #     # normalizer_power = 3
    #     # peak_time = 0.8 # 0.3
    #     # horizon = min(1, 10**(-normalizer_power) * np.exp(t * np.log(10**normalizer_power) / (peak_time * self.training_iterations))) # exponential growth of horizon from 0.001 to 1
    #     if t < 500:
    #         horizon = 0
    #     else:
    #         horizon = 1
    #     num_states = len(S)
    #     goal_state_indices = [index for index, terrain in S.items() if terrain == self.goal_state]
    #     min_goal_state_index = min(goal_state_indices)
    #     max_goal_state_index = max(goal_state_indices)
    #     lower_limit = min_goal_state_index - (horizon * num_states) # maybe try including 0.5 factor
    #     upper_limit = max_goal_state_index + (horizon * num_states)
    #     lower_limit = int(max(0, lower_limit))
    #     upper_limit = int(min(num_states - 1, upper_limit))
    #     # lower_limit = 0
    #     # upper_limit = len(S) - 1
    #     s = np.random.choice(range(lower_limit, upper_limit + 1))
    #     # s = np.random.choice(range(min_goal_state_index, max_goal_state_index + 1))
    #     # kep history of visits to a state and over time, increase probability of visiting states that have not been visited
    #     return s, horizon

    # def choose_start_state(self, S, t): # another solution: maybe choose states with probability to inverse of their frequency of visit
    #     # normalizer_power = 3
    #     # peak_time = 0.8 # 0.3
    #     # horizon = min(1, 10**(-normalizer_power) * np.exp(t * np.log(10**normalizer_power) / (peak_time * self.training_iterations))) # exponential growth of horizon from 0.001 to 1
    #     track_state_indices = [(index, terrain) for index, terrain in S.items() if terrain != self.forbidden_state]
    #     goal_state_indices = [index for index, terrain in track_state_indices if terrain == self.goal_state]
    #     min_goal_state_index = min(goal_state_indices)
    #     max_goal_state_index = max(goal_state_indices)
    #     if t < 200:
    #         horizon = 0
    #         lower_limit = min_goal_state_index
    #         upper_limit = max_goal_state_index
    #         s = np.random.choice(range(lower_limit, upper_limit + 1)) # pick a valid state near goal state to start
    #     else:
    #         horizon = 1
    #         track_state_indices = [index for index, terrain in track_state_indices]
    #         s = np.random.choice(track_state_indices) # pick any valid state to start
    #     return s, horizon

    # def choose_start_state(self, S, t): # another solution: maybe choose states with probability to inverse of their frequency of visit
    #     # normalizer_power = 3
    #     # peak_time = 0.8 # 0.3
    #     # horizon = min(1, 10**(-normalizer_power) * np.exp(t * np.log(10**normalizer_power) / (peak_time * self.training_iterations))) # exponential growth of horizon from 0.001 to 1
    #     track_state_indices = [index for index, terrain in S.items() if terrain != self.forbidden_state and terrain != self.goal_state]
    #     s = np.random.choice(track_state_indices) # pick any valid state to start
    #     horizon = 1
    #     # goal_state_indices = [index for index, terrain in track_state_indices if terrain == self.goal_state]
    #     # min_goal_state_index = min(goal_state_indices)
    #     # max_goal_state_index = max(goal_state_indices)
    #     # if t < 200:
    #     #     horizon = 0
    #     #     lower_limit = min_goal_state_index
    #     #     upper_limit = max_goal_state_index
    #     #     s = np.random.choice(range(lower_limit, upper_limit + 1)) # pick a valid state near goal state to start
    #     # else:
    #     #     horizon = 1
    #     #     track_state_indices = [index for index, terrain in track_state_indices]
    #     #     s = np.random.choice(track_state_indices) # pick any valid state to start
    #     return s, horizon
    
    # def choose_start_state(self, S, visits, t): # another solution: maybe choose states with probability to inverse of their frequency of visit
    #     # normalizer_power = 3
    #     # peak_time = 0.8 # 0.3
    #     # horizon = min(1, 10**(-normalizer_power) * np.exp(t * np.log(10**normalizer_power) / (peak_time * self.training_iterations))) # exponential growth of horizon from 0.001 to 1
    #     # state_visit_freq = {index: sum(visits[index]) for index in self.track_state_indices}
    #     # state_visit_prob = {index: 1 / (1 + freq) for index, freq in state_visit_freq.items()}
    #     # state_visit_normalized_prob = {index: prob / sum(state_visit_prob.values()) for index, prob in state_visit_prob.items()}
    #     indices = np.array(self.track_state_indices)
    #     freqs = np.sum(visits[self.track_state_indices], axis=1)
    #     probs = 1 / (1 + freqs) # probability inversely proportional to frequency of visit
    #     normalized_probs = probs / sum(probs) # normalize probabilities so all sum up to 1
    #     s = random.choices(list(indices), weights=list(normalized_probs), k=1)[0]
    #     # s = None
    #     # area = 0
    #     # if t == 1000:
    #     #     debug = ''
    #     # # choose a state based on the cumulative distribution
    #     # random_num = np.random.uniform(0, 1)
    #     # # create a cumulative distribution of state visit probabilities
    #     # for index, prob in state_visit_normalized_prob.items():
    #     #     area += prob
    #     #     if random_num <= area:
    #     #         s = index
    #     #         break
    #         # state_visit_normalized_prob[index] = area
    #     # choose a state based on the cumulative distribution
    #     # random_num = np.random.uniform(0, 1)
    #     # s = None
    #     # for index, prob in state_visit_normalized_prob.items():
    #     #     if random_num <= prob:
    #     #         s = index
    #     #         break
    #     horizon = 1
    #     return s, horizon

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

    def choose_start_state(self, visits, t):
        initial_horizon = 10 # start with 10 states at start near terminal state
        peak_time = 0.8 # time when horizon covers entire space of indices
        growth_factor = np.log(len(self.track_state_indices) / initial_horizon) / peak_time # control growth rate
        horizon = min(len(self.track_state_indices) ,initial_horizon * np.exp(t * growth_factor / self.training_iterations)) # exponential growth of horizon from 10 states to all track states
        indices = self.track_state_indices[:int(horizon)] # expanding horizon of indices
        freqs = np.sum(visits[indices], axis=1)
        probs = 1 / (1 + freqs) # probability inversely proportional to frequency of visit
        normalized_probs = probs / sum(probs) # normalize probabilities so all sum up to 1
        s = random.choices(list(indices), weights=list(normalized_probs), k=1)[0] # choose a state according to its probability
        return s, horizon/len(self.track_state_indices)
    
    def epsilon_greedy_choice(self, Q, s, t):
        # epsilon = 0.25 * (1 / (t + 1)) # epsilon decreases over time (exploration-exploitation tradeoff)
        # epsilon = 1.0 * np.exp(-t / (0.8 * self.training_iterations)) # exponential decay of epsilon, #0.5
        epsilon = 1 - (t /self.training_iterations) # linear decay from 1.0 to 0.1 up to 90% of iterations
        if t > 0.9 * self.training_iterations:
            epsilon = 0.1
        if t == 0:
            self.initial_greedy_epsilon = epsilon
        values = list(Q[s])
        random_num = np.random.uniform(0, 1)
        # exploration choice has probability epsilon of choosing a random action
        if random_num < epsilon:
            return np.random.choice(range(len(values))), epsilon
        # exploitation choice (greedy) has probability 1 - epsilon of choosing the best action
        else:
            return self.get_best_action(Q, s), epsilon
        
    def find_nearby_state(self, x, y, offset=1):
        unit_moves = [-1, 0, 1]
        moves1 = list(itertools.product(unit_moves, unit_moves))
        if offset == 1:
            moves = moves1
        else:
            offset_moves = list(range(-offset, offset + 1))
            moves2 = list(itertools.product(offset_moves, offset_moves))
            moves = list(set(moves2) - set(moves1))
        neighbors = []
        for move in moves:
            new_x = x + move[0]
            new_y = y + move[1]
            if self.inside_boundary((new_x, new_y, 0, 0)):
                if self.world[new_y][new_x] != self.forbidden_state:
                    neighbors.append((new_x, new_y))
        if neighbors != []:
                nearby_state = random.choice(neighbors)

                return nearby_state # return a random nearby neighbor
        else:
            offset += 1
            nearby_state = self.find_nearby_state(x, y, offset) # recursively find a nearby state
        return nearby_state
    
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
        # self.export_line(points, start, end) # for debugging
        for point in points:
            x, y = point
            if not self.inside_boundary((x, y, 0, 0)): # check if point is outside boundary
                continue # you will get to a point that is a wall before you get to a point that is outside the boundary, so it is safe to skip
            if self.world[y][x] == self.forbidden_state:
                # soft version --> reduce velocity to 0 and place agent near crash site
                ### FIX: iterate through points starting from start and stop when you reach a forbidden state. Then assign the state right before that to be the nearby state.
                if self.crash_algorithm == 'soft':
                    nearby_state = self.find_nearby_state(x, y) # find a nearby state to place agent
                    return (nearby_state[0], nearby_state[1], 0, 0) # reset velocity to 0
                # hard version --> reset agent to start state with velocity 0
                elif self.crash_algorithm == 'harsh':
                    s = self.initialize_agent() # pick a start state index at random
                    start_state = self.index_to_state[s] # convert state index to (x, y, vx, vy)
                    return (start_state[0], start_state[1], 0, 0) # reset velocity to 0
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
        # Q_track = [[s] + [self.index_to_state[s]] + row for s, row in enumerate(Q) if self.S[s] != self.forbidden_state]
        # File path to save the CSV file
        csv_file_path = 'Q.csv'
        # directory to save file 
        csv_file_path = os.path.join(self.output, 'Q.csv')
        # Open the CSV file in write mode
        with open(csv_file_path, 'w', newline='') as csv_file:
            # Create a CSV writer object
            csv_writer = csv.writer(csv_file)
            # Write each row (list) to the CSV file
            for row in Q:
                csv_writer.writerow(row)
        return None
    
    def print_stats(self, world, S, Q, Q_history, visits, visit_history):
        Q_forbidden = [Q[s] for s, terrain in S.items() if terrain == self.forbidden_state]
        Q_forbidden_max = np.max(Q_forbidden)
        Q_forbidden_mean = np.mean(Q_forbidden)
        Q_track = [Q[s] for s, terrain in S.items() if terrain != self.forbidden_state]
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
                Q_track_visits[state] = np.array(visits[s])

        # Q_track_visits = {}
        # for s in range(len(visits)):
        #     state = self.index_to_state[s]
        #     if world[state[1]][state[0]] != self.forbidden_state:
        #         pos = state[:2]
        #         # Q_track_visits[pos] = np.array(copy.deepcopy(visits[s])) # state position regardless of velocity
        #         if pos not in Q_track_visits:
        #             Q_track_visits[pos] = np.array(visits[s])
        #         else:
        #             Q_track_visits[pos] += np.array(visits[s])




                # Q_track_visits.append(visits[s])
        # look at state-action pairs not visited
        # Q_track_visits = np.array(Q_track_visits)
        # not_visited = np.sum(Q_track_visits == 0) * 100 / (len(Q_track_visits) * len(Q_track_visits[0]))
        # look at states not visited regardless of velocity
        # Q_track_visits = np.array(Q_track_visits.values())
        Q_track_visits = np.stack(list(Q_track_visits.values()))
        not_visited = np.sum(Q_track_visits == 0) * 100 / (len(Q_track_visits) * len(Q_track_visits[0]))
        visit_history.append(not_visited)
        return Q_history, visit_history, not_visited,  Q_forbidden_mean, Q_track_mean

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
            start_time = time.time()
            s, horizon = self.choose_start_state(visits, t) # initialize state randomly close to terminal state
            a, greedy_epsilon = self.epsilon_greedy_choice(Q, s, t) # choose action using epsilon-greedy policy
            while S[s] != self.goal_state:
                action = self.actions[a] # action is a tuple (ddx, ddy)
                state = self.index_to_state[s] # convert state index to (x, y, vx, vy)
                random_num = np.random.uniform(0, 1)
                if random_num < self.transition['fail']:
                    action = (0, 0) # no acceleration happens
                t_inner += 1
                new_state = self.apply_kinematics(state, action) # apply action to get new state
                new_state = self.handle_collision(state, new_state) # handle collision with walls
                new_s = self.state_to_index[new_state] # convert new state to state index
                new_a, greedy_epsilon = self.epsilon_greedy_choice(Q, new_s, t) # choose action using epsilon-greedy policy
                alpha = self.learning_rate(t)
                Q[s][a] += alpha * (R[s][a] + (self.gamma * Q[new_s][new_a]) - Q[s][a]) # update Q function 
                visits[s][a] += 1 # update visit count
                s = new_s # update state
                a = new_a # update action
            end_time = time.time()
            Q_history, visit_history, not_visited,  Q_forbidden_mean, Q_track_mean = self.print_stats(world, S, Q, Q_history, visits, visit_history)
            print(f'        Outer Iteration {t}, horizon: {horizon:.3f}, greedy-epsilon: {greedy_epsilon:.3f}, learning rate: {alpha:.3f}, not_visited: {not_visited:.3f}%, Q_track_mean: {Q_track_mean:.3f}, inner iterations: {t_inner+1:0{4}d}, {end_time - start_time:.2f}s')
        outer_end_time = time.time()
        self.learning_metrics = (Q_history, visit_history)
        self.visit_history = visits
        self.final_alpha = alpha 
        self.final_greedy_epsilon = greedy_epsilon
        self.final_not_visited = not_visited
        P = self.extract_policy(Q) # extract policy from Q function
        self.export_Q(Q) # export Q for debugging
        print('\nPolicy learning complete...')
        print(f'Training iterations: {t} --> {outer_end_time-outer_start_time:.3f} --- {not_visited:.3f}% of position state-action pairs were not visited')
        return P


    # def sarsa(self, world):
    #     S, _, _, R, Q, _, _, _ = self.initialize_vars(world)
    #     Q_history = {'Q_forbidden_max': [], 'Q_track_max': [], 'Q_forbidden_mean': [], 'Q_track_mean': []}
    #     visit_history = []
    #     visits = np.zeros((len(Q), len(Q[0])))
    #     num_Q_values = len(Q) * len(Q[0])
    #     for t in range(self.training_iterations + 1):
    #         t_inner = 0
    #         start_time = time.time()
    #         s, horizon = self.choose_start_state(S, t) # initialize state randomly close to terminal state
    #         a = self.epsilon_greedy_choice(Q, s, t) # choose action using epsilon-greedy policy
    #         while S[s] != self.goal_state:
    #             action = self.actions[a] # action is a tuple (ddx, ddy)
    #             state = self.index_to_state[s] # convert state index to (x, y, vx, vy)
    #             random_num = np.random.uniform(0, 1)
    #             if random_num < self.transition['fail']:
    #                 action = (0, 0) # no acceleration happens
    #             t_inner += 1
    #             # if t_inner % 1 == 0:
    #             #     print(f'            Inner Iteration {t_inner} --> random state: {state}, terrain: {S[s]}')
    #             new_state = self.apply_kinematics(state, action) # apply action to get new state
    #             new_state = self.handle_collision(state, new_state) # handle collision with walls
    #             # if not self.inside_boundary(new_state):
    #             #     Q[s][a] = -99999 # do i ever enter this line ???
    #             #     s, horizon = self.choose_start_state(S, t) # initialize state randomly close to terminal state
    #             #     a = self.epsilon_greedy_choice(Q, s, t) # choose action using epsilon-greedy policy
    #             #     continue
    #             new_s = self.state_to_index[new_state] # convert new state to state index
    #             new_a = self.epsilon_greedy_choice(Q, new_s, t) # choose action using epsilon-greedy policy
    #             Q[s][a] += self.alpha * (R[s][a] + (self.gamma * Q[new_s][new_a]) - Q[s][a]) # update Q function 
    #             visits[s][a] += 1 # update visit count
    #             s = new_s # update state
    #             a = new_a # update action
    #         # print(f'                Inner Iteration {t_inner+1} --> random state: {self.index_to_state[s]}, terrain: {S[s]}')
    #         end_time = time.time()
    #         Q_history, visit_history, not_visited,  Q_forbidden_mean, Q_track_mean = self.print_stats(world, S, Q, Q_history, visits, visit_history)
    #         # Q_forbidden = [Q[s] for s, terrain in S.items() if terrain == self.forbidden_state]
    #         # Q_forbidden_max = np.max(Q_forbidden)
    #         # Q_forbidden_mean = np.mean(Q_forbidden)
    #         # Q_track = [Q[s] for s, terrain in S.items() if terrain != self.forbidden_state]
    #         # Q_track_max = np.max(Q_track)
    #         # Q_track_mean = np.mean(Q_track)
    #         # Q_history['Q_forbidden_max'].append(Q_forbidden_max)
    #         # Q_history['Q_forbidden_mean'].append(Q_forbidden_mean)
    #         # Q_history['Q_track_max'].append(Q_track_max)
    #         # Q_history['Q_track_mean'].append(Q_track_mean)
    #         # Q_track_visits = []
    #         # for s in range(len(visits)):
    #         #     state = self.index_to_state[s]
    #         #     if world[state[1]][state[0]] != self.forbidden_state:
    #         #         Q_track_visits.append(visits[s])
    #         # Q_track_visits = np.array(Q_track_visits)
    #         # not_visited = np.sum(Q_track_visits == 0) * 100 / (len(Q_track_visits) * len(Q_track_visits[0]))
    #         # visit_history.append(not_visited)
    #         print(f'        Outer Iteration {t}, horizon: {horizon:.3f}, not_visited: {not_visited:.3f}%, Q_forbidden_mean: {Q_forbidden_mean:.3f}, Q_track_mean: {Q_track_mean:.3f}, inner iterations: {t_inner+1:0{4}d}, {end_time - start_time:.2f}s')
    #     self.learning_metrics = (Q_history, visit_history)
    #     self.visit_history = visits
    #     # num_Q_values = len(Q) * len(Q[0])
    #     # not_visited = np.sum(visits == 0) * 100 / num_Q_values
    #     P = self.extract_policy(Q) # extract policy from Q function
    #     print('Policy learning complete...')
    #     print(f'        Training iterations: {t} --> {not_visited:.3f}% of Q values were not visited')
    #     return P

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
            policy = self.value_iteration(world)
        elif self.engine == 'q_learning':
            policy = self.q_learning(world) # learn optimal policy
        elif self.engine == 'sarsa':
            policy = self.sarsa(world) # learn optimal policy
        self.function = policy
        return self.learning_metrics
        # return None
    
    def initialize_agent(self):
        # get index of all start states in world 
        start_state_indices = [index for index, terrain in self.S.items() if terrain == self.start_state]
        # randomly select a start state
        start_state_index = np.random.choice(start_state_indices)
        return start_state_index
    
    # def create_path(self, start_state_index):
    #     policy = self.function
    #     path = []
    #     current_state_index = start_state_index
    #     current_state = self.index_to_state[current_state_index]
    #     action_index = policy[current_state_index]
    #     if action_index is None:
    #         print('         Policy contains a None action. Failed to reach goal state.')
    #         return path
    #     action = self.actions[action_index]
    #     new_state = self.apply_kinematics(current_state, action)
    #     # path.append((current_state, action, new_state))
    #     if self.inside_boundary(new_state):
    #         new_state_index = self.state_to_index[new_state]
    #     else:
    #         print('         Policy generates path outside boundary. Failed to reach goal state.')
    #         return path
    #     path.append((current_state, action, new_state))
    #     terrain = self.S[new_state_index]
    #     while terrain != self.goal_state:
    #         prev_state_index = current_state_index # pointer to prev state to detect self loop
    #         current_state_index = new_state_index
    #         current_state = self.index_to_state[current_state_index]
    #         action_index = policy[current_state_index]
    #         if action_index is None:
    #             print('         Policy contains a None action. Failed to reach goal state.')
    #             return path
    #         action = self.actions[action_index]
    #         new_state = self.apply_kinematics(current_state, action)
    #         # path.append((current_state, action, new_state))
    #         if self.inside_boundary(new_state):
    #             new_state_index = self.state_to_index[new_state]
    #             if new_state_index == prev_state_index:
    #                 print('         Policy generates self loop. Failed to reach goal state.')
    #                 return path
    #         else:
    #             print('         Policy generates path outside boundary. Failed to reach goal state.')
    #             return path
    #         path.append((current_state, action, new_state))
    #         terrain = self.S[new_state_index]
    #     return path

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