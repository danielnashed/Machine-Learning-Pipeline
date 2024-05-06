import pandas as pd
import numpy as np
import copy
import itertools
import time
import random
import csv
import os
from inference_pipeline.reinforcement_learning import value_iteration as __value_iteration__
from inference_pipeline.reinforcement_learning import sarsa as __sarsa__
from inference_pipeline.reinforcement_learning import q_learning as __q_learning__
from inference_pipeline.reinforcement_learning import bresenham_line as BresenhamLineAlgorithm

"""
This module contains the ReinforcementLearning class which is used to train a model using reinforcement learning. It contains several utility functions used by
several RL algorithms such as value iteration, SARSA, and Q-learning. The ReinforcementLearning class is responsible for training the model using reinforcement
learning, simulating the environment with which the agent interacts with, and testing the model by generating a path through the environment using the optimal
policy extracted from the different RL algorithms.

The ReinforcementLearning class contains the following attributes:
    - hyperparameters: hyperparameters for the model
    - function: function learned from training data
    - engine: engine for agent
    - crash_algorithm: crash algorithm for agent
    - start_state: start state for agent
    - goal_state: goal state for agent
    - forbidden_state: forbidden state for agent
    - actions: permissable actions
    - costs: costs for each state
    - transition: transition probabilities
    - velocity_limit: velocity limits
    - training_iterations: number of training iterations
    - reward: reward for goal state
    - alpha: learning rate
    - gamma: discount factor
    - convergence_epsilon: epsilon for convergence for value iteration
    - world: world for agent
    - visit_history: history of visits to states
    - transfer_learning: directory for transfer learning
    - output: path to save functions to
    - learning_metrics: logs of model metric as function of training iterations

The ReinforcementLearning class contains the following methods:
    - set_params: set the hyperparameters for the model
    - mapping: create a mapping of states to terrain
    - inside_boundary: check if state is inside boundary
    - apply_kinematics: apply kinematics to state
    - initialize_rewards: initialize rewards for each state
    - initialize_vars: initialize variables for model
    - initialize_via_transfer: initialize Q function using transfer learning
    - get_best_action: get the best action for a state
    - reorder_states: reorder states based on distance to goal state
    - choose_start_state: choose a start state for agent
    - epsilon_greedy_choice: choose an action using epsilon-greedy policy
    - pretty_grid: convert grid to emojis for visualization
    - export_line: export line to txt file for visualization
    - handle_collision: handle collision with wall
    - learning_rate: calculate learning rate
    - extract_policy: extract policy from Q function
    - export_Q: export function to txt file
    - print_stats: print stats for model
    - fit: train the model using reinforcement learning
    - initialize_agent: initialize agent at start state
    - create_path: create path for agent
    - simulator: simulate the model using reinforcement learning
    - predict: test the model using reinforcement learning
"""
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

    """
    'mapping' method is responsible for creating a mapping of states to terrain. Each state is represented as a tuple (x, y, vx, vy) where x and y are the
    coordinates of the agent and vx and vy are the velocities of the agent in the x and y directions respectively. The mapping is stored in a dictionary where
    the key is the state index and the value is the terrain of the state. The terrain is extracted from the world grid.
    Args:
        world (list): world for agent
    Returns:
        S (dictionary): mapping of states to terrain
    """
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

    """
    'inside_boundary' method is responsible for checking if a state is inside the boundary of the world. The state is represented as a tuple (x, y, vx, vy)
    where x and y are the coordinates of the agent and vx and vy are the velocities of the agent in the x and y directions respectively. The method returns
    True if the state is inside the boundary of the world grid and False otherwise.
    Args:
        state (tuple): state of agent
    Returns:
        bool: True if state is inside boundary, False otherwise
    """
    def inside_boundary(self, state):
        x, y, x_vel, y_vel = state
        if x < 0 or x >= len(self.world[0]) or y < 0 or y >= len(self.world):
            return False
        return True

    """
    'apply_kinematics' method is responsible for applying kinematics to a state. The method updates the velocity of the agent based on the acceleration and 
    caps the velocity at the velocity limit. The method then updates the position of the agent based on the new velocity.
    Args:
        current_state (tuple): current state of agent (x, y, vx, vy)
        action (tuple): action to apply to agent (ddx, ddy)
    Returns:
        tuple: new state of agent
    """
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

    """
    'initialize_rewards' method is responsible for initializing the rewards for each state. The method sets the reward for the goal state to the reward value
    and sets the cost for each state to the cost of the terrain of the state. The method returns the reward matrix.
    Args:
        R (list): reward matrix
        S (dictionary): mapping of states to terrain
    Returns:
        R (list): reward matrix
    """
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
                    R[s][a] = self.costs[terrain] # set cost to enter new state terrain
        return R

    """
    'initialize_vars' method is responsible for initializing the variables for the model. The method initializes the state space, value function, last value
    function, reward function, Q function, policy, and visit history. The method returns the initialized variables.
    Args:
        world (list): world for agent
    Returns:
        tuple: initialized variables
    """ 
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
            Q = [[-9]*num_actions for _ in range(num_states)] # Q function
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
    
    """
    'initialize_via_transfer' method is responsible for initializing the Q function using transfer learning. The method reads the Q function from a csv file
    and returns the Q function.
    Returns:
        Q (list): Q function
    """
    def initialize_via_transfer(self):
        with open(self.transfer_learning, 'r') as f:
            reader = csv.reader(f)
            # Q = list(reader)
            Q = [[float(cell) for cell in row] for row in reader]
        return Q

    """
    'get_best_action' method is responsible for getting the best action for a state. The method returns the index of the best action in the Q function for
    the state.
    Args:
        Q (list): Q function
        s (int): state index
    Returns:
        int: index of best action in Q function
    """
    def get_best_action(self, Q, s):
        lst = np.array(Q[s])
        # if all actions are of equal value, choose an action randomly 
        if len(np.unique(lst)) == 1:
            return np.random.choice(range(len(lst)))
        action_index = np.argmax(lst) # index of best action in Q[s]
        return action_index
    
    """
    'reorder_states' method is responsible for reordering the states based on the distance to the goal state. The method calculates the euclidean distance
    from each state to the goal state and sorts the states based on the distance to the goal state. This can be used to choose a start state for the agent
    that is close to the goal state and can help the agent learn the optimal policy faster.
    Args:
        S (dictionary): mapping of states to terrain
    Returns:
        None
    """
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

    """
    'choose_start_state' method is responsible for choosing a start state for the agent. The method chooses a start state based on the number of visits to each
    state. The probability of picking a state is inversely proportional to the frequency of visit to that state. This helps the agent explore more previously
    unseen states and speed up concergence. 
    
    Moreover, the method uses a limited receptive field that moves from terminal states to start states. The receptive field can be thought of as a block of 
    states of limited number that sweeps across the state space. The block has a tail end and a front end to limit the range of states avaiable for picking.
    The agent can only choose a start state from the states within that receptive field. The speed at which the receptive field moves across the state space 
    is controlled by its velocity and peak time at which it reaches the start states. These are hyperparameters that can be tuned. Two peak times are provided. 
    The first peak time is when the horizon covers the entire space of indices. The delta between first peak time and second peak time is the time spent focusing 
    only on start states. Beyond peak time 2, the horizon expands to cover the entire space of state inices. Note that if both peak times are set to 0, the
    receptive field simply covers the entire space of state indices from the first episode. 
    Args:
        visits (list): history of visits to states
        t (int): training iteration 
    Returns:
        tuple: start state and frontier position
    """
    def choose_start_state(self, visits, t):
        peak_time_1 = 0.0 # time to reach start states 
        peak_time_2 = 0.0 # time to expand horizon to cover all states
        receptive_field = 4000 # max number of states to look at for choosing a state
        # limited receptive field that moves from terminal states to start states
        if t < peak_time_1 * self.training_iterations:
            initial_frontier_pos = 2000 # start 2000 states at start near terminal state
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
    
    """
    'epsilon_greedy_choice' method is responsible for choosing an action using an epsilon-greedy policy. The method chooses an action using an epsilon-greedy
    policy where the agent explores with probability epsilon and exploits with probability 1 - epsilon. The method also returns the value of epsilon and a flag
    indicating whether the agent is exploring or exploiting. For exploration, the method picks an action that has not been visited frequently in the past in
    the current state to encourage exploration of new states. This is achieved by sampling actions with a probability inversely proportional to their frequency 
    of visits in the past. For exploitation, the method picks the best action based on the Q function for the current state.
    Args:
        Q (list): Q function
        s (int): state index
        visits (list): history of visits to states
        t (int): training iteration
    Returns:
        tuple: action, epsilon, flag
    """
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
        return a, epsilon, flag
    
    """
    'pretty_grid' method is responsible for converting the grid to emojis for visualization. The method converts the grid to emojis for visualization purposes.
    Args:
        grid (list): grid to convert to emojis
    Returns:
        list: grid converted to emojis
    """
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
    
    """
    'export_line' method is responsible for exporting a line to a txt file for visualization. The method exports a line to a txt file for visualization purposes.
    Args:
        points (list): points on the line
        start (tuple): start position
        end (tuple): end position
    Returns:
        None
    """
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
    
    """
    'handle_collision' method is responsible for handling a collision with a wall. The method checks if the new state is outside the boundary of the world or if
    the new state is a forbidden state. Two versions of crash are provided. In the soft case, if a collision is detected with a wall, the agent is placed near 
    the wall on the track where it collided and its velocity is set to 0. In the harsh case, if a collision is detected with a wall, the agent is reset to the
    start state with velocity 0. The collision detection makes use of the Bresenham Line Algorithm to rasterize a line between the start and end positions of
    the agent and check for collisions with walls.
    Args:
        state (tuple): current state of agent
        new_state (tuple): new state of agent
    Returns:
        tuple: new state of agent
    """
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
                    return (prev_point[0], prev_point[1], 0, 0) # last point on trajectory that is not a wall
                # harsh version --> reset agent to start state with velocity 0
                elif self.crash_algorithm == 'harsh':
                    s = self.initialize_agent() # pick a start state index at random
                    start_state = self.index_to_state[s] # convert state index to (x, y, vx, vy)
                    return (start_state[0], start_state[1], 0, 0) # reset velocity to 0
            prev_point = current_point
        return new_state
    
    """
    'learning_rate' method is responsible for anneahling the learning rate. The method calculates the learning rate using an exponential decay function that
    decreases the learning rate over time. The learning rate is calculated as alpha * exp(-t / (0.5 * training_iterations)) where t is the training iteration.
    This helps the agent to learn the optimal policy faster by allowing the agent to learn more at the start and less at the end of training.
    Args:
        t (int): training iteration
    Returns:
        float: learning rate
    """
    def learning_rate(self, t):
        alpha = self.alpha * np.exp(-t / (0.5 * self.training_iterations)) # exponential decay of learning rate
        return alpha
    
    """
    'extract_policy' method is responsible for extracting the policy from the Q function. The method achieves that by choosing the best action for each state 
    based on the Q function. The method returns the policy.
    Args:
        Q (list): Q function
    Returns:
        list: policy
    """
    def extract_policy(self, Q):
        P = [None for _ in range(len(Q))]
        for s in range(len(Q)):
            P[s] = self.get_best_action(Q, s)
        return P
    
    """
    'export_Q' method is responsible for exporting the Q function to a CSV file. The method exports the Q function to a CSV file for visualization purposes.
    Args:
        Q (list): Q function
    Returns:
        None
    """
    def export_Q(self, Q):
        csv_file_path = 'Q.csv' # file path to save CSV file
        csv_file_path = os.path.join(self.output, 'Q.csv') # directory to save file 
        # open the CSV file in write mode
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file) # create CSV writer object
            for row in Q:
                csv_writer.writerow(row) # write each row (list) to the CSV file
        return None
    
    """
    'print_stats' method is responsible for printing the stats for the model. The method calculates the mean and max Q values for forbidden states and track
    states. The method also calculates the percentage of position state-action pairs that were not visited during training. The method returns the Q history,
    visit history, percentage of not visited state-action pairs, mean Q value for forbidden states, and mean Q value for track states.
    Args:
        Q (list): Q function
        Q_history (dictionary): history of Q values
        visits (list): history of visits to states
        visit_history (list): history of visits to states
    Returns:
        tuple: Q history, visit history, percentage of not visited state-action pairs, mean Q value for forbidden states, mean Q value for track states
    """
    def print_stats(self, Q, Q_history, visits, visit_history):
        Q_forbidden = [Q[s] for s in self.Q_forbidden_indices] # values for forbidden states
        Q_forbidden_max = np.max(Q_forbidden)
        Q_forbidden_mean = np.mean(Q_forbidden)
        Q_track = [Q[s] for s in self.Q_track_indices] # values for track states
        Q_track_max = np.max(Q_track)
        Q_track_mean = np.mean(Q_track)
        Q_history['Q_forbidden_max'].append(Q_forbidden_max)
        Q_history['Q_forbidden_mean'].append(Q_forbidden_mean)
        Q_history['Q_track_max'].append(Q_track_max)
        Q_history['Q_track_mean'].append(Q_track_mean)
        Q_track_visits = {}
        for s in self.Q_track_indices:
            state = self.index_to_state[s]
            Q_track_visits[state] = np.array(visits[s]) # store visit count for all track states
        Q_track_visits = np.stack(list(Q_track_visits.values())) # array manipulation
        not_visited = np.sum(Q_track_visits == 0) * 100 / (len(Q_track_visits) * len(Q_track_visits[0])) # find state-action pairs not visited
        visit_history.append(not_visited)
        return Q_history, visit_history, not_visited,  Q_forbidden_mean, Q_track_mean
    
    """
    'fit' method is responsible for training the model using reinforcement learning. The method trains the model using the specified engine and simulator. The 
    method runs the RL engine on the world to learn the policy. The method returns the learning metrics.
    Args:
        world (list): world for agent
    Returns:
        list: learning metrics
    """
    def fit(self, world):
        print(f'        Training RL agent using {self.engine}...')
        self.world = world
        if self.engine == 'value_iteration':
            engine = __value_iteration__.ValueIteration(self)
        elif self.engine == 'q_learning':
            engine = __q_learning__.QLearning(self)
        elif self.engine == 'sarsa':
            engine = __sarsa__.SARSA(self)
        policy = self.simulator(engine, world) # run RL engine on world to learn policy
        self.function = policy
        return self.learning_metrics
    
    """
    'simulator' method is responsible for simulating the model using reinforcement learning. The method initializes the variables for the model and runs the RL
    engine on the world to learn the policy. The method returns the policy.
    Args:
        engine (object): RL engine
        world (list): world for agent
    Returns:
        list: policy
    """
    def simulator(self, engine, world):
        S, V, Vlast, R, Q, P, visits = self.initialize_vars(world)
        self.Q_forbidden_indices = [s for s, terrain in S.items() if terrain == self.forbidden_state] # indices for forbidden states
        self.Q_track_indices = [s for s, terrain in S.items() if terrain != self.forbidden_state] # values for track states
        self.start_state_indices = [index for index, terrain in self.S.items() if terrain == self.start_state] # indices for start states
        engine.state_to_index = self.state_to_index
        engine.index_to_state = self.index_to_state
        outer_start_time = time.time()
        if self.engine == 'value_iteration':
            P, learning_metrics = engine.learn(S, V, Vlast, R, Q, P)
            iterations = len(learning_metrics)
            not_visited = 0
            self.learning_metrics = (learning_metrics) # delta V over iterations
        else:
            self.reorder_states(S) # reorder track states in order of distance from goal states
            P, learning_metrics = engine.learn(S, R, Q, visits)
            Q_history, visit_history, visits, alpha, greedy_epsilon, not_visited = learning_metrics
            self.visit_history = visits
            self.final_alpha = alpha 
            self.final_greedy_epsilon = greedy_epsilon
            self.final_not_visited = not_visited
            iterations = self.training_iterations
            self.learning_metrics = (Q_history, visit_history) # Q and visit history
        outer_end_time = time.time()
        print('\nPolicy learning complete...')
        print(f'Training iterations: {iterations} --> {outer_end_time-outer_start_time:.3f}s --- {not_visited:.3f}% of position state-action pairs were not visited')
        return P
    
    """
    'initialize_agent' method is responsible for initializing the agent at a start state. The method gets the index of all start states in the world and
    randomly selects a start state. The method returns the index of the start state.
    Returns:
        int: index of start state
    """
    def initialize_agent(self):
        # get index of all start states in world 
        # start_state_indices = [index for index, terrain in self.S.items() if terrain == self.start_state]
        # randomly select a start state
        start_state_index = np.random.choice(self.start_state_indices)
        # start_state_index = np.random.choice(start_state_indices)
        return start_state_index

    """
    'create_path' method is responsible for creating a path for the agent using the optimal policy. The method applies kinematics to the agent to update its 
    position and velocity based on the action chosen by the policy in the corresponding state. The agent is placed at a start state and the algorithm runs until 
    the agent reaches a goal state. If the agent however is stuck in a self-loop where it oscillates back and forth between two states, we terminate the path
    generation after the agent oscillated for 10 times. The method also handles collisions with walls by placing the agent near the wall on the track where it 
    collided or at the start state and setting its velocity to 0. The method checks if the agent is inside the boundary of the world and if the agent is in the 
    goal state. The method returns the path, status, distance to the goal state, and number of collisions.
    Args:
        start_state_index (int): index of start state
    Returns:
        tuple: path, status, distance to goal state, number of collisions
    """
    def create_path(self, start_state_index):
        policy = self.function
        goal_state_index = [index for index, terrain in self.S.items() if terrain == self.goal_state][0] # get index of a goal state
        goal_state = self.index_to_state[goal_state_index] # terrain of goal state 
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
                status = False
                break
            action = self.actions[action_index]
            random_num = np.random.uniform(0, 1)
            if random_num < self.transition['fail']:
                action = (0, 0) # no acceleration happens
            new_state = self.apply_kinematics(current_state, action)
            new_state_after_collision = self.handle_collision(current_state, new_state)
            if new_state_after_collision != new_state:
                collisions += 1 # collision with wall detected
            new_state = new_state_after_collision
            if self.inside_boundary(new_state):
                new_state_index = self.state_to_index[new_state]
                if new_state_index == prev_state_index:
                    count += 1  # self-loop detected
                    if count > 10:
                        status = False # policy generates self loop. Failed to reach goal state.
                        break
            else:
                status = False # policy generates path outside boundary. Failed to reach goal state.')
                break
            path.append((current_state, action, new_state)) # add tuple to path
            terrain = self.S[new_state_index]
        dist_to_goal = np.linalg.norm(np.array(new_state[:2]) - np.array(goal_state[:2]))
        return path, status, dist_to_goal, collisions

    """
    'predict' method is responsible for testing the optimal policy genetarated by an RL algorithm. The method initializes the agent at a start state and 
    creates a path for the agent using the optimal policy. The method calculates the path metrics and returns the path, path metrics, status, distance to the 
    goal state, and number of collisions.
    Returns:
        tuple: path, path metrics, status, distance to goal state, number of collisions
    """
    def predict(self):
        start_state_index = self.initialize_agent() # place agent at a start state
        path, status, dist_to_goal, collisions = self.create_path(start_state_index) # create path from optimal policy
        path_metrics = {'length': len(path), 'cost': 0, 'training_iters': self.training_iterations} # calculate path metrics
        return path, path_metrics, status, dist_to_goal, collisions
