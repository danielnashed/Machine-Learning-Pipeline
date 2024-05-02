import copy
import os

class PathVisualizer:
    def __init__(self, model, image_path, collisions, average_metrics):
        self.model = model
        self.world = model.world # get the environment
        self.engine = model.engine # get the engine name
        self.visit_history = model.visit_history
        self.image_path = image_path # get the path to save the image
        self.collisions = collisions # collisions for this race
        self.average_metrics = average_metrics # average over all races

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

    def compass_heading(self, velocity):
        if velocity[0] < 0 and velocity[1] < 0:
            return '↖️'
        elif velocity[0] < 0 and velocity[1] == 0:
            return '⬅️'
        elif velocity[0] < 0 and velocity[1] > 0:
            return '↙️'
        elif velocity[0] == 0 and velocity[1] < 0:
            return '⬆️'
        elif velocity[0] == 0 and velocity[1] > 0:
            return '⬇️'
        elif velocity[0] > 0 and velocity[1] < 0:
            return '↗️'
        elif velocity[0] > 0 and velocity[1] == 0:
            return '➡️'
        elif velocity[0] > 0 and velocity[1] > 0:
            return '↘️'
        else:
            return '⏹️'

    def visualize_path(self, path): 
        grid = copy.deepcopy(self.world) # copy the world grid
        grid = self.pretty_grid(grid) # convert the grid to emojis for visualization
        average_x_vel = 0   
        average_y_vel = 0
        average_speed = 0
        average_x_accel = 0   
        average_y_accel = 0

        # mark the path taken by the agent
        for triplet in path:
            state = triplet[0][0:2] # current state in form (x, y)
            velocity = triplet[0][2:] # current velocity in form (vx, vy)
            acceleration = triplet[1]
            average_x_vel += velocity[0] # accumulate x velocity
            average_y_vel += velocity[1] # accumulate y velocity
            average_x_accel +=  acceleration[0] # accumulate x acceleration
            average_y_accel += acceleration[1] # accumulate y acceleration
            velocity = self.compass_heading(velocity) # convert action to compass heading
            grid[state[1]][state[0]] = velocity

        # plot final state
        if len(path) != 0:
            state = path[-1][2][0:2] # final state in form (x, y)
            if grid[state[1]][state[0]] == '🏁':
                grid[state[1]][state[0]] = '🟢'
            else:
                grid[state[1]][state[0]] = '🔴'

        # stats: average x and y velocities
        if len(path) != 0:
            average_x_vel = average_x_vel / len(path)
            average_y_vel = average_y_vel / len(path)
            average_speed = (average_x_vel ** 2 + average_y_vel ** 2) ** 0.5
            average_x_accel = average_x_accel / len(path)
            average_y_accel = average_y_accel / len(path)

        # add the outcome of the path (success or failure)
        if grid[state[1]][state[0]] == '🟢':
            outcome = 'Goal Reached'
        if grid[state[1]][state[0]] == '🔴':
            outcome = 'Fail'

        # add header to file 
        grid_string = '\n                                           ' + self.engine.capitalize() + ': ' + outcome + '\n'
        grid_string += '\n        ' + '🟥: Start, 🏁: Goal, ⬛: Obstacle, ⬜: Empty, 🟢: Reached Goal, 🔴: Crashed\n\n'
        for row in grid:
            grid_string += '    ' + ''.join(row) + '\n'

        # add stats for this race
        grid_string += '\n\n        STATS FOR THIS RACE: ' + '\n'
        grid_string += '        Number of steps:        ' + str(len(path)) + '\n'
        grid_string += '        Number of collisions:   ' + str(self.collisions) + '\n'
        grid_string += '        Average Speed:          ' + str(round(average_speed, 3)) + '\n'
        grid_string += '        Average X Velocity:     ' + str(round(average_x_vel, 3)) + '\n'
        grid_string += '        Average Y Velocity:     ' + str(round(average_y_vel, 3)) + '\n'
        grid_string += '        Average X Acceleration: ' + str(round(average_x_accel, 3)) + '\n'
        grid_string += '        Average Y Acceleration: ' + str(round(average_y_accel, 3)) + '\n'

        # add stats for average over all 5000 races 
        grid_string += '\n\n        STATS FOR 5000 RACES: ' + '\n'
        grid_string += '        Success rate to reach goal: ' + str(round(self.average_metrics['success_rate'], 3)) + '%\n'
        grid_string += '        Avg number of steps:        ' + str(int(self.average_metrics['average_steps'])) + '\n'
        grid_string += '        Avg number of collisions:   ' + str(int(self.average_metrics['average_collisions'])) + '\n'
        grid_string += '        Avg speed:                  ' + str(round(self.average_metrics['average_speed'], 3)) + '\n'

        # add stats for average over all 5000 races 
        grid_string += '\n\n        POLICY STATS & PARAMETERS: ' + '\n'
        grid_string += '        Number of training iterations: ' + str(self.model.training_iterations) + '\n'
        grid_string += '        Crash mode: ' + str(self.model.crash_algorithm) + '\n'
        grid_string += '        Reward: ' + str(self.model.reward) + '\n'
        grid_string += '        Discount factor: ' + str(self.model.gamma) + '\n'
        grid_string += '        Initial learning rate: ' + str(round(self.model.alpha, 3)) + '\n'
        grid_string += '        Final learning rate: ' + str(round(self.model.final_alpha, 3)) + '\n'
        grid_string += '        Initial greedy epsilon: ' + str(round(self.model.initial_greedy_epsilon, 3)) + '\n'
        grid_string += '        Final greedy epsilon: ' + str(round(self.model.final_greedy_epsilon, 3)) + '\n'
        grid_string += '        %% of state-action pairs not visited: ' + str(round(self.model.final_not_visited, 2)) + '%\n'

        # export to txt file for visualization
        with open(self.image_path, 'w') as f:
            f.write(grid_string)
        return None