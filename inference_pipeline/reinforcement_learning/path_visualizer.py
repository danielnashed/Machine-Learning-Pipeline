import copy
import os

class PathVisualizer:
    def __init__(self, model, image_path):
        self.world = model.world # get the environment
        self.engine = model.engine # get the engine name
        self.image_path = image_path # get the path to save the image

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
        # mark the path taken by the agent
        for triplet in path:
            state = triplet[0][0:2] # current state in form (x, y)
            velocity = triplet[0][2:] # current velocity in form (vx, vy)
            average_x_vel += velocity[0] # accumulate x velocity
            average_y_vel += velocity[1] # accumulate y velocity
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
        else:
            average_x_vel = 0
            average_y_vel = 0
            average_speed = 0

            # velocity = path[-1][2][2:]  # final velocity in form (vx, vy)
            # velocity = self.compass_heading(velocity) # convert velocity to compass heading
            # grid[state[1]][state[0]] = velocity


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
        # add stats to end of file 
        grid_string += '\n\n        Number of steps: ' + str(len(path)) + '\n'
        grid_string += '        Average X Velocity: ' + str(round(average_x_vel, 3)) + '\n'
        grid_string += '        Average Y Velocity: ' + str(round(average_y_vel, 3)) + '\n'
        grid_string += '        Average Speed:      ' + str(round(average_speed, 3)) + '\n'

        # export to txt file for visualization
        with open(self.image_path, 'w') as f:
            f.write(grid_string)
        return None