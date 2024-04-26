import copy
import os

class PathVisualizer:
    def __init__(self, model, image_path):
        self.world = model.world # get the environment
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
        # mark the path taken by the agent
        for triplet in path:
            state = triplet[0][0:2] # current state in form (x, y)
            velocity = triplet[0][2:] # current velocity in form (vx, vy)
            velocity = self.compass_heading(velocity) # convert action to compass heading
            grid[state[1]][state[0]] = velocity
        # plot final state
        if len(path) != 0:
            state = path[-1][2][0:2] # final state in form (x, y)
            if grid[state[1]][state[0]] == '🏁':
                grid[state[1]][state[0]] = '🟢'
            else:
                grid[state[1]][state[0]] = '🔴'
            # velocity = path[-1][2][2:]  # final velocity in form (vx, vy)
            # velocity = self.compass_heading(velocity) # convert velocity to compass heading
            # grid[state[1]][state[0]] = velocity
        grid_string = '\n\n'
        for row in grid:
            grid_string += '    ' + ''.join(row) + '\n'
            # print(''.join(row))
        # export to txt file for visualization
        with open(self.image_path, 'w') as f:
            f.write(grid_string)
        return None