
"""
This module contains the BresenhamLine class which is used to draw a line between two points using the Bresenham line algorithm
for rasterification of linear lines through a series of discrete pixels on a 2D grid. The following code implementation has been 
borrowed from an outside source found at https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm.

The BresenhamLine class contains the following attributes:
    - start: the starting point of the line
    - end: the ending point of the line

The BresenhamLine class contains the following methods:
    - draw_low_slope_line: draw a line with a low slope
    - draw_high_slope_line: draw a line with a high slope
    - draw_line: draw the line between the start and end points using the Bresenham line algorithm  
"""

class BresenhamLine():
    def __init__(self, start, end):
        self.start = start
        self.end = end

    """
    'draw_low_slope_line' method is responsible for drawing a line with a high slope smaller than 1
    Args:
        x0 (int): x-coordinate of the starting point
        y0 (int): y-coordinate of the starting point
        x1 (int): x-coordinate of the ending point
        y1 (int): y-coordinate of the ending point
    Returns:
        points (list): list of points that make up the line
    """
    def draw_low_slope_line(self, x0, y0, x1, y1):
        points = []
        dx = x1 - x0 # total change in x
        dy = y1 - y0 # total change in y
        yi = 1 # increment for y
        # if slope is negative, increment y in negative direction
        if dy < 0:
            yi = -1
            dy = -dy
        D = (2 * dy) - dx # initial decision parameter
        y = y0
        for x in range(x0, x1 + 1):
            points.append((x, y))
            # if decision parameter is positive, increment y
            if D > 0:
                y = y + yi
                D = D + (2 * (dy - dx))
            else:
                D = D + 2*dy
        return points
    
    """
    'draw_high_slope_line' method is responsible for drawing a line with a high slope greater than 1
    Args:
        x0 (int): x-coordinate of the starting point
        y0 (int): y-coordinate of the starting point
        x1 (int): x-coordinate of the ending point
        y1 (int): y-coordinate of the ending point
    Returns:
        points (list): list of points that make up the line
    """
    def draw_high_slope_line(self, x0, y0, x1, y1):
        points = []
        dx = x1 - x0 # total change in x
        dy = y1 - y0 # total change in y
        xi = 1 # increment for x
        # if slope is negative, increment x in negative direction
        if dx < 0:
            xi = -1
            dx = -dx
        D = (2 * dx) - dy # initial decision parameter
        x = x0
        for y in range(y0, y1 + 1):
            points.append((x, y))
            # if decision parameter is positive, increment x
            if D > 0:
                x = x + xi
                D = D + (2 * (dx - dy))
            else:
                D = D + 2*dx
        return points

    """
    'draw_line' method is responsible for drawing the line between the start and end points using the Bresenham line algorithm
    Returns:
        points (list): list of points that make up the line
    """
    def draw_line(self):
        x0, y0 = self.start
        x1, y1 = self.end
        if abs(y1 - y0) < abs(x1 - x0):
            if x0 > x1:
                points = self.draw_low_slope_line(x1, y1, x0, y0)
            else:
                points = self.draw_low_slope_line(x0, y0, x1, y1)
        else:
            if y0 > y1:
                points = self.draw_high_slope_line(x1, y1, x0, y0)
            else:
                points = self.draw_high_slope_line(x0, y0, x1, y1)
        return points