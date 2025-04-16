import numpy as np
from collections import defaultdict
from constants import WALL_RESOLUTION

"""
Walls class. Just inserts configurations of hardcoded "walls" on the map
Plotting, etc all done in other functions 

4/12/25 empirical testing seems to work on large spaces where it's just 
horizontal/vertical walls.

diagonal walls untested
if car spawns on a wall, that behavior is undefined
"""

class Walls:
    def __init__(self):
        """ 
        walls: 
        (x, y) coords that we want to avoid. no cost right now
        we treat this like closed set (visited)
        """
        self.walls = {} # (x1, y1, x2, y2):Wall. assume unique, duplicates undefined
        self.occupied_spaces = defaultdict(int) # occupied coordinates with count
    def add_wall(self, x1, y1, x2, y2, offset_x, offset_y, resolution=WALL_RESOLUTION):
        """
        adds a Wall to self.walls and increments counter of (x, y)'s this Wall occupies
        in self.occupied_spaces
        """
        # add Wall obj to walls list
        wall = Wall(x1, y1, x2, y2, offset_x, offset_y, resolution)
        self.walls[(x1, y1, x2, y2)] = wall

        # add Wall's occupied spaces to total occupied spaces
        for point in wall.points:
            self.occupied_spaces[point] += 1
    def remove_wall(self, x1, y1, x2, y2):
        if (x1, y1, x2, y2) in self.walls:
            # remove occupied_spaces counter
            wall = self.walls[(x1, y1, x2, y2)]
            for point in wall.points:
                self.occupied_spaces[point] -= 1
                # this can just go negative lol
            del self.walls[(x1, y1, x2, y2)]
        else:
            print(f"Could not find wall with start and end: {x1, y1, x2, y2}")
    def is_occupied(self, x, y):
        if (round(x), round(y)) in self.occupied_spaces and self.occupied_spaces[(round(x), round(y))] > 0:
            return True
        return False
    def intersects_path():
        """ 
        if each Wall could check, then we just check all 
        """
        pass

class Wall:
    def __init__(self, x1, y1, x2, y2, offset_x, offset_y, resolution=WALL_RESOLUTION):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.offset_x, self.offset_y = offset_x, offset_y
        self.resolution = resolution
        # convenience with plotting for xpoints, ypoints so no offset
        # points will have actual offset tho so we can have occupied_spaces
        self.points, self.xpoints, self.ypoints = self._discretize_wall()
        
    
    def _discretize_wall(self):
        """ 
        use x1y1, x2y2 and draw the line
        returns: points (dict), xpoints, ypoints (np arrays)
        """
        length = np.hypot(self.x2 - self.x1, self.y2 - self.y1) # L2

        num_points = int(np.ceil(length/self.resolution))

        t = np.linspace(0, 1, num_points + 1) # because of the 0. now it partitions nicely
        xpoints = self.x1 + t * (self.x2 - self.x1)
        ypoints = self.y1 + t * (self.y2 - self.y1)

        xpoints_offset = (self.x1 + self.offset_x) + t * (self.x2 - self.x1) # only difference is needed
        ypoints_offset = (self.y1 + self.offset_y) + t * (self.y2 - self.y1)


        return {(round(x), round(y)) for x, y in zip(xpoints_offset, ypoints_offset)}, np.round(xpoints), np.round(ypoints)
    def intersects_segment(self):
        """ 
        check if line segments intersect. Need to figure out how to 
        check intersection with planner "line" 
        """
        pass