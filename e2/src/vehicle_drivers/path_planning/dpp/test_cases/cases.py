from math import pi


class TestCase:
    """ Provide some test cases for a 10x10 map. """

    def __init__(self):
        
        self.start_pos = [4.6, 2.4, 0]
        self.end_pos = [30, 27, -pi/2]

        # self.start_pos = [4.6, 2.4, 0]
        # self.end_pos = [1.6, 8, -pi/2]

        # self.start_pos2 = [4, 4, 0]
        # self.end_pos2 = [4, 8, 1.2*pi]

        self.obs = [
            [2, 3, 6, 0.1],
            [2, 3, 0.1, 1.5],
            [4.3, 0, 0.1, 1.8],
            [6.5, 1.5, 0.1, 1.5],
            [0, 6, 3.5, 0.1],
            [5, 6, 5, 0.1]
        ]
        

    # 30 * 15
    # def __init__(self):
    #     self.lx = 30          # map length in meters
    #     self.ly = 15          # map width in meters

    #     self.start_pos = [3.0, 20.0, 0.0]        # starting position (x, y, theta)
    #     self.end_pos = [27.0, 14.0, -pi/2]       # ending position (x, y, theta)

    #     Sample obstacles:
    #     self.obs = [
    #         [5.0, 4.0, 6.0, 0.2],    # horizontal wall in the lower middle region
    #         [10.0, 0.0, 0.2, 7.0],   # vertical wall on the left area
    #         [15.0, 8.0, 10.0, 0.2],  # long horizontal wall across the middle
    #         [20.0, 5.0, 0.2, 5.0]    # vertical wall on the right side
    #     ]
    #     self.obs = [
    #         [2, 3, 6, 0.1], 
    #         [2, 3, 0.1, 1.5],
    #         [4.3, 0, 0.1, 1.8],
    #         [6.5, 1.5, 0.1, 1.5],
    #         [0, 6, 3.5, 0.1],
    #         [5, 6, 5, 0.1]
    #     ]