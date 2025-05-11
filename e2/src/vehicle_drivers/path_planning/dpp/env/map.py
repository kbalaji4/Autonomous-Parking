class Map:
    def __init__(self):
        self.obs = [[0 ,25.5, 110, 0.1],[94,0,0.1,30],[12,0,0.1,30],[0 ,5, 110, 0.1]] #Highbay borders
        self.grid_top_left = (-25, 10)
        self.grid_bottom_right = (85, -20)
        self.lx = self.grid_bottom_right[0] - self.grid_top_left[0]  # 110m
        self.ly = self.grid_top_left[1] - self.grid_bottom_right[1]  # 30m
        self.cell_size = max(0.25, self.lx / 200)
        self.cones = []  # list to store cone positions
        
    
    def add_walls(self):
        #self.obs.append([60,10,0.1,10]) 
        # spot is 5 m * 2.7432m
        # 2.7432 from driver to entrance
        # 1.3716 from car center to left wall
        spot = (49.5948455962297, 12.416345388704688)
     
        # self.obs.append([spot[0]+1.3716,spot[1]-(5-2.7432),0.1,5]) 
        # self.obs.append([spot[0]+1.3716-2.7432,spot[1]-(5-2.7432),0.1,5])
        # self.obs.append([spot[0]+1.3716-2.7432,spot[1]-(5-2.7432),2.7432,0.1]) 
        
        # self.obs.append([spot[0]+2,spot[1]-3,0.1,6]) # real spot
        # self.obs.append([spot[0]-2,spot[1]-3,0.1,6])
        # self.obs.append([spot[0]-2,spot[1]-3,4,0.1]) 

        self.obs.append([spot[0]+2-0.25,spot[1]-3,0.1,7]) # real spot
        self.obs.append([spot[0]-2+0.25,spot[1]-3,0.1,7])
        self.obs.append([spot[0]-2+0.25,spot[1]-3,3.5,0.1]) 
        
        
        for i in range(1,7):
            self.obs.append([spot[0]+2-(3.5*i)-0.25,spot[1]-3,0.1,7]) # dummy spot
            self.obs.append([spot[0]-2-(3.5*i)+0.25,spot[1]-3,0.1,7])
            self.obs.append([spot[0]-2-(3.5*i)+0.25,spot[1]-3,3.5,0.1]) 
            self.obs.append([spot[0]+2+(3.5*i)-0.25,spot[1]-3,0.1,7])  # dummy spot
            self.obs.append([spot[0]-2+(3.5*i)+0.25,spot[1]-3,0.1,7])
            self.obs.append([spot[0]-2+(3.5*i)+0.25,spot[1]-3,3.5,0.1]) 

    def add_cone(self, x, y):
        """Add a cone as a 0.5x0.5m obstacle at the specified position"""
        # Check if there's already a cone nearby
        for cone in self.cones:
            if ((cone[0] - x) ** 2 + (cone[1] - y) ** 2) ** 0.5 < 0.5:
                return False  # Cone already exists nearby
        
        # Add new cone
        self.cones.append((x, y))
        print(f"Added cone at ({x}, {y})")
        self.obs.append((x, y, 0.5, 0.5))  # Center the 0.5x0.5m obstacle at (x,y)
        print(len(self.obs))
        return True

    def clear_cones(self):
        """Remove all cones from the map"""
        # Remove cone obstacles from obs list
        self.obs = [ob for ob in self.obs if ob not in [(x - 0.25, y - 0.25, 0.5, 0.5) for x, y in self.cones]]
        self.cones = [] 