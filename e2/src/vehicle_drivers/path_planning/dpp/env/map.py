class Map:
    def __init__(self):
        self.obs = [[0 ,25.5, 110, 0.1],[94,0,0.1,30],[12,0,0.1,30],[0 ,5, 110, 0.1]] #Highbay borders
        self.grid_top_left = (-25, 10)
        self.grid_bottom_right = (85, -20)
        self.lx = self.grid_bottom_right[0] - self.grid_top_left[0]  # 110m
        self.ly = self.grid_top_left[1] - self.grid_bottom_right[1]  # 30m
        self.cell_size = max(0.25, self.lx / 200)
        
    
    def add_walls(self):
        #self.obs.append([60,10,0.1,10]) 
        # spot is 5 m * 2.7432m
        # 2.7432 from driver to entrance
        # 1.3716 from car center to left wall
        spot = (49.5948455962297, 12.416345388704688)
     
        # self.obs.append([spot[0]+1.3716,spot[1]-(5-2.7432),0.1,5]) 
        # self.obs.append([spot[0]+1.3716-2.7432,spot[1]-(5-2.7432),0.1,5])
        # self.obs.append([spot[0]+1.3716-2.7432,spot[1]-(5-2.7432),2.7432,0.1]) 
        
        self.obs.append([spot[0]+2,spot[1]-3,0.1,6]) # real spot
        self.obs.append([spot[0]-2,spot[1]-3,0.1,6])
        self.obs.append([spot[0]-2,spot[1]-3,4,0.1]) 
        self.obs.append([spot[0]+2+4,spot[1]-3,0.1,6])  # dummy spot
        self.obs.append([spot[0]-2+4,spot[1]-3,0.1,6])
        self.obs.append([spot[0]-2+4,spot[1]-3,4,0.1]) 
        self.obs.append([spot[0]+2+8,spot[1]-3,0.1,6]) # dummy spot
        self.obs.append([spot[0]-2+8,spot[1]-3,0.1,6])
        self.obs.append([spot[0]-2+8,spot[1]-3,4,0.1]) 