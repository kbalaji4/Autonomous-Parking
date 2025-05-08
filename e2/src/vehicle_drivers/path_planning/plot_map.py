import sys
import os

scripts_dir = os.path.dirname(__file__)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
    
from e2.src.vehicle_drivers.path_planning.dpp.env.map import Map

class PlotMap(object):
    def __init__(self):
        self.map = Map()
        