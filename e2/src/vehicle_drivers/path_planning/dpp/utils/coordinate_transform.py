#!/usr/bin/env python3
import numpy as np
from math import radians, cos, sin

class CoordinateTransform:
    def __init__(self, center_lat=40.0928174, center_lon=-88.2356714, grid_size=80.0):
        """
        Initialize coordinate transformer
        Args:
            center_lat: Center latitude of the grid
            center_lon: Center longitude of the grid
            grid_size: Size of the grid in meters
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.grid_size = grid_size
        self.half_grid = grid_size / 2.0
        
        # Earth's radius in meters
        self.EARTH_RADIUS = 6371000.0
        
        # Calculate meters per degree at the center latitude
        self.meters_per_deg_lat = self.EARTH_RADIUS * np.pi / 180.0
        self.meters_per_deg_lon = self.meters_per_deg_lat * cos(radians(center_lat))
        
    def gps_to_local(self, lat, lon):
        """
        Convert GPS coordinates to local grid coordinates
        Args:
            lat: Latitude
            lon: Longitude
        Returns:
            (x, y) in local grid coordinates (meters from center)
        """
        # Convert to meters from center
        x = (lon - self.center_lon) * self.meters_per_deg_lon
        y = (lat - self.center_lat) * self.meters_per_deg_lat
        
        return x, y
        
    def local_to_gps(self, x, y):
        """
        Convert local grid coordinates to GPS coordinates
        Args:
            x: X coordinate in meters from center
            y: Y coordinate in meters from center
        Returns:
            (lat, lon) in GPS coordinates
        """
        lon = self.center_lon + (x / self.meters_per_deg_lon)
        lat = self.center_lat + (y / self.meters_per_deg_lat)
        
        return lat, lon
        
    def is_in_grid(self, x, y):
        """
        Check if a point is within the grid boundaries
        Args:
            x: X coordinate in meters from center
            y: Y coordinate in meters from center
        Returns:
            bool: True if point is in grid
        """
        return (abs(x) <= self.half_grid and 
                abs(y) <= self.half_grid)
                
    def gps_to_grid_coordinates(self, lat, lon):
        """
        Convert GPS coordinates to grid coordinates (0 to grid_size)
        Args:
            lat: Latitude
            lon: Longitude
        Returns:
            (x, y) in grid coordinates (0 to grid_size)
        """
        x, y = self.gps_to_local(lat, lon)
        # Convert to grid coordinates (0 to grid_size)
        grid_x = x + self.half_grid
        grid_y = y + self.half_grid
        return grid_x, grid_y
        
    def grid_to_gps_coordinates(self, grid_x, grid_y):
        """
        Convert grid coordinates to GPS coordinates
        Args:
            grid_x: X coordinate in grid (0 to grid_size)
            grid_y: Y coordinate in grid (0 to grid_size)
        Returns:
            (lat, lon) in GPS coordinates
        """
        # Convert from grid coordinates to local coordinates
        x = grid_x - self.half_grid
        y = grid_y - self.half_grid
        return self.local_to_gps(x, y)
        
    def validate_grid_point(self, x, y):
        """
        Validate if a grid point is within bounds
        Args:
            x: X coordinate in grid (0 to grid_size)
            y: Y coordinate in grid (0 to grid_size)
        Returns:
            bool: True if point is valid
        """
        return (0 <= x <= self.grid_size and 
                0 <= y <= self.grid_size) 