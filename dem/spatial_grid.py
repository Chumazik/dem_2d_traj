"""
Spatial hashing (uniform grid) for efficient contact detection in 2D DEM simulations.
Reduces pairwise contact detection from O(N²) to O(N) by only checking particles
in neighboring grid cells.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .particle import Particle


class SpatialGrid:
    """Uniform grid spatial hashing for 2D particle contact detection."""

    def __init__(self, cell_size: float):
        """
        Initialize spatial grid.
        
        Args:
            cell_size: Size of grid cells (should be ≈ 2× particle diameter)
        """
        self.cell_size = float(cell_size)
        self.grid: Dict[Tuple[int, int], List[Particle]] = {}
        self.particle_to_cell: Dict[int, Tuple[int, int]] = {}
        self.min_bounds = np.array([0.0, 0.0])
        self.max_bounds = np.array([0.0, 0.0])

    def update_bounds(self, particles: List[Particle]) -> None:
        """Update grid bounds based on particle positions."""
        if not particles:
            return
            
        positions = np.array([p.pos for p in particles])
        self.min_bounds = np.min(positions, axis=0) - self.cell_size
        self.max_bounds = np.max(positions, axis=0) + self.cell_size

    def clear(self) -> None:
        """Clear the grid."""
        self.grid.clear()
        self.particle_to_cell.clear()

    def add_particle(self, particle: Particle) -> None:
        """Add a particle to the grid."""
        cell = self._get_cell(particle.pos)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(particle)
        self.particle_to_cell[particle.id] = cell

    def _get_cell(self, position: np.ndarray) -> Tuple[int, int]:
        """Get grid cell coordinates for a position."""
        # Ensure position is within bounds
        pos = np.maximum(position, self.min_bounds)
        pos = np.minimum(pos, self.max_bounds)
        
        cell_x = int((pos[0] - self.min_bounds[0]) // self.cell_size)
        cell_y = int((pos[1] - self.min_bounds[1]) // self.cell_size)
        return (cell_x, cell_y)

    def get_neighboring_particles(self, particle: Particle) -> List[Particle]:
        """Get all particles in neighboring cells (including own cell)."""
        cell = self.particle_to_cell.get(particle.id)
        if cell is None:
            return []
            
        cell_x, cell_y = cell
        neighbors = []
        
        # Check all 9 neighboring cells (including own cell)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                if neighbor_cell in self.grid:
                    neighbors.extend(self.grid[neighbor_cell])
        
        return neighbors

    def update_grid(self, particles: List[Particle]) -> None:
        """Update the grid with current particle positions."""
        self.clear()
        self.update_bounds(particles)
        for particle in particles:
            self.add_particle(particle)


def create_spatial_grid(particles: List[Particle], particle_radius: float) -> SpatialGrid:
    """
    Create a spatial grid optimized for the given particles.
    
    Args:
        particles: List of particles
        particle_radius: Radius of particles (used to determine cell size)
        
    Returns:
        SpatialGrid instance
    """
    # Cell size ≈ 2× particle diameter (good balance between cell size and neighbor checks)
    cell_size = 2.2 * (2 * particle_radius)
    grid = SpatialGrid(cell_size)
    grid.update_grid(particles)
    return grid