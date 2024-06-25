"""
Module containing the RhoCube class, which wraps the cube_tools.cube class from
https://github.com/funkymunkycool/Cube-Toolz. Allows reading and manipulation of
cube files, with added functionality for generating STM images.
"""

from typing import List

import ase
import numpy as np
from scipy.interpolate import CubicSpline

import cube_tools
from cube_tools import cube


class RhoCube(cube_tools.cube):

    def __init__(self, filename):
        super(RhoCube, self).__init__(filename)
        self.ase_frame = self.ase_frame()

    def ase_frame(self) -> ase.Atoms:
        """
        Builds an ASE atoms object from the atomic positions and chemical
        symbols in the cube file.
        """
        return ase.Atoms(
            symbols=np.array(self.atoms, dtype=int), positions=np.array(self.atomsXYZ)
        )

    def get_slab_slice(
        self, axis: int = 2, center_coord: float = None, thickness: float = 1.0
    ) -> np.ndarray:
        """
        Get a 2D grid of the cube data, sliced at the specified center
        coordinate of the specified axis (i.e. 0 =
        x, 1 = y, 2 = z), and summed over the thickness of the slab.

        For instance, passing `axis=2`, `center_coord=7.5`, and `thickness=1.0`,
        a 2D grid of cube data will be returned by summing over XY arrays of Z
        coordinates that are in the range 7.0 to 8.0.
        """
        if axis not in [0, 1, 2]:
            raise ValueError("Invalid axis")
        if not (
            (self.X[1] == self.X[2] == 0)
            and (self.Y[0] == self.Y[2] == 0)
            and (self.Z[0] == self.Z[1] == 0)
        ):
            raise ValueError("Can only handle X, Y, Z axis aligned cubes")

        # Define the min and max Z coordinate of the surface slab to be summed over
        slab_min = center_coord - (thickness / 2)
        slab_max = center_coord + (thickness / 2)

        keep_axes = [i for i in range(3) if i != axis]
        cube_sizes = [self.X, self.Y, self.Z]
        n_cubes = [self.NX, self.NY, self.NZ]

        # Return grid coordinates of the axes that are kept
        axis_a_coords = (
            np.arange(n_cubes[keep_axes[0]]) * cube_sizes[keep_axes[0]][keep_axes[0]]
        ) + self.origin[keep_axes[0]]
        axis_b_coords = (
            np.arange(n_cubes[keep_axes[1]]) * cube_sizes[keep_axes[1]][keep_axes[1]]
        ) + self.origin[keep_axes[0]]

        # Initialize the 2D grid of cube data to return
        axis_c_vals = np.zeros((n_cubes[keep_axes[0]], n_cubes[keep_axes[1]]))
        for i_cube in range(n_cubes[axis]):
            coord_of_curr_slice = i_cube * cube_sizes[axis][axis] + self.origin[axis]

            # If within the Z coords of the slab, accumulate density of this XY slice
            if slab_min <= coord_of_curr_slice <= slab_max:
                if axis == 0:
                    axis_c_vals += self.data[i_cube, :, :]
                elif axis == 1:
                    axis_c_vals += self.data[:, i_cube, :]
                else:
                    assert axis == 2
                    axis_c_vals += self.data[:, :, i_cube]

        return axis_a_coords, axis_b_coords, axis_c_vals

    def get_height_profile_map(
        self,
        isovalue: float,
        tolerance: float,
        grid_multiplier: int,
        z_min: float = None,
        z_max: float = None,
        xy_tiling: List[int] = None,
    ) -> np.ndarray:
        """
        Calculates the height profile of the density in `self.data` at the target
        `isovalue`, within the specified `tolerance`.

        For each XY point, the Z-coordinates of the cube array are densified by a factor
        of `grid_multiplier`, then interpolated with splines to find the Z-coordinates
        that match the taregt `isovalue` within the specified `tolerance`.

        The intended use of this function is for visualizing slab surface densities. As
        such, the surface is assumed to be aligned at Z=0, with the slab at negative
        Z-coordinates. Any Z-coordinate that is outside the range of `min_z` and
        `max_z` is ignored and returned as NaN.
        """

        height_map = np.ones((self.NX, self.NY)) * np.nan
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):

                # Spline this 1D array of values, build a cubic spline
                z_grid = (np.arange(self.NZ) * self.Z[2]) + self.origin[2]
                spliner = CubicSpline(z_grid, self.data[i, j, :])

                # Build a finer grid for interpolation
                z_grid_fine = np.linspace(
                    z_grid.min(), z_grid.max(), self.NZ * grid_multiplier
                )

                # Find the idxs of the values that match the isovalue, within the tolerance
                match_idxs = np.where(
                    np.abs(spliner(z_grid_fine) - isovalue) < tolerance
                )[0]
                if len(match_idxs) == 0:
                    continue

                # Find the idx of the matching value with the greatest z-coordinate
                match_idx = match_idxs[np.argmax(z_grid_fine[match_idxs])]
                z_height = z_grid_fine[match_idx]
                if (z_min is not None and z_height < z_min) or (
                    z_max is not None and z_height > z_max
                ):
                    continue
                height_map[i, j] = z_grid_fine[match_idx]

        if xy_tiling is None:
            xy_tiling = [1, 1]
        assert len(xy_tiling) == 2

        X = (np.arange(self.NX * xy_tiling[0]) * self.X[0]) + self.origin[0]
        Y = (np.arange(self.NY * xy_tiling[1]) * self.Y[1]) + self.origin[1]
        Z = np.tile(height_map, reps=xy_tiling)

        return X, Y, Z.T
