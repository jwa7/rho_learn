"""
Module containing the RhoCube class, which wraps the cube_tools.cube class from
https://github.com/funkymunkycool/Cube-Toolz. Allows reading and manipulation of
cube files, with added functionality for generating STM images.
"""
import ase
import numpy as np

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