import numpy as np
import os
import uuid

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class VoxelizerFactory(object):
    def __init__(self, voxelizer, output_shape, save_voxels_to=None):
        if not isinstance(voxelizer, str) or\
               voxelizer not in ["tsdf_grid", "occupancy_grid"]:
                    raise AttributeError("The voxelizer is invalid")
        if not isinstance(output_shape, np.ndarray):
            raise ValueError(
                "Output_shape should be a np.ndarray, but is %r"
                % (output_shape,)
            )

        self._voxelizer = voxelizer
        # Arguments for teh OccupancyGrid
        self.output_shape = output_shape
        self._save_voxels_to = save_voxels_to

    @property
    def voxelizer(self):
        if self._voxelizer == "tsdf_grid":
            return TSDFPrecomputed(self._voxelizer, self.output_shape)
        elif self._voxelizer == "occupancy_grid":
            return OccupancyGrid(self._voxelizer, self.output_shape, None,
                                 self._save_voxels_to)  # TODO: Remove save_voxels_to


class VoxelizerBase(object):
    def __init__(self, name):
        self.name = name


class OccupancyGrid(VoxelizerBase):
    """OccupancyGrid class is a wrapper for the occupancy grid.
    """
    def __init__(self, name, output_shape, bbox=None, save_voxels_to=None):
        super(OccupancyGrid, self).__init__(name)

        self._output_shape = output_shape

        # Array that contains the voxel centers
        self._voxel_grid = None
        self.save_voxels_to = save_voxels_to

    @property
    def output_shape(self):
        return ((1,) + tuple(self._output_shape))

    def _bbox_from_points(self, pts):
        bbox = np.array([-0.51]*3 + [0.51]*3)
        step = (bbox[3:] - bbox[:3])/self._output_shape

        return bbox, step

    def voxel_grid(self, pts):
        if self._voxel_grid is None:
            # Get the bounding box from the points
            bbox, _ = self._bbox_from_points(pts)

            self._voxel_grid = get_voxel_grid(
                bbox.reshape(-1, 1),
                self.output_shape
            ).astype(np.float32)
        return self._voxel_grid

    def voxelize(self, mesh):
        """Given a Mesh object we want to estimate the occupancy grid
        """
        pts = mesh.points
        # Make sure that the points have the correct shape
        assert pts.shape[0] == 3
        # Make sure that points lie in the unit cube
        if (np.any(np.abs(pts.min(axis=-1)) > 0.51) or
                np.any(pts.max(axis=-1) > 0.51)):
            raise Exception(
                "The points do not lie in the unit cube min:%r - max:%r"
                % (pts.min(axis=-1), pts.max(axis=-1))
            )

        bbox, step = self._bbox_from_points(pts)
        occupancy_grid = np.zeros(tuple(self._output_shape), dtype=np.float32)

        idxs = ((pts.T - bbox[:3].T)/step).astype(np.int32)
        # Array that will contain the occupancy grid, namely if a point lies
        # within a voxel then we assign one to this voxel, otherwise we assign
        # 0
        occupancy_grid[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = 1.0
        if self.save_voxels_to is not None:
            unique_filename = str(uuid.uuid4())
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.voxels(occupancy_grid, edgecolor='k')
            ax.view_init(elev=10, azim=80)
            plt.savefig(
                os.path.join(self.save_voxels_to, unique_filename+".png")
            )
            plt.close()
        return occupancy_grid[np.newaxis]

    def get_occupied_voxel_centers(self, mesh):
        occ = self.voxelize(mesh)
        voxel_grid = self.voxel_grid(mesh.points)

        return voxel_grid[:, occ == 1]

    def get_X(self, model):
        gt_mesh = model.groundtruth_mesh
        return self.voxelize(gt_mesh)


def get_voxel_grid(bbox, grid_shape):
    """Given a bounding box and the dimensionality of a grid generate a grid of
    voxels and return their centers.

    Arguments:
    ----------
        bbox: array(shape=(6, 1), dtype=np.float32)
              The min and max of the corners of the bbox that encloses the
              scene
        grid_shape: array(shape(3,), dtype=int32)
                    The dimensions of the voxel grid used to discretize the
                    scene
    Returns:
    --------
        voxel_grid: array(shape=(3,)+grid_shape)
                    The centers of the voxels
    """
    # Make sure that we have the appropriate inputs
    assert bbox.shape[0] == 6
    assert bbox.shape[1] == 1

    xyz = [
        np.linspace(s, e, c, endpoint=True, dtype=np.float32)
        for s, e, c in
        zip(bbox[:3], bbox[3:], grid_shape)
    ]
    bin_size = np.array([xyzi[1]-xyzi[0] for xyzi in xyz]).reshape(3, 1, 1, 1)
    return np.stack(np.meshgrid(*xyz, indexing="ij")) + bin_size/2
