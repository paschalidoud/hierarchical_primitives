#!/usr/bin/env python3
"""Script used for exporting occupancy pairs for training
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from hierarchical_primitives.common.model_factory import DatasetBuilder
from hierarchical_primitives.common.base import splits_factory
from hierarchical_primitives.external.libmesh import check_mesh_contains
from hierarchical_primitives.mesh import Trimesh
from hierarchical_primitives.utils.progbar import Progbar


def export_surface_pairs(path_to_mesh_file, N, normalize):
    m = Trimesh(path_to_mesh_file, normalize=normalize)
    P = m.sample_faces(N)
    points = P[:, :3]
    normals = P[:, 3:]

    s = np.random.randn(N, 1) * 0.01
    points_hat = points + s * normals
    labels = check_mesh_contains(m.mesh, points_hat)

    return points_hat, labels


def export_volume_pairs(path_to_mesh_file, N, normalize):
    m = Trimesh(path_to_mesh_file, normalize=normalize)
    random_points = np.random.rand(N, 3) - 0.5
    labels = check_mesh_contains(m.mesh, random_points.reshape(-1, 3))

    return random_points, labels


def export_occupancy_pairs(occupancy_type):
    return {
       "surface": export_surface_pairs,
       "volume": export_volume_pairs,
    }[occupancy_type]


def occupancy_pairs_subdir(occupancy_type, normalize_mesh):
    subdir = {
       "surface": "surface_points_seq",
       "volume": "points_seq"
    }[occupancy_type]
    return subdir


def ensure_parent_directory_exists(filepath):
    try:
        os.mkdir(os.path.dirname(filepath))
    except FileExistsError:
        pass


class DirLock(object):
    def __init__(self, dirpath):
        self._dirpath = dirpath
        self._acquired = False

    @property
    def is_acquired(self):
        return self._acquired

    def acquire(self):
        if self._acquired:
            return
        try:
            os.mkdir(self._dirpath)
            self._acquired = True
        except FileExistsError:
            pass

    def release(self):
        if not self._acquired:
            return
        try:
            os.rmdir(self._dirpath)
            self._acquired = False
        except FileNotFoundError:
            self._acquired = False
        except OSError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def main(argv):
    parser = argparse.ArgumentParser(
        description="Export occupancy pairs for training"
    )
    parser.add_argument(
        "dataset_directory",
        help="Path to the directory containing the dataset"
    )
    parser.add_argument(
        "train_test_splits_file",
        help="Path to the train-test splits file"
    )
    parser.add_argument(
        "--dataset_type",
        default="shapenet_v1",
        choices=[
            "shapenet_quad",
            "shapenet_v1",
            "shapenet_v2",
            "surreal_bodies",
            "dynamic_faust",
        ],
        help="The type of the dataset type to be used"
    )
    parser.add_argument(
        "--occupancy_type",
        default="surface",
        choices=[
            "surface",
            "volume"
        ],
        help="Choose whether to export occpairs from surface or volume"
    )
    parser.add_argument(
        "--model_tags",
        type=lambda x: x.split(","),
        default=[],
        help="Tags to the models to be used"
    )
    parser.add_argument(
        "--category_tags",
        type=lambda x: x.split(","),
        default=[],
        help="Category tags to the models to be used"
    )
    parser.add_argument(
        "--n_surface_samples",
        type=int,
        default=100000,
        help="Number of points to be sampled from the surface of the mesh"
    )
    parser.add_argument(
        "--normalize_mesh",
        action="store_true",
        help="When set normalize mesh while loading"
    )

    args = parser.parse_args(argv)

    dataset = (DatasetBuilder(dict(data={}))
            .with_dataset(args.dataset_type)
            .filter_train_test(
                splits_factory(args.dataset_type)(args.train_test_splits_file),
               ["train", "test", "val"]
             )
            .filter_category_tags(args.category_tags)
            .filter_tags(args.model_tags)
            .build(args.dataset_directory))

    prog = Progbar(len(dataset))
    i = 0
    for sample in dataset:
        # Update progress bar
        prog.update(i+1)
        i += 1

        # Assemble the target path and ensure the parent dir exists
        category_tag = sample.tag.split(":")[0]
        model_tag = sample.tag.split(":")[-1]
        path_to_file = os.path.join(
            args.dataset_directory,
            category_tag,
            occupancy_pairs_subdir(args.occupancy_type, args.normalize_mesh),
            "{}.npz".format(model_tag)
        )
        ensure_parent_directory_exists(path_to_file)

        # Make sure we are the only ones creating this file
        with DirLock(path_to_file + ".lock") as lock:
            if not lock.is_acquired:
                continue
            if os.path.exists(path_to_file):
                continue

            points, labels = export_occupancy_pairs(args.occupancy_type)(
                sample.path_to_mesh_file,
                args.n_surface_samples,
                normalize=args.normalize_mesh
            )

            np.savez(
                path_to_file,
                points=points.reshape(-1, 3),
                occupancies=labels
            )

    prog.update(len(dataset))


if __name__ == "__main__":
    main(sys.argv[1:])
