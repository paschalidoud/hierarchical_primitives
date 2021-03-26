#!/usr/bin/env python3
"""Script used to evaluate the predicted mesh
"""
import argparse
import os
import sys

import numpy as np
from pykdtree.kdtree import KDTree
import torch

from arguments import add_dataset_parameters
from training_utils import get_loss_options, load_config

from hierarchical_primitives.common.base import build_dataset
from hierarchical_primitives.common.dataset import DatasetWithTags
from hierarchical_primitives.networks.base import build_network
from hierarchical_primitives.primitives import get_implicit_surface
from hierarchical_primitives.utils.metrics import compute_iou
from hierarchical_primitives.utils.progbar import Progbar
from hierarchical_primitives.utils.sq_mesh import sq_mesh_from_primitive_params


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


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


class MeshEvaluator(object):
    """ Class for evaluation the predicted mesh
    Adapted code from
    https://github.com/autonomousvision/occupancy_networks/eval.py
    to handlethe mesh evaluation process.
    Arguments:
    ---------
        n_points (int): number of points to be used for evaluation
    """
    def __init__(self, n_points=100000):
        self.n_points = n_points

    def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt,
                  points_iou, occ_tgt):
        ''' Evaluates a mesh.
        Args:
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            points_iou (numpy_array): points tensor for IoU evaluation
            occ_tgt (numpy_array): GT occupancy values for IoU points
        '''
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            pointcloud, idx = mesh.sample(self.n_points, return_index=True)
            pointcloud = pointcloud.astype(np.float32)
            normals = mesh.face_normals[idx]
        else:
            pointcloud = np.empty((0, 3))
            normals = np.empty((0, 3))

        metrics = self.eval_pointcloud(
            pointcloud, pointcloud_tgt, normals, normals_tgt)

        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            occ = check_mesh_contains(mesh, points_iou)
            metrics['iou'] = compute_iou(occ, occ_tgt)
        else:
            metrics['iou'] = 0.

        return metrics

    def eval_pointcloud(self, pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None):
        """
        Evaluates a point cloud wrt to another pointcloud

        Arguments:
        ----------
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        """
        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        # chamfer = completeness2 + accuracy2
        chamfer = 0.5 * (completeness + accuracy)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )

        metrics = {
            "completeness": completeness,
            "accuracy": accuracy,
            "normals_completeness": completeness_normals,
            "normals_accuracy": accuracy_normals,
            "normals": normals_correctness,
            "completeness2": completeness2,
            "accuracy2": accuracy2,
            "ch_l1": chamfer,
        }

        return metrics

    def eval_predicted_mesh(
        self,
        pred_mesh,
        target_points,
        target_labels
    ):
        """
        Arguments:
        ---------
        pred_mesh: Trimesh instance containing the predicted mesh that we want
                   to evaluate
        target_points: numpy array of size Nx3 containing N points from the
                       target mesh
        target_labels: numpy array of size Nx1 containing the corresponding
                       occupancy labels of the target_points
        """
        points = pred_mesh.sample(self.n_points)
        metrics = self.eval_pointcloud(points, target_points)

        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            occ = pred_mesh.contains(target_points)
            metrics['iou'] = compute_iou(occ, target_labels)
        else:
            metrics['iou'] = 0.

        return metrics

    def eval_mesh_with_primitive_params(
        self,
        prim_params,
        target_points,
        target_labels,
        target_weights,
        surface_points,
        options
    ):
        """
        Arguments:
        ---------
        target_points: numpy array of size Nx3 containing N points in the bbox
                       of the target mesh
        target_labels: numpy array of size Nx1 containing the corresponding
                       occupancy labels of the target_points
        target_weights: numpy array of size Nx1 containing the corresponding
                       sampling probabilities of the target_points
        surface_points: numpy array of size Sx3 containing S points on the
                        surface of the target mesh
        """
        assert len(target_points.shape) == 3
        points = sq_mesh_from_primitive_params(
            prim_params, surface_points.shape[1]
        ).astype(np.float32)
        metrics = self.eval_pointcloud(
            points,
            surface_points[0].cpu().detach().numpy().astype(np.float32)
        )
        F, _ = get_implicit_surface(
            target_points,
            prim_params.translations_r,
            prim_params.rotations_r,
            prim_params.sizes_r,
            prim_params.shapes_r,
            prim_params.sharpness_r,
        )
        occ = torch.max(F, dim=-1)[0]
        metrics["iou"] = compute_iou(occ, target_labels, target_weights)

        return metrics


def main(argv):
    parser = argparse.ArgumentParser(
        description="Do the forward pass and estimate a set of primitives"
    )
    parser.add_argument(
        "dataset_directory",
        help="Path to the directory containing the dataset"
    )
    parser.add_argument(
        "output_directory",
        help="Save the output files in that directory"
    )
    parser.add_argument(
        "train_test_splits_file",
        default=None,
        help="Path to the train-test splits file"
    )
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="The path to the previously trainined model to be used"
    )
    parser.add_argument(
        "--eval_on_train",
        action="store_true",
        help="When true evaluate on training set"
    )

    parser.add_argument(
        "--run_on_gpu",
        action="store_true",
        help="Use GPU"
    )

    add_dataset_parameters(parser)
    args = parser.parse_args(argv)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if args.run_on_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Build the network architecture to be used for training
    config = load_config(args.config_file)
    network = build_network(args.config_file, args.weight_file, device=device)
    network.eval()

    dataset = build_dataset(
        config,
        args.dataset_directory,
        args.dataset_type,
        args.train_test_splits_file,
        args.model_tags,
        args.category_tags,
        ["test"],
        #["test", "train", "val"],
        # config["data"].get("test_split", ["test"]) if not args.eval_on_train else ["train"],
        random_subset=args.random_subset
    )
    dataset = DatasetWithTags(dataset)

    prog = Progbar(len(dataset))
    chamfer_l1 = []
    iou = []
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            # Update progress bar
            prog.update(i+1)

            # Fix the shape as if we had batch size 1
            X = sample[0].to(device).unsqueeze(0)
            y_target = [yi.to(device).unsqueeze(0) for yi in sample[1:-1]]

            # Create the output path
            tag = dataset.internal_dataset_object[i].tag
            assert len(tag.split(":")) == 2
            category_tag = tag.split(":")[0]
            model_tag = tag.split(":")[1]
            path_to_file = os.path.join(
                args.output_directory,
                "{}_{}.npz".format(category_tag, model_tag)
            )
            stats_per_tag = dict()

            # Optimistically check whether the file already exists in order to
            # reduce the burden to the file system with less locking
            # NOTE: Locking is still required to ensure that we have no race
            #       condition
            if os.path.exists(path_to_file):
                continue

            with DirLock(path_to_file + ".lock") as lock:
                if not lock.is_acquired:
                    continue
                if os.path.exists(path_to_file):
                    continue

                # Do the forward pass and estimate the primitive parameters
                y_hat = network(X)

                metrics = MeshEvaluator().eval_mesh_with_primitive_params(
                    y_hat,
                    y_target[0],
                    y_target[1].squeeze(-1),
                    y_target[2].squeeze(-1),
                    y_target[3][..., :3],
                    get_loss_options(config)
                )
                stats_per_tag[tag] = {
                    "iou": metrics["iou"],
                    "accuracy": metrics["accuracy"],
                    "normals_completeness": metrics["normals_completeness"],
                    "normals_accuracy": metrics["normals_accuracy"],
                    "normals": metrics["normals"],
                    "completeness2": metrics["completeness2"],
                    "accuracy2": metrics["accuracy2"],
                    "ch_l1": metrics["ch_l1"]
                }
                    
                np.savez(path_to_file, stats=stats_per_tag)


if __name__ == "__main__":
    main(sys.argv[1:])
