#!/usr/bin/env python3
"""Script used for visualizing the partitioning process
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import torch

from arguments import add_dataset_parameters
from training_utils import load_config
from utils import build_dataloader_and_network_from_args
from visualization_utils import scene_init, load_ground_truth, get_color

from simple_3dviz import Mesh
from simple_3dviz.behaviours.misc import CycleThroughObjects, LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.window import simple_window


def _unpack(p, i, max_depth, colors, color_siblings=False):
    alpha = p.sizes.view(-1, 3).to("cpu").detach().numpy()
    epsilon = p.shapes.view(-1, 2).to("cpu").detach().numpy()
    t = p.translations.view(-1, 3).to("cpu").detach().numpy()
    R = np.stack([
        Quaternion(r.to("cpu").detach().numpy()).rotation_matrix
        for r in p.rotations.view(-1, 4)
    ], axis=0)
    M = alpha.shape[0]
    if max_depth > 0:
        colors = []
        for idx in range(M):
            colors.append(get_color(i, idx, max_depth))
    else:
        if color_siblings and i != 0:
            c = (((np.arange(0, M, 2)+i) % len(colors)).tolist())
            c_tiled = []
            for ci, m in zip(c, [2]*len(c)):
                c_tiled.extend([ci]*m)
            colors = colors[c_tiled]
        else:
            colors = colors[((np.arange(M)+i) % len(colors)).tolist()]
    return alpha, epsilon, t, R, colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the space partitioning"
    )
    parser.add_argument(
        "dataset_directory",
        help="Path to the directory containing the dataset"
    )
    parser.add_argument(
        "train_test_splits_file",
        default=None,
        help="Path to the train-test splits file"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="The path to the previously trainined model to be used"
    )
    parser.add_argument(
        "--config_file",
        default="../config/default.yaml",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--run_on_gpu",
        action="store_true",
        help="Use GPU"
    )
    parser.add_argument(
        "--mesh",
        type=load_ground_truth,
        help="File of ground truth mesh"
    )
    parser.add_argument(
        "--interval",
        type=lambda x: int(float(x)*60),
        default=30,
        help="Set the interval to update the partition in seconds"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=5,
        help="Save every that many frames"
    )
    parser.add_argument(
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="0,0,0,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,1",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-2.0,-2.0,-2.0",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--color_siblings",
        action="store_true",
        help="Use the same color to depict siblings"
    )
    parser.add_argument(
        "--from_fit",
        action="store_true",
        help="Visulize everything based on primitive_params.fit"
    )

    add_dataset_parameters(parser)
    args = parser.parse_args()

    if args.run_on_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    config = load_config(args.config_file)
    dataloader, network = build_dataloader_and_network_from_args(
        args, config, device=device
    )

    for sample in dataloader:
        # Do the forward pass and estimate the primitive parameters
        X = sample[0].to(device)
        y_hat = network(X)
        F = y_hat.fit
        [C, P] = y_hat.space_partition
        n_primitives = y_hat.n_primitives
        colors = torch.tensor(np.array(
            plt.cm.jet(np.linspace(0, 1, 16))
        ))

        if args.from_fit:
            max_depth = 2**(len(F) - 1)
        else:
            max_depth = -1
        meshes = [
            [Mesh.from_superquadrics(
                *_unpack(p, i, max_depth, colors, args.color_siblings)
            )]
            for i, p in enumerate(F if args.from_fit else P)
        ]
        behaviours = [
            CycleThroughObjects(meshes, interval=args.interval),
            LightToCamera()
        ]
        if args.save_frames:
            behaviours += [
                SaveFrames(args.save_frames, args.save_frequency)
            ]
        if args.with_rotating_camera:
            behaviours += [
                CameraTrajectory(
                    Circle([0, 0, 1], [4, 0, 1], [0, 0, 1]), 0.001
                ),
            ]
        simple_window(
            scene_init(args.mesh, args.up_vector, args.camera_position,
                       args.camera_target, args.background)
        ).add_behaviours(behaviours).show()
