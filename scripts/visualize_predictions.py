#!/usr/bin/env python3
"""Script used for perfoming a forward pass on a previously trained model and
   visualizing the predicted primitives.
"""
import argparse
import os
import pickle
import sys

import torch

from arguments import  add_dataset_parameters
from compute_metrics import report_metrics
from training_utils import load_config
from visualization_utils import scene_init, load_ground_truth, \
    get_renderables, get_primitive_parameters_from_indices, \
    save_renderables_as_ply, visualize_sharpness
from utils import build_dataloader_and_network_from_args, \
    get_non_overlapping_primitives
from hierarchical_primitives.utils.sq_mesh import sq_meshes

from simple_3dviz.behaviours import SceneInit
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.keyboard import SnapshotOnKey
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.utils import render
from simple_3dviz.window import show


def main(argv):
    parser = argparse.ArgumentParser(
        description="Do the forward pass and estimate a set of primitives"
    )
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        help="Save the output files in that directory"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="The path to the previously trainined model to be used"
    )
    parser.add_argument(
        "--run_on_gpu",
        action="store_true",
        help="Use GPU"
    )
    parser.add_argument(
        "--qos_threshold",
        default=1.0,
        type=float,
        help="Split primitives if predicted qos less than this threshold"
    )
    parser.add_argument(
        "--vol_threshold",
        default=0.0,
        type=float,
        help="Discard primitives with volume smaller than this threshold"
    )
    parser.add_argument(
        "--prob_threshold",
        default=0.0,
        type=float,
        help="Discard primitives with probability smaller than this threshold"
    )
    parser.add_argument(
        "--with_post_processing",
        action="store_true",
        help="Remove overlapping primitives"
    )
    parser.add_argument(
        "--mesh",
        type=load_ground_truth,
        help="File of ground truth mesh"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=200,
        help="Number of frames to be rendered"
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
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-2.0,-2.0,-2.0",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=3,
        help="Maximum depth to visualize"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--from_fit",
        action="store_true",
        help="Visulize everything based on primitive_params.fit"
    )
    parser.add_argument(
        "--from_flat_partition",
        action="store_true",
        help=("Visulize everything based on primitive_params.space_partition"
              " with a single depth")
    )
    parser.add_argument(
        "--group_color",
        action="store_true",
        help="Color the active prims based on the group"
    )
    parser.add_argument(
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object"
    )
    parser.add_argument(
        "--visualize_sharpness",
        action="store_true",
        help="When set visualize the sharpness together with the prediction"
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

    config = load_config(args.config_file)
    dataloader, network = build_dataloader_and_network_from_args(
        args, config, device=device
    )

    for sample in dataloader:
        # Do the forward pass and estimate the primitive parameters
        X = sample[0].to(device)
        y_hat = network(X)
        #import matplotlib.pyplot as plt
        #import seaborn as sns
        #import numpy as np
        #f = plt.figure(figsize=(8, 6))
        #sns.barplot(
        #    np.arange(y_hat.n_primitives),
        #    y_hat.sharpness_r.squeeze(0).detach().numpy()[:, 0]
        #)
        #plt.title("Epoch {}".format(args.weight_file.split("/")[-1].split("_")[-1]))
        #plt.ylim([0, 10.5])
        #plt.ylabel("Sharpness")
        #plt.xlabel("Primitive id")
        #plt.savefig("/tmp/sharpness_{:03d}.png".format(
        #    int(args.weight_file.split("/")[-1].split("_")[-1]))
        #)
        #plt.close()

        renderables, active_prims = get_renderables(y_hat, args)
        with open(os.path.join(args.output_directory, "renderables.pkl"), "wb") as f:
            pickle.dump(renderables, f)
        print(active_prims)

        behaviours = [
            SceneInit(
                scene_init(
                    args.mesh,
                    args.up_vector,
                    args.camera_position,
                    args.camera_target,
                    args.background
                )
            ),
            LightToCamera(),
        ]
        if args.with_rotating_camera:
            behaviours += [
                CameraTrajectory(
                    Circle(
                        args.camera_target,
                        args.camera_position,
                        args.up_vector
                    ),
                    speed=1/180
                )
            ]
        if args.without_screen:
            behaviours += [
                SaveFrames(args.save_frames, 2),
                SaveGif("/tmp/out.gif", 2)
            ]
            render(renderables, size=args.window_size, behaviours=behaviours,
                   n_frames=args.n_frames)
        else:
            behaviours += [
                SnapshotOnKey(path=args.save_frames, keys={"<ctrl>", "S"})
            ]
            show(renderables, size=args.window_size, behaviours=behaviours)

        # Based on the active primitives report the metrics
        active_primitive_params = \
            get_primitive_parameters_from_indices(y_hat, active_prims, args)
        report_metrics(
            active_primitive_params,
            config,
            config["data"]["dataset_type"],
            args.model_tags,
            config["data"]["dataset_directory"]
        )
        if args.with_post_processing:
            indices = get_non_overlapping_primitives(y_hat, active_prims)
        else:
            indices = None
        for i, m in enumerate(sq_meshes(y_hat, indices)):
            m.export(
                os.path.join(args.output_directory, "predictions-{}.ply").format(i),
                file_type="ply"
            )

        if y_hat.space_partition is not None:
            torch.save(
                [y_hat.space_partition, y_hat.fit],
                os.path.join(args.output_directory, "space_partition.pkl")
            )
        if args.visualize_sharpness:
            visualize_sharpness(
                y_hat.sharpness_r.squeeze(0).detach().numpy()[:, 0],
                int(args.weight_file.split("/")[-1].split("_")[-1])
            )


if __name__ == "__main__":
    main(sys.argv[1:])
