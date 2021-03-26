import argparse

import numpy as np
from pyquaternion import Quaternion
import torch

from hierarchical_primitives.utils.filter_sqs import filter_primitives, \
    primitive_parameters_from_indices, qos_less, volume_larger

from arguments import add_dataset_parameters
from training_utils import load_config
from visualization_utils import scene_init, load_ground_truth, get_color
from utils import build_dataloader_and_network_from_args

from simple_3dviz import Mesh, Lines
from simple_3dviz.behaviours import SceneInit
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.behaviours.keyboard import SnapshotOnKey
from simple_3dviz.behaviours.misc import LightToCamera, SortTriangles
from simple_3dviz.behaviours.movements import LocalModelRotation, \
    LightTrajectory, CameraTrajectory, CameraTargetTrajectory
from simple_3dviz.behaviours.trajectory import Circle, BackAndForth, \
    Lines as TrajectoryLine
from simple_3dviz.utils import render
from simple_3dviz.window import show


def _unpack(p, offset, active, color_active, color_inactive):
    alpha = p.sizes.view(-1, 3).to("cpu").detach().numpy()
    epsilon = p.shapes.view(-1, 2).to("cpu").detach().numpy()
    t = p.translations.view(-1, 3).to("cpu").detach().numpy()
    R = np.stack([
        Quaternion(r.to("cpu").detach().numpy()).rotation_matrix
        for r in p.rotations.view(-1, 4)
    ], axis=0)
    colors = [color_inactive]*p.n_primitives
    for i, c in color_active:
        colors[i] = c
    return alpha, epsilon, t, R, colors


def get_filtered_renderables(tree, group_indices, qos_threshold, vol_threshold,
                             padding=0.3, order="xyz", no_lines=False,
                             visualize_as_tree=False,
                             color_inactive=(0.8, 0.8, 0.8, 1.0),
                             group_color=False, vertex_count=100,
                             line_offset=[0, 0, 0]):
    max_depth = max(d for (d, i) in group_indices)

    def get_center(depth, index):
        if visualize_as_tree:
            max_width = (2.**max_depth) * (1+padding)
            width = (2.**(max_depth - depth)) * (1+padding)
            tree = dict(
                x=0,
                y=float(index)*(width)-(max_width-width)/2,
                z=-float(depth)*(1+padding)
            )
        else:
            width = (2.**depth) * (1+padding)
            tree = dict(
                x=0,
                y=float(index)*(1+padding)-(width-1-padding)/2,
                z=-float(depth)*(1+padding)
            )
        return np.array([[tree[axis] for axis in order]])

    def get_ancestors(depth, index, storage):
        storage.add((depth, index))
        pdepth = depth-1
        pindex = index//2
        parent = pdepth, pindex
        if parent not in storage:
            get_ancestors(pdepth, pindex, storage)

    active_prims = filter_primitives(
        tree,
        qos_less(qos_threshold),
        volume_larger(vol_threshold)
    )
    active_prims_map = {p: i for (i, p) in enumerate(active_prims)}
    valid_nodes = set([(0, 0)])
    for d, i in active_prims:
        get_ancestors(d, i, valid_nodes)

    def children(d, i):
        if (d, i) in active_prims_map:
            return [active_prims_map[d, i]]
        elif d < len(tree):
            return children(d+1, 2*i) + children(d+1, 2*i+1)
        else:
            return []

    def colors(d, i):
        max_d = len(tree)
        if (d, i) in active_prims_map:
            return [(active_prims_map[d, i], get_color(d, i, 2**(max_d-1)))]
        elif d < max_d:
            return colors(d+1, 2*i) + colors(d+1, 2*i+1)
        else:
            return []

    for d, i in group_indices:
        if d > 0 and (d, i) in valid_nodes and len(children(d-1, i//2)) == 1:
            valid_nodes.remove((d, i))

    P = primitive_parameters_from_indices(tree, active_prims)
    meshes = []
    lines = []
    for d, i in group_indices:
        if (d, i) not in valid_nodes:
            continue
        sq_colors = colors(d, i)
        if group_color:
            sq_colors = [
                (j, get_color(d, i))
                for j, _ in sq_colors
            ]
        if len(group_indices) == 1:
            meshes.append(Mesh.from_superquadrics(
                *_unpack(
                    P,
                    get_center(0, 0),
                    children(d, i),
                    sq_colors,
                    color_inactive
                ),
                vertex_count=vertex_count,
                offset=get_center(0, 0)
            ))
        else:
            meshes.append(Mesh.from_superquadrics(
                *_unpack(
                    P,
                    get_center(d, i),
                    children(d, i),
                    sq_colors,
                    color_inactive
                ),
                vertex_count=vertex_count,
                offset=get_center(d, i)
            ))
        if d > 0 and not no_lines:
            lines.extend([
                get_center(d-1, i//2)[0] + np.asarray(line_offset),
                get_center(d, i)[0] + np.asarray(line_offset)
            ])
    meshes.append(Lines(lines, (0.1, 0.1, 0.1), width=0.05))

    if no_lines:
        meshes.pop()

    return meshes


def get_renderables(P, group_indices, vertex_count=100):
    padding = 0.3
    order = "xyz"
    color_inactive = (0.8, 0.8, 0.8, 0.4)
    max_depth = int(np.log(P.n_primitives)/np.log(2)) + 1

    def get_center(depth, index):
        width = (2.**depth) * (1+padding)
        tree = dict(
            x=0,
            y=float(index)*(1+padding)-(width-1-padding)/2,
            z=-float(depth)*(1+padding)
        )
        return np.array([[tree[axis] for axis in order]])

    def children(d, i):
        if d+1 < max_depth:
            return children(d+1, 2*i) + children(d+1, 2*i+1)
        else:
            return [i]

    def colors(d, i):
        if d+1 < max_depth:
            return colors(d+1, 2*i) + colors(d+1, 2*i+1)
        else:
            return [(i, get_color(d, i, 2**(max_depth-1)))]

    meshes = []
    lines = []
    for d, i in group_indices:
        meshes.append(Mesh.from_superquadrics(
            *_unpack(
                P,
                get_center(d, i),
                children(d, i),
                colors(d, i),
                color_inactive
            ),
            vertex_count=vertex_count
        ))
        if d > 0:
            lines.extend([
                get_center(d-1, i//2)[0],
                get_center(d, i)[0]
            ])
    meshes.append(Lines(lines, (0.1, 0.1, 0.1), width=0.05))

    return meshes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Do the forward pass and visualize the parsing tree"
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
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="0,0,0,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=185,
        help="Number of frames to be rendered"
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
        default="0.0,0.0,0.0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--group_indices",
        type=lambda x: [
            [int(xij) for xij in xi.split(",")]
            for xi in x.split(";")
        ],
        default="0,0;1,0;1,1;2,0;2,1;2,2;2,3",
        help="Choose the depth, index pairs to be visualized"
    )
    parser.add_argument(
        "--full_indices",
        action="store_true",
        help="Assume the group indices are the full tree"
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
        "--qos_threshold",
        type=float,
        default=1.0,
        help="Stop partitioning based on the predicted quality"
    )
    parser.add_argument(
        "--vol_threshold",
        type=float,
        default=0.0,
        help="Only show primitives with volume larger than vol_threshold"
    )
    parser.add_argument(
        "--order",
        default="xyz",
        help="Defines the axis for the tree"
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.3,
        help="Define the padding for the tree"
    )
    parser.add_argument(
        "--sorting",
        action="store_true",
        help="Perform the sorting to correctly display transparent objects"
    )
    parser.add_argument(
        "--no_lines",
        action="store_true",
        help="Visualize the tree without the lines"
    )
    parser.add_argument(
        "--visualize_as_tree",
        action="store_true",
        help="Visualize the tree as a typical tree"
    )
    parser.add_argument(
        "--group_color",
        action="store_true",
        help="Color the active prims based on the group"
    )
    parser.add_argument(
        "--vertex_count",
        type=int,
        default=100,
        help="Number of vertices per superquadric"
    )
    parser.add_argument(
        "--with_rotating_tree",
        action="store_true",
        help="Visualize while rotating the tree"
    )
    parser.add_argument(
        "--camera_zoom",
        action="store_true",
        help="Zoom out while visualizing"
    )
    parser.add_argument(
        "--line_offset",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Define an offset for the lines"
    )

    add_dataset_parameters(parser)
    args = parser.parse_args()

    if args.run_on_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on ", device)

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

        if args.full_indices:
            args.group_indices = sum(
                [[(d, k) for k in range(2**d)] for d in range(len(F))],
                []
            )
        if args.qos_threshold > 0 or args.vol_threshold > 0:
            renderables = get_filtered_renderables(
                F,
                args.group_indices,
                args.qos_threshold,
                args.vol_threshold,
                order=args.order,
                padding=args.padding,
                no_lines=args.no_lines,
                visualize_as_tree=args.visualize_as_tree,
                group_color=args.group_color,
                line_offset=args.line_offset
            )
        else:
            renderables = get_renderables(
                F[-1] if args.from_fit else P[-1], args.group_indices
            )
        behaviours = [
            SceneInit(
                scene_init(
                    args.mesh,
                    args.up_vector,
                    args.camera_position,
                    args.camera_target,
                    args.background
                )
            )
        ]
        if args.with_rotating_tree:
            behaviours += [
                LocalModelRotation(args.up_vector, speed=np.pi/90),
                LightTrajectory(
                    Circle(
                        args.camera_target,
                        args.camera_position,
                        args.up_vector
                    ),
                    speed=1/180
                )
            ]
        else:
            behaviours += [LightToCamera()]

        if args.camera_zoom:
            d = np.array(args.camera_position) - args.camera_target
            d /= np.sqrt(d.dot(d))
            behaviours += [
                CameraTargetTrajectory(
                    BackAndForth(TrajectoryLine(
                        args.camera_target,
                        args.camera_target - np.array(args.up_vector)*5
                    )),
                    speed=0.001
                ),
                CameraTrajectory(
                    BackAndForth(TrajectoryLine(
                        args.camera_position,
                        args.camera_position + d*15
                    )),
                    speed=0.001
                )
            ]

        if args.sorting:
            behaviours.append(SortTriangles())

        # Behaviours do be considered while rendering
        if args.without_screen:
            behaviours += [
                SaveFrames(args.save_frames, 2),
                SaveGif("/tmp/out.gif", 2)
            ]
            render(renderables, size=args.window_size, behaviours=behaviours,
                   n_frames=args.n_frames)
        else:
            show(renderables, size=args.window_size,
                 behaviours=behaviours + [SnapshotOnKey()])
