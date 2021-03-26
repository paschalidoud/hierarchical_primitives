import os

import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
import trimesh

from simple_3dviz import Mesh
from simple_3dviz.scripts.mesh_viewer import tab20

from hierarchical_primitives.networks.primitive_parameters import\
    PrimitiveParameters
from hierarchical_primitives.utils.filter_sqs import filter_primitives, \
    qos_less, volume_larger, get_primitives_indices, \
    primitive_parameters_from_indices

from utils import get_non_overlapping_primitives


def scene_init(mesh, up_vector, camera_position, camera_target, background):
    def inner(scene):
        scene.background = background
        scene.up_vector = up_vector
        scene.camera_position = camera_position
        scene.camera_target = camera_target
        if mesh is not None:
            scene.add(mesh)
    return inner


def load_ground_truth(mesh_file):
    m = Mesh.from_file(mesh_file, color=(0.8, 0.8, 0.8, 0.3))
    # m.to_unit_cube()
    return m


def save_renderables_as_ply(
    renderables,
    args,
    filepath="/tmp/prediction.ply"
):
    """
    Arguments:
    ----------
        renderables: simple_3dviz.Mesh objects that were rendered
    """
    m = None
    for r in renderables:
        # Build a mesh object using the vertices loaded before and get its
        # convex hull
        _m = trimesh.Trimesh(vertices=r._vertices).convex_hull
        # Apply color
        for i in range(len(_m.faces)):
            _m.visual.face_colors[i] = (r._colors[i] * 255).astype(np.uint8)
        m = trimesh.util.concatenate(_m, m)

    m.export(filepath, file_type="ply")

    print("Saved prediction as ply file in {}".format(
        os.path.join(args.output_directory, "prediction.ply")
    ))


def get_color(d, i, M=0):
    colors = np.array(plt.cm.jet(np.linspace(0, 1, 16)))
    order = np.array([12, 13,  7,  8,  9,  3, 14, 11,
                      10,  2,  1,  0,  5,  4,  6, 15])
    colors = np.array(tab20)
    order = np.arange(len(tab20))
    d = np.asarray(d)
    i = np.asarray(i)
    # order = [ 8, 38, 20, 15,  3,  0, 30, 37, 34, 28, 32, 24, 26, 14, 39,
    #           27, 11, 4,  1, 17, 35, 23, 36,  2, 21, 31,  5, 19,  7, 10,
    #           9, 22, 16, 25, 33, 18, 13, 29,  6, 12]
    return colors[order[(2**d-1+i) % len(colors)]]


def get_color_groupped(d, i, max_depth):
    if d <= max_depth:
        return get_color(d, i, 0)
    else:
        return get_color_groupped(d-1, i // 2, max_depth)


def _unpack(p):
    alpha = p.sizes.view(-1, 3).to("cpu").detach().numpy()
    epsilon = p.shapes.view(-1, 2).to("cpu").detach().numpy()
    t = p.translations.view(-1, 3).to("cpu").detach().numpy()
    R = np.stack([
        Quaternion(r.to("cpu").detach().numpy()).rotation_matrix
        for r in p.rotations.view(-1, 4)
    ], axis=0)
    return alpha, epsilon, t, R


def _renderables_from_fit(y_hat, args):
    F = y_hat.fit
    active_prims = filter_primitives(
        F,
        qos_less(args.qos_threshold),
        volume_larger(args.vol_threshold),
    )
    active_prims_map = {p: i for (i, p) in enumerate(active_prims)}
    return [
        Mesh.from_superquadrics(*_unpack(
            PrimitiveParameters.with_keys(
                translations=F[depth].translations_r[:, idx],
                rotations=F[depth].rotations_r[:, idx],
                sizes=F[depth].sizes_r[:, idx],
                shapes=F[depth].shapes_r[:, idx]
            )),
            get_color_groupped(depth, idx, args.max_depth) if args.group_color
            else get_color(depth, idx)
        )
        for depth, idx in active_prims_map
    ], active_prims


def _renderables_from_partition(y_hat, args):
    [C, P] = y_hat.space_partition
    active_prims_map = {
        p: i for (i, p) in enumerate(get_primitives_indices(P))
    }
    return [
        Mesh.from_superquadrics(*_unpack(
            PrimitiveParameters.with_keys(
                translations=P[depth].translations_r[:, idx],
                rotations=P[depth].rotations_r[:, idx],
                sizes=P[depth].sizes_r[:, idx],
                shapes=P[depth].shapes_r[:, idx]
            )),
            get_color_groupped(depth, idx, args.max_depth) if args.group_color
            else get_color(depth, idx)
        )
        for depth, idx in active_prims_map
    ], get_primitives_indices(P)


def _renderables_from_flat_partition(y_hat, args):
    _, P = y_hat.space_partition
    # Collect the sqs that have prob larger than threshold
    indices = [
        i for i in range(y_hat.n_primitives)
        if y_hat.probs_r[0, i] >= args.prob_threshold
    ]
    active_prims_map = {
        (0, j): i for i, j in enumerate(indices)
    }
    return [
        Mesh.from_superquadrics(*_unpack(
            PrimitiveParameters.with_keys(
                translations=P[-1].translations_r[0, indices],
                rotations=P[-1].rotations_r[0, indices],
                sizes=P[-1].sizes_r[0, indices] / 2.0,
                shapes=P[-1].shapes_r[0, indices]
            )),
            get_color(0, indices)
        )
    ], indices


def _renderables_from_flat_primitives(y_hat, args):
    # Collect the sqs that have prob larger than threshold
    indices = [
        i for i in range(y_hat.n_primitives)
        if y_hat.probs_r[0, i] >= args.prob_threshold
    ]
    active_prims_map = {
        (0, j): i for i, j in enumerate(indices)
    }

    if args.with_post_processing:
        indices = get_non_overlapping_primitives(y_hat, indices)

    return [Mesh.from_superquadrics(
        *_unpack(
            PrimitiveParameters.with_keys(
                translations=y_hat.translations_r[0, indices],
                rotations=y_hat.rotations_r[0, indices],
                sizes=y_hat.sizes_r[0, indices],
                shapes=y_hat.shapes_r[0, indices]
            )),
            get_color(0, indices)
    )], indices


def get_renderables(y_hat, args):
    """Depending on the arguments compute which primitives should be rendered
    """
    # len(y_hat.fit) == 1 means that we do not have hierarhcy
    if len(y_hat.fit) == 1:
        if args.from_flat_partition:
            return _renderables_from_flat_partition(y_hat, args)
        else:
            return _renderables_from_flat_primitives(y_hat, args)

    if args.from_fit:
        return _renderables_from_fit(y_hat, args)
    else:
        return _renderables_from_partition(y_hat, args)


def get_primitive_parameters_from_indices(y_hat, active_prims, args):
    # len(y_hat.fit) == 1 means that we do not have hierarhcy
    if len(y_hat.fit) == 1:
        return primitive_parameters_from_indices(
            y_hat.fit,
            [(-1, ap) for ap in active_prims]
        )
    if args.from_fit:
        return primitive_parameters_from_indices(y_hat.fit, active_prims)
    else:
        [C, P] = y_hat.space_partition
        return primitive_parameters_from_indices(P, active_prims)

def visualize_sharpness(sharpness, epoch):
    import seaborn as sns
    f = plt.figure(figsize=(8, 6))
    sns.barplot(
        np.arange(sharpness.shape[0]), sharpness
    )
    plt.title("Epoch {}".format(epoch))
    plt.ylim([0, 10.5])
    plt.ylabel("Sharpness")
    plt.xlabel("Primitive id")
    plt.savefig("/tmp/sharpness_{:03d}.png".format(epoch))
    plt.close()
