#!/usr/bin/env python3

import argparse
from subprocess import call
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from hierarchical_primitives.utils.filter_sqs import filter_primitives, \
    primitive_parameters_from_indices, qos_less, volume_larger
from hierarchical_primitives.utils.sq_mesh import sq_mesh_from_primitive_params

from visualization_utils import get_color


def save_ply(f, points, normals, colors):
    assert len(points.shape) == 2
    assert len(points.shape) == len(normals.shape)
    assert len(points.shape) == len(colors.shape)

    header = "\n".join([
        "ply",
        "format binary_{}_endian 1.0".format(sys.byteorder),
        "comment Raynet pointcloud!",
        "element vertex {}".format(len(points)),
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property uchar alpha",
        "end_header\n"
    ])
    f.write(header.encode("ascii"))
    colors = (colors*255).astype(np.uint8)
    for p, n, c in zip(points, normals, colors):
        p.astype(np.float32).tofile(f)
        n.astype(np.float32).tofile(f)
        c.tofile(f)
    f.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create a single mesh from the predicted SQs"
    )
    parser.add_argument(
        "primitives_file",
        help="Path to the file containing the primitives"
    )
    parser.add_argument(
        "recon_binary",
        help="Poisson reconstruction binary"
    )
    parser.add_argument(
        "output_file",
        help="Save the mesh in this file"
    )
    parser.add_argument(
        "--qos_threshold",
        type=float,
        default=1,
        help="Stop partitioning based on the predicted quality"
    )
    parser.add_argument(
        "--vol_threshold",
        default=0,
        type=float,
        help="Discard primitives with volume smaller than this threshold"
    )

    args = parser.parse_args(argv)

    [C, P], F = torch.load(args.primitives_file)
    active_primitives = filter_primitives(
        F,
        qos_less(args.qos_threshold),
        volume_larger(args.vol_threshold)
    )
    primitives = primitive_parameters_from_indices(
        F,
        active_primitives
    )
    pts, normals, prim_indices = sq_mesh_from_primitive_params(
        primitives,
        normals=True,
        prim_indices=True
    )
    colors = np.array([
        get_color(d, i)
        for d, i in active_primitives
    ])[prim_indices].reshape(-1, 4)
    with NamedTemporaryFile(suffix=".ply") as f:
        save_ply(f, pts, normals, colors)
        call([
            args.recon_binary,
            "--in", f.name,
            "--out", args.output_file,
            "--colors"
        ])


if __name__ == "__main__":
    main()
