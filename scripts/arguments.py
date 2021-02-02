
def add_voxelizer_parameters(parser):
    parser.add_argument(
        "--voxelizer_factory",
        choices=[
            "occupancy_grid",
            "tsdf_grid"
        ],
        default="occupancy_grid",
        help="The voxelizer factory to be used (default=occupancy_grid)"
    )

    parser.add_argument(
        "--grid_shape",
        type=lambda x: tuple(map(int, x.split(","))),
        default="32,32,32",
        help="The dimensionality of the voxel grid (default=(32, 32, 32)"
    )
    parser.add_argument(
        "--save_voxels_to",
        default=None,
        help="Path to save the voxelised input to the network"
    )


def add_dataset_parameters(parser):
    parser.add_argument(
        "--dataset_type",
        default="shapenet_v1",
        choices=[
            "shapenet",
            "dynamic_faust",
        ],
        help="The type of the dataset type to be used"
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
        "--random_subset",
        type=float,
        default=1.0,
        help="Percentage of dataset to be used for evaluation"
    )
    parser.add_argument(
        "--val_random_subset",
        type=float,
        default=1.0,
        help="Percentage of dataset to be used for validation"
    )
