from torch.utils.data import DataLoader

from evaluate import MeshEvaluator

from hierarchical_primitives.common.dataset import PointsAndLabels, PointsOnMesh, \
    DatasetCollection
from hierarchical_primitives.common.model_factory import DatasetBuilder
from hierarchical_primitives.primitives import \
    compute_accuracy_and_recall_from_primitive_params


def report_metrics(
    prim_params,
    config,
    dataset_type,
    model_tags,
    dataset_directory
):
    data_source = (DatasetBuilder(config)
        .with_dataset(dataset_type)
        .filter_tags(model_tags)
        .build(dataset_directory))
    in_bbox = PointsAndLabels(data_source)
    on_surface = PointsOnMesh(data_source)
    dataset = DatasetCollection(in_bbox, on_surface)
    for y_target in DataLoader(dataset, batch_size=1, num_workers=4):
        accuracy, positive_accuracy =\
            compute_accuracy_and_recall_from_primitive_params(
                y_target[:-1], prim_params)
        print(("accuracy:%.7f - recall:%.7f") % (accuracy, positive_accuracy))

        metrics = MeshEvaluator().eval_mesh_with_primitive_params(
            prim_params,
            y_target[0],
            y_target[1].squeeze(-1),
            y_target[2].squeeze(-1),
            y_target[3][..., :3],
            config
        )
        print(("chamfer-l1: %.7f - iou %.7f") % (
            metrics["ch_l1"], metrics["iou"]
        ))
