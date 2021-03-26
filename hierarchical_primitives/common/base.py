from .dataset import dataset_factory
from .model_factory import DatasetBuilder
from .parse_splits import ShapeNetSplitsBuilder, DynamicFaustSplitsBuilder, \
    CSVSplitsBuilder

from torch.utils.data import DataLoader


def splits_factory(dataset_type):
    return {
        "shapenet": ShapeNetSplitsBuilder,
        "dynamic_faust": DynamicFaustSplitsBuilder,
    }[dataset_type]


def build_dataset(
    config,
    model_tags,
    category_tags,
    keep_splits,
    random_subset=1.0,
    cache_size=0
):
    dataset_directory = config["data"]["dataset_directory"]
    dataset_type = config["data"]["dataset_type"]
    train_test_splits_file = config["data"]["splits_file"]
    # Create a dataset instance to generate the samples for training
    dataset = dataset_factory(
        config["data"]["dataset_factory"],
        (DatasetBuilder(config)
            .with_dataset(dataset_type)
            .filter_train_test(
                splits_factory(dataset_type)(train_test_splits_file),
                keep_splits
             )
            .filter_category_tags(category_tags)
            .filter_tags(model_tags)
            .random_subset(random_subset)
            .build(dataset_directory))
    )
    return dataset


def build_dataloader(
    config,
    model_tags,
    category_tags,
    split,
    batch_size,
    n_processes,
    random_subset=1.0,
    cache_size=0
):
    dataset = build_dataset(
        config,
        model_tags,
        category_tags,
        split,
        random_subset=random_subset,
        cache_size=cache_size,
    )
    print("Dataset has {} elements".format(len(dataset)))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_processes,
        shuffle=True
    )

    return dataloader
