"""Script used to train the network that will be used to learn the parameters
of M primitives given an image.
"""
import argparse

import json
import random
import os
import string
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from arguments import add_dataset_parameters
from training_utils import get_loss_options, get_regularizer_options, \
    load_config

from hierarchical_primitives.common.base import build_dataloader, build_dataset
from hierarchical_primitives.sample_points_on_primitive import PrimitiveSampler
from hierarchical_primitives.networks.base import build_network, train_on_batch, \
    optimizer_factory, validate_on_batch
from hierarchical_primitives.losses import get_loss

from output_logger import get_logger


def set_num_threads(nt):
    nt = str(nt)
    os.environ["OPENBLAS_NUM_THREDS"] = nt
    os.environ["NUMEXPR_NUM_THREDS"] = nt
    os.environ["OMP_NUM_THREDS"] = nt
    os.environ["MKL_NUM_THREDS"] = nt


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def yield_infinite(iterable):
    while True:
        for item in iterable:
            yield item


def lr_schedule(optimizer, current_epoch, config):
    def inner(epoch):
        for i, e in enumerate(reductions):
            if epoch < e:
                return init_lr*factor**(-i)
        return init_lr*factor**(-len(reductions))

    init_lr = config["loss"].get("lr", 1e-3)
    factor = config["loss"].get("lr_factor", 1.0)
    reductions = config["loss"].get("lr_epochs", [500,1000,1500])

    for param_group in optimizer.param_groups:
        param_group['lr'] = inner(current_epoch)

    return optimizer


def save_experiment_params(args, experiment_tag, directory):
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    git_dir = os.path.dirname(os.path.realpath(__file__))
    git_head_hash = "foo"
    try:
        git_head_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip()
    except subprocess.CalledProcessError:
        # Keep the current working directory to move back in a bit
        cwd = os.getcwd()
        os.chdir(git_dir)
        git_head_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip()
        os.chdir(cwd)
    params["git-commit"] = str(git_head_hash)
    params["experiment_tag"] = experiment_tag
    for k, v in list(params.items()):
        if v == "":
            params[k] = None
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a network to predict primitives"
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
        help=("The path to a previously trained model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--run_on_gpu",
        action="store_true",
        help="Use GPU"
    )
    parser.add_argument(
        "--probs_only",
        action="store_true",
        help="Optimize only using the probabilities"
    )
    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--credentials",
        default=os.path.join(os.path.dirname(__file__), ".credentials"),
        help="The credentials file for the Google API"
    )

    parser.add_argument(
        "--cache_size",
        type=int,
        default=0,
        help="The batch provider cache size"
    )

    parser.add_argument(
        "--n_processes",
        type=int,
        default=8,
        help="The numper of processed spawned by the batch provider"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )

    add_dataset_parameters(parser)
    # Parameters related to the loss function and the loss weights
    args = parser.parse_args(argv)
    set_num_threads(1)

    if args.run_on_gpu:  # and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Get the parameters and their ordering for the spreadsheet
    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in %s" % (experiment_tag,))

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    config = load_config(args.config_file)
    # Build the network architecture to be used for training
    network = build_network(args.config_file, args.weight_file, device=device)
    network.train()
    loss_options = get_loss_options(config)

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(config, network)

    # Create an object that will sample points in equal distances on the
    # surface of the primitive
    n_points_from_sq_mesh = config["data"].get("n_points_from_sq_mesh", 200)
    sampler = PrimitiveSampler(n_points_from_sq_mesh)

    # Instantiate a dataloader to generate the samples for training
    dataloader = build_dataloader(
        config,
        args.model_tags,
        args.category_tags,
        config["data"].get("train_split", ["train", "val"]),
        config["loss"].get("batch_size", 32),
        args.n_processes,
        cache_size=args.cache_size,
    )
    # Instantiate a dataloader to generate the samples for validation
    val_dataset = build_dataset(
        config,
        [],
        [],
        config["data"].get("test_split", ["test"]),
        random_subset=args.val_random_subset,
        cache_size=args.cache_size
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["data"].get("validation_batch_size", 8),
        num_workers=args.n_processes,
        shuffle=True
    )

    epochs = config["loss"].get("epochs", 150)
    steps_per_epoch = config["loss"].get("steps_per_epoch", 500)
    # Create logger to keep track of the training statistics
    logger = get_logger(
        epochs,
        steps_per_epoch,
        os.path.join(experiment_directory, "train.txt")
    )

    # Create logger to keep track of the validation statistics
    val_every = config["data"].get("validation_every", 1000)
    val_logger = get_logger(
        epochs // val_every,
        len(val_dataset),
        os.path.join(experiment_directory, "val.txt"),
        prefix="Validation Epoch"
    )
    # Counter to keep track of the validation epochs
    val_epochs = 0

    save_every = config["data"].get("save_every", 5)

    for i in range(epochs):
        logger.new_epoch(i)
        for b, sample in zip(list(range(steps_per_epoch)), yield_infinite(dataloader)):
            X = sample[0].to(device)
            y_target = [yi.to(device) for yi in sample[1:]]
            if len(y_target) == 1:
                y_target = y_target[0]

            # Train on batch
            reg_terms, reg_options = get_regularizer_options(config, i)
            loss_options.update(reg_options)
            batch_loss, preds = train_on_batch(
                network,
                lr_schedule(optimizer, i, config),
                get_loss(
                    config["loss_type"],
                    reg_terms,
                    sampler,
                    loss_options
                ),
                X,
                y_target,
                i
            )

            logger.new_batch(b, batch_loss)
        if i %  save_every == 0:
            torch.save(
                network.state_dict(),
                os.path.join(
                    experiment_directory,
                    "model_%d" % (i + args.continue_from_epoch,)
                )
            )

        # Perform validation every validation every epochs
        if i %  val_every == 0 and i>0:
            val_logger.new_epoch(val_epochs)
            total = 0
            for sample in val_dataloader:
                X = sample[0].to(device)
                y_target = [yi.to(device) for yi in sample[1:]]
                if len(y_target) == 1:
                    y_target = y_target[0]
                val_l, preds = validate_on_batch(
                    network,
                    get_loss(
                        config["loss_type"],
                        reg_terms,
                        sampler,
                        loss_options
                    ),
                    X,
                    y_target
                )
                total += X.shape[0]
                val_logger.new_batch(total, val_l)
            # Increment counter by one
            val_epochs += 1

    print("Saved statistics in %s" % (experiment_tag,))


if __name__ == "__main__":
    main(sys.argv[1:])
