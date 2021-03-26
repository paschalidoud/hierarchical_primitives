#!/usr/bin/env python3
"""Script used to evaluate the predicted mesh and save the results in an sqlite
database."""

import argparse
from hashlib import sha256

import os
import sys
import time

import mysql.connector
from mysql.connector import errorcode
import numpy as np
import torch

from arguments import add_dataset_parameters
from evaluate import MeshEvaluator
from training_utils import get_loss_options, load_config

from hierarchical_primitives.common.base import build_dataset
from hierarchical_primitives.common.dataset import DatasetWithTags
from hierarchical_primitives.networks.base import build_network
from hierarchical_primitives.utils.filter_sqs import filter_primitives, \
    primitive_parameters_from_indices, qos_less, volume_larger
from hierarchical_primitives.utils.progbar import Progbar


def hash_file(filepath):
    h = sha256()
    with open(filepath, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def get_db(dbhost):
    CREATE_TABLE = """CREATE TABLE results (
        model_tag VARCHAR(255),
        weight_file VARCHAR(255),
        config VARCHAR(255),
        run INT,
        epoch INT,
        ch_l1 REAL,
        iou REAL,
        subset REAL,
        CONSTRAINT pk PRIMARY KEY (model_tag, weight_file, config, run)
    );
    """
    conn = mysql.connector.connect(
        user="evalscript", password=os.getenv("DBPASS"),
        host=dbhost,
        database="shapenet_evaluation"
    )
    cursor = conn.cursor()
    try:
        cursor.execute(CREATE_TABLE)
    except mysql.connector.Error as err:
        if err.errno != errorcode.ER_TABLE_EXISTS_ERROR:
            raise
    conn.commit()

    return conn


def start_run(conn, model_tag, weight_file, config, run):
    tags = None
    while True:
        cursor = conn.cursor()
        cursor.execute(
            ("SELECT model_tag FROM results "
             "WHERE weight_file=%s AND config=%s AND run=%s"),
            (weight_file, config, run)
        )
        tags = set(t[0] for t in cursor)
        try:
            cursor.execute(
                ("INSERT INTO results (model_tag, weight_file, config, run) "
                 "VALUES (%s, %s, %s, %s)"),
                (model_tag, weight_file, config, run)
            )
            conn.commit()
            return True, tags
        except mysql.connector.IntegrityError:
            return False, tags
        except mysql.connector.Error as err:
            conn.rollback()
            print(err)
            time.sleep(1)
        finally:
            cursor.close()


def fill_run(conn, model_tag, weight_file, config, run, epoch, stats, subset):
    while True:
        try:
            cursor = conn.cursor()
            cursor.execute(
                ("UPDATE results "
                 "SET epoch=%s, ch_l1=%s, iou=%s, subset=%s "
                 "WHERE model_tag=%s AND weight_file=%s AND config=%s "
                 "AND run=%s"),
                (epoch, float(stats["chamfer"]), float(stats["iou"]), subset,
                 model_tag, weight_file, config, run)
            )
            conn.commit()
            break
        except mysql.connector.Error as err:
            conn.rollback()
            print(err)
            time.sleep(1)
        finally:
            cursor.close()


def get_started_tags(conn, weight_file, config, run):
    cursor = conn.cursor()
    cursor.execute(
        ("SELECT model_tag FROM results "
         "WHERE weight_file=%s AND config=%s AND run=%s"),
        (weight_file, config, run)
    )
    return set(t[0] for t in cursor)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Do the forward pass and estimate a set of primitives"
    )
    parser.add_argument(
        "dataset_directory",
        help="Path to the directory containing the dataset"
    )
    parser.add_argument(
        "train_test_splits_file",
        help="Path to the train-test splits file"
    )
    parser.add_argument(
        "output_db",
        help="Save the results in this sqlite database"
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
        "--run",
        type=int,
        default=0,
        help="Run id to be able to evaluate many times the same model"
    )

    add_dataset_parameters(parser)
    args = parser.parse_args(argv)

    # Get the database connection
    conn = get_db(args.output_db)

    # Build the network architecture to be used for training
    config = load_config(args.config_file)
    network = build_network(args.config_file, args.weight_file, device="cpu")
    network.eval()

    eval_config = config.get("eval", {})
    config_hash = hash_file(args.config_file)
    captured_at_epoch = (
        -1 if args.weight_file is None else
        int(args.weight_file.split("/")[-1].split("_")[-1])
    )

    dataset = build_dataset(
        config,
        args.dataset_directory,
        args.dataset_type,
        args.train_test_splits_file,
        args.model_tags,
        args.category_tags,
        config["data"].get("test_split", ["test"]) if not args.eval_on_train else ["train"],
        random_subset=args.random_subset
    )
    dataset = DatasetWithTags(dataset)

    prog = Progbar(len(dataset))
    tagset = get_started_tags(
        conn,
        args.weight_file or "",
        config_hash,
        args.run
    )
    i = 0
    for sample in dataset:
        if sample[-1] in tagset:
            continue
        start, tagset = start_run(
            conn,
            sample[-1],
            args.weight_file or "",
            config_hash,
            args.run
        )
        if not start:
            continue

        X = sample[0].unsqueeze(0)
        y_target = [yi.unsqueeze(0) for yi in sample[1:-1]]

        # Do the forward pass and estimate the primitive parameters
        y_hat = network(X)
        if (
            "qos_threshold" in eval_config or
            "vol_threshold" in eval_config
        ):
            primitive_indices = filter_primitives(
                y_hat.fit,
                qos_less(float(eval_config.get("qos_threshold", 1))),
                volume_larger(float(eval_config.get("vol_threshold", 0)))
            )
            if len(primitive_indices) == 0:
                continue
            y_hat = primitive_parameters_from_indices(
                y_hat.fit,
                primitive_indices
            )

        metrics = MeshEvaluator().eval_mesh_with_primitive_params(
            y_hat,
            y_target[0],
            y_target[1].squeeze(-1),
            y_target[2].squeeze(-1),
            get_loss_options(config)
        )
        fill_run(
            conn,
            sample[-1],
            args.weight_file or "",
            config_hash,
            args.run,
            captured_at_epoch,
            metrics,
            args.random_subset
        )

        # Update progress bar
        prog.update(i+1)
        i += 1
    prog.update(len(dataset))


if __name__ == "__main__":
    main(sys.argv[1:])
