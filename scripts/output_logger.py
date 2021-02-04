

from hierarchical_primitives.utils.progbar import Progbar
from hierarchical_primitives.utils.stats_logger import StatsLogger


_REGULARIZERS = {
    "regularizers.sparsity": "sparsity",
    "regularizers.entropy_bernoulli": "entropy",
    "regularizers.parsimony": "parsimony",
    "regularizers.overlapping": "overl",
    "regularizers.proximity": "proximity",
    "regularizers.siblings_proximity": "siblings_prox",
    "regularizers.overlapping_on_depths": "part-overl",
    "regularizers.volumes": "vol"
}

_LOSSES = {
    "losses.cvrg": "cvrg",
    "losses.cnst": "cnst",
    "losses.partition": "part",
    "losses.hierarchy": "hier",
    "losses.vae": "vae",
    "losses.coverage": "cvrg",
    "losses.fit": "fit",
    "losses.qos": "qos",
    "losses.prox": "prox",
    "losses.reconstruction": "rec",
    "losses.kinematic": "kin",
    "losses.structure": "str",
    "losses.fit_parts": "fit_parts",
    "losses.fit_shape": "fit_shape"
}

_METRICS = {
    "metrics.positive_accuracy": "pos_acc",
    "metrics.accuracy": "acc",
    "metrics.iou": "iou",
    "metrics.chl1": "chl1",
    "metrics.exp_n_prims": "exp_n_prims"
}


class LossLogger(object):
    def __init__(self, epochs, steps_per_epoch, keys, messages,
                 stats_filepath, prefix="Epoch {}/{}"):
        self._prefix = prefix
        self._epochs = epochs
        self._steps_per_epoch = steps_per_epoch
        self._keys = keys
        self._messages = messages

        self._stats = StatsLogger.instance()
        self._stats_fp = open(stats_filepath, "a")
        self._epoch = 0

    def new_epoch(self, e):
        print(self._prefix.format(e, self._epochs))
        self._progbar = Progbar(self._steps_per_epoch)
        self._stats.clear()
        self._epoch = e

    def new_batch(self, batch_index, batch_loss):
        stats = [("loss", batch_loss)] + [
            (m, self._stats[k]) for k, m in zip(self._keys, self._messages)
            if k in self._stats
        ]
        self._progbar.update(batch_index+1, stats)
        self._save_to_file(self._epoch, batch_index, stats)
        self._stats.clear()

    def _save_to_file(self, epoch, batch, stats):
        if epoch == 0 and batch == 0:
            print(
                " ".join(["epoch", "batch"] + [s[0] for s in stats]),
                file=self._stats_fp,
            )
        print(
            " ".join([str(epoch), str(batch)] + [str(s[1]) for s in stats]),
            file=self._stats_fp,
        )
        self._stats_fp.flush()


def get_logger(epochs, steps_per_epoch, stats_filepath, prefix="Epoch {}/{}"):
    keys, messages = list(zip(*(
        list(_LOSSES.items()) +
        list(_REGULARIZERS.items()) +
        list(_METRICS.items())
    )))

    return LossLogger(
        epochs, steps_per_epoch, keys, messages, stats_filepath, prefix=prefix
    )
