import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def get_loss_options(config):
    # Create a dicitionary with the loss options based on the input arguments
    loss_weights = {}
    for k, v in config.items():
        if "loss_weight" in k:
            loss_weights[k] = v

    loss_options = dict(loss_weights=loss_weights)
    # Update the loss options based on the config file
    for k, v in config["loss"].items():
        loss_options[k] = v

    return loss_options


def get_regularizer_options(config, current_epoch):
    def get_weight(w, epoch, current_epoch):
        if current_epoch < epoch:
            return 0.0
        else:
            return w

    # Parse the regularizer and its corresponding weight from the config
    regularizer_terms = []
    regularizer_options = {}
    for r in config.get("regularizers", []):
        regularizer_weight = 0.0
        # Update the regularizer options based on the config file
        for k, v in config[r].items():
            regularizer_options[k] = v
            if "weight" in k:
                regularizer_weight = v
        regularizer_terms.append(
            (r,
             get_weight(
                regularizer_weight,
                config[r].get("enable_regularizer_after_epoch", 0),
                current_epoch
            ))
        )
    return regularizer_terms, regularizer_options


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config
