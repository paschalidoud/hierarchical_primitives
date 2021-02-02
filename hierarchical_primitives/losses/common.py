import numpy as np


def loss_weights_on_depths(loss_type, start_value, end_value, N):
    start_value = float(start_value)
    end_value = float(end_value)

    if loss_type == "equal":
        return [start_value]*N
    elif loss_type == "linear":
       return np.linspace(start_value, end_value, N).tolist()


def clustering_distance(clustering_type, dists, precision_matrix=None):
    if clustering_distance == "euclidean":
        return torch.sqrt(torch.einsum(
            "bnmj,bmjk,bnmk->bnm",
            [dists, percision_matrix, dists]
        ))
    elif clustering_distance == "mahalanobis":
        return dists.sum(-1)
