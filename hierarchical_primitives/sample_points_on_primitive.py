import torch

from .equal_distance_sampler_sq import EqualDistanceSamplerSQ
from .primitives import fexp


def sample_uniformly_on_sq(
    alphas,
    epsilons,
    sq_sampler
):
    """
    Given the sampling steps in the parametric space, we want to ge the actual
    3D points on the sq.

    Arguments:
    ----------
        alphas: Tensor with size BxMx3, containing the size along each
                axis for the M primitives
        epsilons: Tensor with size BxMx2, containing the shape along the
                  latitude and the longitude for the M primitives

    Returns:
    ---------
        P: Tensor of size BxMxSx3 that contains S sampled points from the
           surface of each primitive
        N: Tensor of size BxMxSx3 that contains the normals of the S sampled
           points from the surface of each primitive
    """
    # Allocate memory to store the sampling steps
    B = alphas.shape[0]  # batch size
    M = alphas.shape[1]  # number of primitives
    S = sq_sampler.n_samples

    etas, omegas = sq_sampler.sample_on_batch(
        alphas.detach().cpu().numpy(),
        epsilons.detach().cpu().numpy()
    )
    # Make sure we don't get nan for gradients
    etas[etas == 0] += 1e-6
    omegas[omegas == 0] += 1e-6

    # Move to tensors
    etas = alphas.new_tensor(etas)
    omegas = alphas.new_tensor(omegas)

    # Make sure that all tensors have the right shape
    a1 = alphas[:, :, 0].unsqueeze(-1)  # size BxMx1
    a2 = alphas[:, :, 1].unsqueeze(-1)  # size BxMx1
    a3 = alphas[:, :, 2].unsqueeze(-1)  # size BxMx1
    e1 = epsilons[:, :, 0].unsqueeze(-1)  # size BxMx1
    e2 = epsilons[:, :, 1].unsqueeze(-1)  # size BxMx1

    x = a1 * fexp(torch.cos(etas), e1) * fexp(torch.cos(omegas), e2)
    y = a2 * fexp(torch.cos(etas), e1) * fexp(torch.sin(omegas), e2)
    z = a3 * fexp(torch.sin(etas), e1)

    # Make sure we don't get INFs
    # x[torch.abs(x) <= 1e-9] = 1e-9
    # y[torch.abs(y) <= 1e-9] = 1e-9
    # z[torch.abs(z) <= 1e-9] = 1e-9
    x = ((x > 0).float() * 2 - 1) * torch.max(torch.abs(x), x.new_tensor(1e-6))
    y = ((y > 0).float() * 2 - 1) * torch.max(torch.abs(y), x.new_tensor(1e-6))
    z = ((z > 0).float() * 2 - 1) * torch.max(torch.abs(z), x.new_tensor(1e-6))

    # Compute the normals of the SQs
    nx = (torch.cos(etas)**2) * (torch.cos(omegas)**2) / x
    ny = (torch.cos(etas)**2) * (torch.sin(omegas)**2) / y
    nz = (torch.sin(etas)**2) / z

    return torch.stack([x, y, z], -1), torch.stack([nx, ny, nz], -1)


def sample_uniformly_on_cube(alphas, sampler):
    """
    Given the sampling steps in the parametric space, we want to ge the actual
    3D points on the surface of the cube.

    Arguments:
    ----------
        alphas: Tensor with size BxMx3, containing the size along each
                axis for the M primitives

    Returns:
    ---------
        P: Tensor of size BxMxSx3 that contains S sampled points from the
           surface of each primitive
    """
    # TODO: Make sure that this is the proper way to do this!
    # Check the device of the angles and move all the tensors to that device
    device = alphas.device

    # Allocate memory to store the sampling steps
    B = alphas.shape[0]  # batch size
    M = alphas.shape[1]  # number of primitives
    S = sampler.n_samples
    N = S/6

    X_SQ = torch.zeros(B, M, S, 3).to(device)

    for b in range(B):
        for m in range(M):
            x_max = alphas[b, m, 0]
            y_max = alphas[b, m, 1]
            z_max = alphas[b, m, 2]
            x_min = -x_max
            y_min = -y_max
            z_min = -z_max

            X_SQ[b, m] = torch.stack([
                torch.stack([
                    torch.ones((N, 1)).to(device)*x_min,
                    torch.rand(N, 1).to(device)*(y_max-y_min) + y_min,
                    torch.rand(N, 1).to(device)*(z_max-z_min) + z_min
                ], dim=-1).squeeze(),
                torch.stack([
                    torch.ones((N, 1)).to(device)*x_max,
                    torch.rand(N, 1).to(device)*(y_max-y_min) + y_min,
                    torch.rand(N, 1).to(device)*(z_max-z_min) + z_min
                ], dim=-1).squeeze(),
                torch.stack([
                    torch.rand(N, 1).to(device)*(x_max-x_min) + x_min,
                    torch.ones((N, 1)).to(device)*y_min,
                    torch.rand(N, 1).to(device)*(z_max-z_min) + z_min
                ], dim=-1).squeeze(),
                torch.stack([
                    torch.rand(N, 1).to(device)*(x_max-x_min) + x_min,
                    torch.ones((N, 1)).to(device)*y_max,
                    torch.rand(N, 1).to(device)*(z_max-z_min) + z_min
                ], dim=-1).squeeze(),
                torch.stack([
                    torch.rand(N, 1).to(device)*(x_max-x_min) + x_min,
                    torch.rand(N, 1).to(device)*(y_max-y_min) + y_min,
                    torch.ones((N, 1)).to(device)*z_min,
                ], dim=-1).squeeze(),
                torch.stack([
                    torch.rand(N, 1).to(device)*(x_max-x_min) + x_min,
                    torch.rand(N, 1).to(device)*(y_max-y_min) + y_min,
                    torch.ones((N, 1)).to(device)*z_max,
                ], dim=-1).squeeze()
            ]).view(-1, 3)

    normals = X_SQ.new_zeros(X_SQ.shape)
    normals[:, :, 0*N:1*N, 0] = -1
    normals[:, :, 1*N:2*N, 0] = 1
    normals[:, :, 2*N:3*N, 1] = -1
    normals[:, :, 3*N:4*N, 1] = 1
    normals[:, :, 4*N:5*N, 2] = -1
    normals[:, :, 5*N:6*N, 2] = 1

    # make sure that X_SQ has the expected shape
    assert X_SQ.shape == (B, M, S, 3)
    return X_SQ, normals


class CuboidSampler(object):
    def __init__(self, n_samples):
        self._n_samples = n_samples

    @property
    def n_samples(self):
        return self._n_samples

    def sample(self, a1, a2, a3):
        pass

    def sample_on_batch(self, shapes, epsilons):
        pass


class PrimitiveSampler(object):
    def __init__(self, n_samples):
        self._n_samples = n_samples

    @property
    def n_samples(self):
        return self._n_samples

    def sample_points_on_primitive(self, use_cuboid, alphas, epsilons):
        if not use_cuboid:
            return sample_uniformly_on_sq(
                alphas,
                epsilons,
                EqualDistanceSamplerSQ(self._n_samples)
            )
        else:
            return sample_uniformly_on_cube(
                alphas,
                CuboidSampler(self._n_samples)
            )
