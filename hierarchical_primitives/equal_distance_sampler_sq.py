import os
import pickle

import numpy as np

from .fast_sampler import fast_sample, fast_sample_on_batch


class EqualDistanceSamplerSQ(object):

    def __init__(self, n_samples):
        self._n_samples = n_samples

    @property
    def n_samples(self):
        return self._n_samples

    def sample(self, **kwargs):
        return fast_sample(
            a1=kwargs.get("a1", 0.25),
            a2=kwargs.get("a2", 0.25),
            a3=kwargs.get("a3", 0.25),
            e1=kwargs.get("eps1", 0.25),
            e2=kwargs.get("eps2", 0.25),
            N=self.n_samples
        )

    def sample_on_batch(self, shapes, epsilons):
        return fast_sample_on_batch(
            shapes,
            epsilons,
            self.n_samples
        )


def fexp(x, p):
    return np.sign(x)*(np.abs(x)**p)


def sq_surface(a1, a2, a3, e1, e2, eta, omega):
    x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
    y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
    z = a3 * fexp(np.sin(eta), e1)
    return x, y, z

def bending(x, y, z, a, gamma):
    b = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2) * np.cos(a-b)
    inv_k = gamma / z
    R = inv_k - (inv_k - r) * np.cos(gamma)
    x = x + (R-r)*np.cos(a)
    y = y + (R-r)*np.sin(a)
    z = (inv_k - r)*np.sin(gamma)
    return x, y, z

def bending_inv(x, y, z, a, gamma):
    R = np.sqrt(x**2 + y**2)
    t1 = np.arctan2(y, x)
    R = R * np.cos(a - t1)
    inv_k = gamma / z
    t2 = inv_k - R
    r = inv_k - np.sqrt(z**2 + t2**2)
    gamma = np.arctan2(z, t2)

    x = x - (R-r)*np.cos(a)
    y = y - (R-r)*np.sin(a)
    z = inv_k * gamma
    return x, y, z


def visualize_points_on_sq_mesh(e, **kwargs):
    print(kwargs)
    e1 = kwargs.get("eps1", 0.25)
    e2 = kwargs.get("eps2", 0.25)
    a1 = kwargs.get("a1", 0.25)
    a2 = kwargs.get("a2", 0.25)
    a3 = kwargs.get("a3", 0.25)
    Kx = kwargs.get("Kx", 0.0)
    Ky = kwargs.get("Ky", 0.0)
    a = kwargs.get("a", 0.1)
    gamma = kwargs.get("gamma", 0.1)

    shapes = np.array([[[a1, a2, a3]]], dtype=np.float32)
    epsilons = np.array([[[e1, e2]]], dtype=np.float32)
    etas, omegas = e.sample_on_batch(shapes, epsilons)
    x, y, z = sq_surface(a1, a2, a3, e1, e2, etas.ravel(), omegas.ravel())

    # Apply tapering
    # fx = Kx * z / a3 + 1
    # fy = Ky * z / a3 + 1
    # fz = 1

    # x = x * fx
    # y = y * fy
    # z = z * fz
    x, y, z = bending_inv(x, y, z, a, gamma)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    # ax.set_xlim([-1.25, 1.25])
    # ax.set_ylim([-1.25, 1.25])
    # ax.set_zlim([-1.25, 1.25])
    plt.show()


if __name__ == "__main__":
    e = EqualDistanceSamplerSQ(600)
    # etas, omegas = e.sample(**{
    #     'a1': 0.2074118,
    #     'a2': 0.0926611,
    #     'a3': 0.2323654,
    #     'eps1': 0.20715195,
    #     'eps2': 1.6855394
    # })
    visualize_points_on_sq_mesh(e, **{
        #'a1': 0.2074118,
        #'a2': 0.0926611,
        #'a3': 0.2323654,
        'a1': 0.15,
        'a2': 0.15,
        'a3': 0.35,
        'eps1': 0.20715195,
        'eps2': 1.3855394,
        'Kx': 0.0,
        'Ky': 0.0,
        'a': 1.0,
        'gamma': 1.0
        # 'k': 0.01
    })
