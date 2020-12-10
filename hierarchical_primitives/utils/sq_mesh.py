import numpy as np
import torch

import trimesh

from ..primitives import transform_to_world_coordinates_system, \
    transform_to_primitives_centric_system, inside_outside_function,\
    quaternions_to_rotation_matrices


def single_sq_mesh(alpha, epsilon, translation, rotation):
    """Create mesh for a superquadric with the provided primitive
    configuration.

    Arguments
    ---------
        alpha: Array of 3 sizes, along each axis
        epsilon: Array of 2 shapes, along each a
        translation: Array of 3 dimensional center
        rotation: Array of size 3x3 containing the rotations
    """
    def fexp(x, p):
        return np.sign(x)*(np.abs(x)**p)

    def sq_surface(a1, a2, a3, e1, e2, eta, omega):
        x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
        y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
        z = a3 * fexp(np.sin(eta), e1)
        return x, y, z

    # triangulate the sphere to be used with the SQs
    eta = np.linspace(-np.pi/2, np.pi/2, 100, endpoint=True)
    omega = np.linspace(-np.pi, np.pi, 100, endpoint=True)
    triangles = []
    for o1, o2 in zip(np.roll(omega, 1), omega):
        triangles.extend([
            (eta[0], 0),
            (eta[1], o2),
            (eta[1], o1),
        ])
    for e in range(1, len(eta)-2):
        for o1, o2 in zip(np.roll(omega, 1), omega):
            triangles.extend([
                (eta[e], o1),
                (eta[e+1], o2),
                (eta[e+1], o1),
                (eta[e], o1),
                (eta[e], o2),
                (eta[e+1], o2),
            ])
    for o1, o2 in zip(np.roll(omega, 1), omega):
        triangles.extend([
            (eta[-1], 0),
            (eta[-2], o1),
            (eta[-2], o2),
        ])
    triangles = np.array(triangles)
    eta, omega = triangles[:, 0], triangles[:, 1]

    # collect the pretriangulated vertices of each SQ
    vertices = []
    a, e, t, R = list(map(
        np.asarray,
        [alpha, epsilon, translation, rotation]
    ))
    M, _ = a.shape  # number of superquadrics
    assert R.shape == (M, 3, 3)
    assert t.shape == (M, 3)
    for i in range(M):
        a1, a2, a3 = a[i]
        e1, e2 = e[i]
        x, y, z = sq_surface(a1, a2, a3, e1, e2, eta, omega)
        # Get points on the surface of each SQ
        V = np.stack([x, y, z], axis=-1)
        V = R[i].T.dot(V.T).T + t[i].reshape(1, 3)
        vertices.append(V)

    # Finalize the mesh
    vertices = np.vstack(vertices)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def sq_meshes(primitive_params, indices=None):
    translations = primitive_params.translations_r
    rotations = primitive_params.rotations_r
    Rs = quaternions_to_rotation_matrices(
        primitive_params.rotations.view(-1, 4)
    ).view(1, -1, 3, 3)
    alphas = primitive_params.sizes_r
    epsilons = primitive_params.shapes_r
    probs = primitive_params.probs

    M = primitive_params.n_primitives
    if indices is None:
        indices = range(M)

    return [
       single_sq_mesh(
            alphas[:, i, :].cpu().detach().numpy(),
            epsilons[:, i, :].cpu().detach().numpy(),
            translations[:, i, :].cpu().detach().numpy(),
            Rs[:, i].cpu().detach().numpy()
        ) for i in indices
    ]
    


def sq_mesh_from_primitive_params(primitive_params, S=100000, normals=False,
                                  prim_indices=False):
    translations = primitive_params.translations_r
    rotations = primitive_params.rotations_r
    Rs = quaternions_to_rotation_matrices(
        primitive_params.rotations.view(-1, 4)
    ).view(1, -1, 3, 3)
    alphas = primitive_params.sizes_r
    epsilons = primitive_params.shapes_r
    probs = primitive_params.probs
    M = primitive_params.n_primitives
    meshes = sq_meshes(primitive_params)
    areas = np.array([m.area for m in meshes])
    areas /= areas.sum()

    P = np.empty((0, 3))
    N = np.empty((0, 3))
    I = np.empty((0, 1))
    cnt = 0
    while cnt < S:
        n_points = np.random.multinomial(S, areas)
        for i in range(M):
            points, faces = trimesh.sample.sample_surface(meshes[i], n_points[i])
            if len(points) == 0:
                continue
            # Filter anything that is in an SQ other than i
            X_SQ = torch.from_numpy(points)
            X_SQ = X_SQ.unsqueeze(0).float().to(alphas.device)

            # Transform the points on the SQs to the other SQs
            X_SQ_transformed = transform_to_primitives_centric_system(
                X_SQ, translations, rotations
            )
            # Compute the inside outside function for every point on every
            # primitive to every other primitive
            F = inside_outside_function(
                X_SQ_transformed, alphas.detach(), epsilons.detach()
            )
            F[:, :, i] = 2.0
            mask = (F>1).all(dim=-1)[0].cpu().numpy().astype(bool)
            points = points[mask]
            P = np.vstack([P, points])
            N = np.vstack([N, meshes[i].face_normals[faces[mask]]])
            I = np.vstack([I, np.ones((len(points), 1))*i])
            cnt += len(points)
    idxs = np.random.choice(len(P), S, replace=False)
    retval = (P[idxs],)
    if normals:
        retval += (N[idxs],)
    if prim_indices:
        retval += (I[idxs].astype(int),)

    if len(retval) == 1:
        retval = retval[0]

    return retval
