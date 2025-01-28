"""Module with functions for the optimization of the
embeddings of the patches using the manopt library."""

import random
from typing import Tuple, Optional
import autograd.numpy as anp
import jax.numpy as jnp
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import numpy as np
import local2global as l2g

from l2g-light.patch.patch import Patch


def double_intersections_nodes(
    patches: list[Patch],
) -> dict[tuple[int, int], list[int]]:
    """TODO: docstring for `double_intersections_nodes`.

    Args:
        patches (list[Patch]): [description]

    Returns:
        dict[tuple[int, int], list[int]]: [description]
    """

    double_intersections = {}
    for i, patch in enumerate(patches):
        for j in range(i + 1, len(patches)):
            double_intersections[(i, j)] = list(
                set(patch.nodes.tolist()).intersection(set(patches[j].nodes.tolist()))
            )
    return double_intersections


def total_loss(
    rotations,
    scales,
    translations,
    nodes,
    patches,
    dim,
    k,
    rand: Optional[bool] = False,
) -> float:
    """TODO: docstring for `total_loss`.

    R: list of orthogonal matrices for embeddings of patches.

    Args:
        rotations ([type]): [description]

        scales ([type]): [description]

        translations ([type]): [description]

        nodes ([type]): [description]

        patches ([type]): [description]

        dim (int): [description]

        k (int): [description]

        rand (Optional[bool]): [description], default is False.

    Returns:
        float: [description]
    """

    l = 0
    fij = {}

    for i, p in enumerate(patches):
        for j, q in enumerate(patches[i + 1 :]):
            if rand:
                for n in random.sample(
                    nodes[i, j + i + 1], min(k * dim + 1, len(nodes[i, i + j + 1]))
                ):
                    theta1 = (
                        scales[i] * p.get_coordinate(n) @ rotations[i] + translations[i]
                    )
                    theta2 = (
                        scales[j + i + 1] * q.get_coordinate(n) @ rotations[j + i + 1]
                        + translations[j + i + 1]
                    )
                    l += np.linalg.norm(theta1 - theta2) ** 2

                    fij[(i, j + 1 + i, n)] = [theta1, theta2]

            else:
                for n in nodes[i, j + i + 1]:
                    theta1 = (
                        scales[i] * rotations[i] @ p.get_coordinate(n) + translations[i]
                    )
                    theta2 = (
                        scales[j + i + 1] * rotations[j + i + 1] @ q.get_coordinate(n)
                        + translations[j + i + 1]
                    )
                    l += np.linalg.norm(theta1 - theta2) ** 2

                    fij[(i, j + 1 + i, n)] = [theta1, theta2]

    return 1 / len(patches) * l, fij


def loss(
    rotations,
    scales,
    translations,
    nodes,
    patches,
    dim,
    k,
    consecutive: Optional[bool] = False,
    random_choice_in_intersections: Optional[bool] = False,
    fij: Optional[bool] = False,
) -> Tuple[float, Optional[list[float]]]:
    """TODO: docstring for `loss`.

    R: list of orthogonal matrices for embeddings of patches.

    Args:

        rotations ([type]): [description]

        scales ([type]): [description]

        translations ([type]): [description]

        nodes ([type]): [description]

        patches ([type]): [description]

        dim (int): [description]

        k (int): [description]

        consecutive (Optional[bool]): [description], default is False.

        random_choice_in_intersections (Optional[bool]): [description], default is False.

        fij (Optional[bool]): [description], default is False.

    Returns:

        float: [description]

        list[float]: [description]

    """

    if consecutive:
        l, f = consecutive_loss(
            rotations,
            scales,
            translations,
            nodes,
            patches,
            dim,
            k,
            rand=random_choice_in_intersections,
        )
        if fij:
            return l, f

        return l, None

    l, f = total_loss(
        rotations,
        scales,
        translations,
        nodes,
        patches,
        dim,
        k,
        rand=random_choice_in_intersections,
    )
    if fij:
        return l, f

    return l, None


def consecutive_loss(
    rotations, scales, translations, nodes, patches, dim, k, rand: Optional[bool] = True
):
    """TODO: docstring for `consecutive_loss`.
    R: list of orthogonal matrices for embeddings of patches.

    Args:

        rotations ([type]): [description]

        scales ([type]): [description]

        translations ([type]): [description]

        nodes ([type]): [description]

        patches ([type]): [description]

        dim (int): [description]

        k (int): [description]

        rand (Optional[bool]): [description], default is True.

    Returns:

            float: [description]

            list[float]: [description]

    """

    l = 0
    fij = {}

    for i in range(len(patches) - 1):
        if rand:
            for n in random.sample(
                nodes[i, i + 1], min(k * dim + 1, len(nodes[i, i + 1]))
            ):
                theta1 = (
                    scales[i] * patches[i].get_coordinate(n) @ rotations[i]
                    + translations[i]
                )
                theta2 = (
                    scales[i + 1] * patches[i + 1].get_coordinate(n) @ rotations[i + 1]
                    + translations[i + 1]
                )
                l += np.linalg.norm(theta1 - theta2) ** 2

                fij[(i, 1 + i, n)] = [theta1, theta2]
        else:
            for n in nodes[i, i + 1]:
                theta1 = (
                    scales[i] * patches[i].get_coordinate(n) @ rotations[i]
                    + translations[i]
                )
                theta2 = (
                    scales[i + 1] * patches[i + 1].get_coordinate(n) @ rotations[i + 1]
                    + translations[i + 1]
                )
                l += np.linalg.norm(theta1 - theta2) ** 2

                fij[(i, 1 + i, n)] = [theta1, theta2]

    return l, fij


# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint does not infer autograd.numpy.linalg.norm so disable no-member
def ANPloss_nodes_consecutive_patches(
    rotations, scales, translations, patches, nodes, dim, k, rand: Optional[bool] = True
):
    """TODO: docstring for `ANPloss_nodes_consecutive_patches`.

    R: list of orthogonal matrices for embeddings of patches.

    Args:

        rotations ([type]): [description]

        scales ([type]): [description]

        translations ([type]): [description]

        patches ([type]): [description]

        nodes ([type]): [description]

        dim (int): [description]

        k (int): [description]

        rand (Optional[bool]): [description], default is True.

    Returns:

        float: [description]

    """
    l = 0
    # fij=dict()
    for i in range(len(patches) - 1):
        if rand:
            for n in random.sample(
                nodes[i, i + 1], min(k * dim + 1, len(nodes[i, i + 1]))
            ):
                theta1 = (
                    scales[i] * patches[i].get_coordinate(n) @ rotations[i]
                    + translations[i]
                )
                theta2 = (
                    scales[i + 1] * patches[i + 1].get_coordinate(n) @ rotations[i + 1]
                    + translations[i + 1]
                )
                l += anp.linalg.norm(theta1 - theta2) ** 2
        else:
            for n in nodes[i, i + 1]:
                theta1 = (
                    scales[i] * patches[i].get_coordinate(n) @ rotations[i]
                    + translations[i]
                )
                theta2 = (
                    scales[i + 1] * patches[i + 1].get_coordinate(n) @ rotations[i + 1]
                    + translations[i + 1]
                )
                l += anp.linalg.norm(theta1 - theta2) ** 2

    return l  # , fij
# pylint: enable=invalid-name
# pylint: enable=no-member


# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint does not infer autograd.numpy.linalg.norm so disable no-member
def ANPloss_nodes(
    rotations, scales, translations, patches, nodes, dim, k, rand: Optional[bool] = True
):
    """TODO: docstring for `ANPloss_nodes`.
    R: list of orthogonal matrices for embeddings of patches.

    Args:

        rotations ([type]): [description]

        scales ([type]): [description]

        translations ([type]): [description]

        patches ([type]): [description]

        nodes ([type]): [description]

        dim (int): [description]

        k (int): [description]

        rand (Optional[bool]): [description], default is True.

    Returns:

        float: [description]

    """
    l = 0
    # fij=dict()

    for i, p in enumerate(patches):
        for j, q in enumerate(patches[i + 1 :]):
            if rand:
                for n in random.sample(
                    nodes[i, j + i + 1], min(k * dim + 1, len(nodes[i, j + 1 + i]))
                ):
                    theta1 = (
                        scales[i] * p.get_coordinate(n) @ rotations[i] + translations[i]
                    )
                    theta2 = (
                        scales[j + i + 1] * q.get_coordinate(n) @ rotations[j + i + 1]
                        + translations[j + i + 1]
                    )
                    l += anp.linalg.norm(theta1 - theta2) ** 2

                    # fij[(i, j+1+i, n)]=[theta1, theta2]

            else:
                for n in nodes[i, j + i + 1]:
                    theta1 = (
                        scales[i] * rotations[i] @ p.get_coordinate(n) + translations[i]
                    )
                    theta2 = (
                        scales[j + i + 1] * rotations[j + i + 1] @ q.get_coordinate(n)
                        + translations[j + i + 1]
                    )
                    l += anp.linalg.norm(theta1 - theta2) ** 2

                    # fij[(i, j+1+i, n)]=[theta1, theta2]

    return 1 / len(patches) * l  # fij
# pylint enable=invalid-name
# pylint enable=no-member

# pylint: disable=no-member
# pylint does not infer autograd.numpy.random.seed so disable no-member
def optimization(
    patches,
    nodes,
    k,
    consecutive: Optional[bool] = True,
    random_choice: Optional[bool] = True,
):
    """TODO: docstring for `optimization`.

    Args:

        patches ([type]): [description]

        nodes ([type]): [description]

        k ([type]): [description]

        consecutive (Optional[bool]): [description], default is True.

        random_choice (Optional[bool]): [description], default is True.

    Returns:

        [type]: [description]

        [type]: [description]

    """
    n_patches = len(patches)
    dim = np.shape(patches[0].coordinates)[1]

    anp.random.seed(42)

    od = [pymanopt.manifolds.SpecialOrthogonalGroup(dim) for i in range(n_patches)]
    rd = [pymanopt.manifolds.Euclidean(dim) for i in range(n_patches)]
    r1 = [pymanopt.manifolds.Euclidean(1) for i in range(n_patches)]
    prod = od + rd + r1

    manifold = pymanopt.manifolds.product.Product(prod)

    if consecutive:

        @pymanopt.function.autograd(manifold)
        def cost(*R):
            rs = list(R[:n_patches])
            ts = list(R[n_patches : 2 * n_patches])
            ss = list(R[2 * n_patches :])
            return ANPloss_nodes_consecutive_patches(
                rs, ss, ts, patches, nodes, dim, k, rand=random_choice
            )
    else:

        @pymanopt.function.autograd(manifold)
        def cost(*R):
            rs = list(R[:n_patches])
            ts = list(R[n_patches : 2 * n_patches])
            ss = list(R[2 * n_patches :])
            return ANPloss_nodes(rs, ss, ts, patches, nodes, dim, k, rand=random_choice)

    problem = pymanopt.Problem(manifold, cost)

    optimizer = pymanopt.optimizers.SteepestDescent()
    result = optimizer.run(problem, reuse_line_searcher=True)

    rotations = result.point[:n_patches]

    translations = result.point[n_patches : 2 * n_patches]

    scales = result.point[2 * n_patches :]
    emb_problem = l2g.AlignmentProblem(patches)

    embedding = np.empty((emb_problem.n_nodes, emb_problem.dim))
    for node, patch_list in enumerate(emb_problem.patch_index):
        embedding[node] = np.mean(
            [
                scales[i] * emb_problem.patches[p].get_coordinate(node) @ rotations[i]
                + translations[i]
                for i, p in enumerate(patch_list)
            ],
            axis=0,
        )

    return result, embedding
# pylint enable=no-member

def loss_dictionary(rs, ss, ts, nodes, patches, dim, k):
    """TODO: docstring for `loss_dictionary`.

    Args:

        Rs ([type]): [description]

        ss ([type]): [description]

        ts ([type]): [description]

        nodes ([type]): [description]

        patches ([type]): [description]

        dim (int): [description]

        k (int): [description]

    Returns:

            [type]: [description]

    """
    l = {}
    for i in range(2):
        for j in range(2):
            l[i, j] = loss(
                rs,
                ss,
                ts,
                nodes,
                patches,
                dim,
                k,
                consecutive=i,
                random_choice_in_intersections=j,
                fij=False,
            )
    return l
