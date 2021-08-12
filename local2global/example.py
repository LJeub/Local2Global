#! /bin/usr/env python3

"""Generate synthetic test data"""

import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes
from sklearn.cluster import KMeans
from statistics import mean
from collections.abc import Iterable
from local2global import utils as ut
from copy import copy
from os import path
from collections import Counter
import networkx as nx
from pathlib import Path


def generate_data(n_clusters, scale=1.0, std=0.5, max_size=200, min_size=10, dim=2):
    """Generate test data with normally-distributed clusters centered on sphere.

    :param int n_clusters: Number of clusters

    :param float scale: Radius of sphere for cluster centers [default: 1.0]

    :param float std: Standard deviation for cluster points [default: 0.5]

    :param max_size: maximum cluster size [default: 200]

    :param min_size: minimum cluster size [default: 10]

    :param dim: data dimension [default: 2]
    """

    # Random parameters of each cluster
    if dim > 2:
        list_shifts = []
        for it in range(n_clusters):
            x = ut.rg.normal(size=(1, dim))
            x /= np.linalg.norm(x)
            x *= scale
            list_shifts.append(x)
    elif dim == 2:
        list_shifts = [np.array([np.cos(t / n_clusters * 2 * np.pi), np.sin(t / n_clusters * 2 * np.pi)]) * scale for t
                       in range(n_clusters)]
    else:
        raise ValueError("Dimension needs to be >= 2")

    list_var = [std] * n_clusters
    list_sizes = [ut.rg.integers(min_size, max_size) for _ in range(n_clusters)]

    # Make union cluster
    list_of_clusters = [ut.rg.normal(scale=1, size=(s, dim)) * v + shift for shift, v, s in
                        zip(list_shifts, list_var, list_sizes)]
    points = np.vstack(list_of_clusters)
    return points


def Voronoi_patches(points, sample_size=100, min_degree=None, min_overlap=None, min_size=None, eps=1.6,
                    return_graph=False, kmeans=True):
    """
    Create patches for points. Starts by sampling patch centers and assigning points to the nearest
    center and any center that is within `eps` of the nearest center to create patches.
    Patches are then grown by incrementally adding the next closest point
    until the patch degree constraint is satisfied. Finally patches that are smaller than ``min_size``
    are expanded and shortest edges are added to make the patch graph connected if necessary.

    :param np.ndarray points: ndarray of floats of shape (N,d), d dimension embedding.

    :param int sample_size: number of patches splitting the set of N points

    :param int min_degree: minimum patch degree, defaults to ``d+1``

    :param int min_overlap: minimum overlap to consider two patches connected, defaults to ``d+1``

    :param int min_size: minimum patch size, defaults to ``len(points)/sample_size``

    :param float eps: tolerance for expanding initial Voronoi patches

    :param bool return_graph: if True, return patch graph as a networkx Graph

    :param bool kmeans: if True, choose patch centers using kmeans,
        otherwise patch centers are sampled uniformly at random from points.

    :return: list of patches, (patch graph if return_graph==True)
    """
    n, d = points.shape
    if min_size is None:
        min_size = n // sample_size
    if min_degree is None:
        min_degree = d + 1
    if min_overlap is None:
        min_overlap = d + 1

    # Find patch centers
    if kmeans:
        k_means = KMeans(n_clusters=sample_size)
        k_means.fit(points)
        centers = k_means.cluster_centers_
    else:
        sample_mask = ut.rg.choice(len(points), size=sample_size, replace=False)
        centers = points[sample_mask, :]

    # list of node indeces for each patch
    node_lists = [[] for _ in centers]
    patch_index = [[] for _ in range(n)]
    overlaps = [Counter() for _ in range(sample_size)]

    # compute distance to centers
    distances = cdist(centers, points)

    # build eps-Voronoi patches
    index = np.argsort(distances, axis=0)
    for node in range(n):
        patch = index[0, node]
        node_lists[patch].append(node)
        for other in patch_index[node]:
                overlaps[other][patch] += 1
                overlaps[patch][other] += 1
        patch_index[node].append(patch)
        min_dist = distances[patch, node]
        for patch in index[1:, node]:
            if distances[patch, node] < eps * min_dist:
                node_lists[patch].append(node)
                for other in patch_index[node]:
                        overlaps[other][patch] += 1
                        overlaps[patch][other] += 1
                patch_index[node].append(patch)
            else:
                break

    # grow patches until degree constraints and size constraints are satisfied

    # find patches that do not satisfy the constraints
    grow = {i for i, ov in enumerate(overlaps) if len(node_lists[i]) < min_size
                                               or sum(v >= min_overlap for v in ov.values()) < min_degree}

    # sort distance matrix (make sure patch members are sorted first)
    for i, nodes in enumerate(node_lists):
        distances[i, nodes] = -1
    index = np.argsort(distances, axis=1)

    while grow:
        patches = list(grow)
        for patch in patches:
            size = len(node_lists[patch])
            if size >= n or (size >= min_size
                             and sum(v >= min_overlap for v in overlaps[patch].values()) >= min_degree):
                grow.remove(patch)
            else:
                next_node = index[patch, size]
                node_lists[patch].append(next_node)
                for other in patch_index[next_node]:
                    overlaps[other][patch] += 1
                    overlaps[patch][other] += 1
                patch_index[next_node].append(patch)

    # check patch network is connected and add edges if necessary
    patch_network = nx.Graph()
    for i, others in enumerate(overlaps):
        for other, ov in others.items():
            if ov >= min_overlap:
                patch_network.add_edge(i, other)

    if not nx.is_connected(patch_network):
        components = list(nx.connected_components(patch_network))
        edges = []
        for c1, patches1 in enumerate(components):
            patches1 = list(patches1)
            for it, patches2 in enumerate(components[c1+1:]):
                patches2 = list(patches2)
                c2 = c1+it+1
                patch_distances = cdist(centers[patches1, :], centers[patches2, :])
                i, j = np.unravel_index(np.argmin(patch_distances), patch_distances.shape)
                edges.append((patch_distances[i, j], patches1[i], patches2[j], c1, c2))
        edges.sort()
        component_graph = nx.Graph()
        component_graph.add_nodes_from(range(len(components)))
        for dist, i, j, c1, c2 in edges:
            nodes1 = set(node_lists[i])
            nodes2 = set(node_lists[j])
            nodes = nodes1.union(nodes2)
            dist_list = [(distances[i, node] + distances[j, node], node) for node in nodes]
            dist_list.sort()
            for it in range(min_overlap):
                node = dist_list[it][1]
                if node not in nodes1:
                    node_lists[i].append(node)
                if node not in nodes2:
                    node_lists[j].append(node)
            component_graph.add_edge(c1, c2)
            patch_network.add_edge(i, j)
            if nx.is_connected(component_graph):
                break

    if return_graph:
        return [ut.Patch(nodes, points[nodes, :]) for nodes in node_lists], patch_network
    else:
        return [ut.Patch(nodes, points[nodes, :]) for nodes in node_lists]


def rand_scale_patches(alignment_problem: ut.AlignmentProblem, min_scale=1e-2):
    """
    randomly scale patches of alignment problem and return the true scales (used for testing)

    :param AlignmentProblem alignment_problem: Alignment problem to be rescaled
    :param float min_scale: minimum scale factor (scale factors are sampled
        log-uniformly from the interval [min_scale, 1/min_scale])

    :return: list of true scales
    """
    scales = np.exp(ut.rg.uniform(np.log(min_scale), np.log(1/min_scale), alignment_problem.n_patches))
    alignment_problem.scale_patches(scales)
    return scales


def rand_rotate_patches(alignment_problem: ut.AlignmentProblem):
    """
    randomly rotate patches of alignment problem and return true rotations (used for testing)

    :param AlignmentProblem alignment_problem: Alignment problem to be transformed

    :return: list of true rotations
    """
    rotations = [rand_orth(alignment_problem.dim) for _ in alignment_problem.patches]
    alignment_problem.rotate_patches(rotations)
    return rotations


def rand_shift_patches(alignment_problem: ut.AlignmentProblem, shift_scale=100.0):
    """
    randomly shift patches by adding a normally distributed vector (used for testing)

    :param AlignmentProblem alignment_problem: Alignment problem to be transformed

    :param float shift_scale: Standard deviation for shifts

    :return: np.ndarray of true shifts
    """
    shifts = ut.rg.normal(loc=0, scale=shift_scale, size=(alignment_problem.n_patches, alignment_problem.dim))
    alignment_problem.translate_patches(shifts)
    return shifts


def add_noise(alignment_problem: ut.AlignmentProblem, noise_level=1, scales=None):
    """
    Add random normally-distributed noise to each point in each patch

    :param AlignmentProblem alignment_problem: Alignment problem to be transformed

    :param noise_level: Standard deviation of noise

    :param scales: (optional) list of scales for each patch (noise for patch is multiplied by corresponding scale)
    """
    if noise_level > 0:
        if scales is None:
            scales = np.ones(alignment_problem.n_patches)
        for patch, scale in zip(alignment_problem.patches, scales):
            noise = ut.rg.normal(loc=0, scale=noise_level * scale, size=patch.shape)
            patch.coordinates += noise


def noise_profile(points, base_problem, max_noise=0.5, steps=101, scales=None,
                  types=None, labels=None, min_overlap=None, plot=True):
    """
    Plot procrustes reconstruction errors as a function of the noise level

    :param points: True data

    :param base_problem: Alignment problem without noise (usually should have rotated/shifted/scaled patches)

    :param max_noise: Maximum standard deviation for noise

    :param steps: number of noise steps between 0 and `max_noise`

    :param scales: scales of patches (noise is scaled accordingly)

    :param types: List of AlignmentProblem subclasses to test (each is tested with the same noise)

    :param labels: Labels to use for the legend

    :param min_overlap: Values of `min_overlap` to include in test.

    :param bool plot: plot results [default: True]

    :return: noise_levels, errors
    """

    # set up labels and min_overlap
    if min_overlap is not None:
        if not isinstance(min_overlap, Iterable):
            min_overlap = [min_overlap]
        else:
            if labels is not None:
                labels = [f"{label}-{ov}" for label in labels for ov in min_overlap]

    if types is None:
        types = [ut.AlignmentProblem]
    errors = [[] for _ in range(steps)]
    noise_levels = np.linspace(0, max_noise, steps)
    for e_l, noise in zip(errors, noise_levels):
        noisy_problem = copy(base_problem)
        add_noise(noisy_problem, noise, scales)
        for problem_cls in types:
            if min_overlap is None:
                problem = copy(noisy_problem)
                problem.__class__ = problem_cls
                e_l.append(ut.procrustes_error(points, problem.get_aligned_embedding(scale=True)))
            else:
                for ov in min_overlap:
                    ov_problem = problem_cls(noisy_problem.patches, min_overlap=ov)
                    e_l.append(ut.procrustes_error(points, ov_problem.get_aligned_embedding(scale=True)))
        print(f"Noise: {noise}, errors: {e_l}")

    plt.plot(noise_levels, errors)
    if labels is not None:
        plt.legend(labels)
    plt.xlabel('noise level')
    plt.ylabel('procrustes errors')
    return noise_levels, errors


def plot_reconstruction(points, problem, scale=True):
    """
    Plot the reconstruction error for each point

    :param points: True positions

    :param problem: Alignment problem

    :param scale: Rescale patches [default: True]
    """
    recovered_pos = problem.get_aligned_embedding(scale=scale)
    points, recovered_pos, error = procrustes(points, recovered_pos)
    plt.plot(np.array([points[:, 0], recovered_pos[:, 0]]), np.array([points[:, 1], recovered_pos[:, 1]]),
             'k', linewidth=0.5)
    plt.plot(recovered_pos[:, 0], recovered_pos[:, 1], 'k.', markersize=1)
    for patch in problem.patches:
        index = list(patch.index.keys())
        old_c = np.mean(points[index, :], axis=0)
        new_c = np.mean(recovered_pos[index, :], axis=0)
        plt.plot([old_c[0], new_c[0]], [old_c[1], new_c[1]], 'r', linewidth=1)
        plt.plot([new_c[0]], [new_c[1]], 'r.', markersize=2)
    return error


def save_data(points, filename):
    filename = ut.ensure_extension(filename, '.csv')
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(points)


def rand_orth(dim):
    """Sample a random orthogonal matrix (for testing).
    Use normal distribution to ensure uniformity."""
    a = ut.rg.normal(size=(dim, 1))
    a = a / np.sqrt(a.T.dot(a))
    M = a

    for _ in range(dim - 1):
        a = ut.rg.normal(size=(dim, 1))
        a = a - M.dot(M.T).dot(a)
        a = a / np.sqrt(a.T.dot(a))
        M = np.hstack((M, a))
    return M


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run local2global example.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=5, type=int, help="Number of clusters in test data")
    parser.add_argument('--max_shift', default=1, type=float, help="Cluster shift")
    parser.add_argument('--kmeans', action='store_true', help="use kmeans to find patch centers")
    parser.add_argument('--max_var', default=0.2, type=float, help="Cluster dispersion")
    parser.add_argument('--max_size', default=2000, type=int, help="Max cluster size")
    parser.add_argument('--sample_size', default=100, type=int, help="Number of patches")
    parser.add_argument('--dim', default=2, type=int, help="Data dimension")
    parser.add_argument('--eps', default=1.6, type=float, help="Tolerance for patch overlaps")
    parser.add_argument('--min_overlap', type=int, default=10,
                        help="Minimum patch overlap for connectivity constraint")
    parser.add_argument('--min_recovery_overlap', type=int, default=[], action='append',
                        help='Minimum patch overlap for recovery (defaults to min_overlap)')
    parser.add_argument('--min_size', type=int, default=None, help="Minimum patch size")
    parser.add_argument('--min_degree', type=int, default=None, help="Minimum patch degree")
    parser.add_argument('--max_noise', default=0.2, type=float, help="Maximum noise level")
    parser.add_argument('--steps', default=101, type=int, help="Number of steps for noise profile")
    parser.add_argument('--plot_noise', '-p', default=[], action='append', type=float,
                        help="Noise level to plot (can be specified multiple times)")
    parser.add_argument('--outdir', '-o', type=str, help='output dir', default='.')
    parser.add_argument('--seed', default=None, type=int, help="Seed for rng")
    args = parser.parse_args()

    if not args.min_recovery_overlap:
        args.min_recovery_overlap = None

    ut.seed(args.seed)
    problem_types = [ut.AlignmentProblem, ut.WeightedAlignmentProblem]
    labels = ['standard', 'weighted']

    # generate random data
    points = generate_data(n_clusters=args.n_clusters, scale=args.max_shift, std=args.max_var,
                           max_size=args.max_size, dim=args.dim)
    outdir = Path(args.outdir)
    save_data(points, filename=outdir / 'points.csv')
    patches = Voronoi_patches(points=points, sample_size=args.sample_size, min_degree=args.min_degree,
                              min_overlap=args.min_overlap, min_size=args.min_size, eps=args.eps, kmeans=args.kmeans)
    base_problem = ut.AlignmentProblem(patches, min_overlap=args.min_overlap)
    rand_shift_patches(base_problem)
    scales = rand_scale_patches(base_problem)
    rand_rotate_patches(base_problem)

    print(f"Mean patch degree: {mean(base_problem.patch_degrees)}")
    if args.steps > 0:
        plt.figure()
        noise_profile(points, base_problem, steps=args.steps, max_noise=args.max_noise, scales=scales,
                      types=problem_types, labels=labels, min_overlap=args.min_recovery_overlap)
        plt.savefig(path.join(args.outdir, 'noise_profile.pdf'))
        plt.close()

    for noise in args.plot_noise:
        if args.dim > 2:
            raise RuntimeError("plotting reconstruction error only works for dim=2")
        noisy_problem = copy(base_problem)
        add_noise(noisy_problem, noise_level=noise, scales=scales)
        for problem_cls, label in zip(problem_types, labels):
            plt.figure()
            problem = copy(noisy_problem)
            problem.__class__ = problem_cls
            error = plot_reconstruction(points, problem)
            plt.title(f"Noise: {noise}, error: {error}")
            plt.savefig(path.join(args.outdir, f'errorplot_{label}_noise{noise}.pdf'))
            plt.close()


