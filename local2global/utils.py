import scipy as sp
from scipy import sparse as ss
from scipy.linalg import orthogonal_procrustes
import numpy as np
from scipy.sparse.linalg import lsqr, lsmr
from scipy.spatial import procrustes
import copy
from collections import defaultdict
from typing import List, Callable, Any
import networkx as nx
from pathlib import Path
import json

rg = np.random.default_rng()
eps = np.finfo(float).eps


def seed(new_seed):
    """Change seed of random number generator.

    :param new_seed: New seed value
    """
    global rg
    rg = np.random.default_rng(new_seed)

def ensure_extension(filename, extension):
    filename = Path(filename)
    if filename.suffix == "":
        filename = filename.with_suffix(extension)
    elif filename.suffix != extension:
        raise ValueError(f"filename should have extension {extension}, not {filename.suffix}")
    return filename


def procrustes_error(coordinates1, coordinates2):
    """compute the procrustes alignment error between two sets of coordinates

    :param coordinates1: First set of coordinates (array-like)
    :param coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape."""
    return procrustes(coordinates1, coordinates2)[2]


def transform_error(transforms):
    """
    Compute the recovery error based on tracked transformations.

    After recovery, all transformations should be constant across patches
    as we can recover the embedding only up to a global scaling/rotation/translation.

    The error is computed as the mean over transformation elements of the standard deviation over patches.
    """
    return np.mean(np.std(transforms, axis=0))


def orthogonal_MSE_error(rots1, rots2):
    """Compute the MSE between two sets of orthogonal transformations up to a global transformation

    :param rots1: First list of orthogonal matrices
    :param rots2: Second list of orthogonal matrices
    """
    dim = len(rots1[0])
    rots1 = np.asarray(rots1)
    rots1 = rots1.transpose((0, 2, 1))
    rots2 = np.asarray(rots2)
    combined = np.mean(rots1 @ rots2, axis=0)
    _, s, _ = sp.linalg.svd(combined)
    return 2*(dim - np.sum(s))


def _cov_svd(coordinates1: np.ndarray, coordinates2: np.ndarray):
    """Compute SVD of covariance matrix between two sets of coordinates

    :param coordinates1: First set of coordinates (array-like)
    :param coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape."""
    coordinates1 = coordinates1 - coordinates1.mean(axis=0)
    coordinates2 = coordinates2 - coordinates2.mean(axis=0)
    cov = coordinates1.T @ coordinates2
    return sp.linalg.svd(cov)


def relative_orthogonal_transform(coordinates1, coordinates2):
    """Find the best orthogonal transformation aligning two sets of coordinates for the same nodes

    :param coordinates1: First set of coordinates (array-like)
    :param coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape.
    """
    # Note this is completely equivalent to the approach in
    # "Closed-Form Solution of Absolute Orientation using Orthonormal Matrices"
    # Journal of the Optical Society of America A · July 1988
    U, s, Vh = _cov_svd(coordinates1, coordinates2)
    return U @ Vh


def nearest_orthogonal(mat):
    """compute nearest orthogonal matrix to a given input matrix

    :param mat: input matrix"""
    U, s, Vh = sp.linalg.svd(mat)
    return U @ Vh


def relative_scale(coordinates1, coordinates2, clamp=1e8):
    """compute relative scale of two sets of coordinates for the same nodes

    :param coordinates1: First set of coordinates (array-like)
    :param coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape.
    """
    scale1 = np.linalg.norm(coordinates1 - np.mean(coordinates1, axis=0))
    scale2 = np.linalg.norm(coordinates2 - np.mean(coordinates2, axis=0))
    if scale1 > clamp * scale2:
        print('extremely large scale clamped')
        return clamp
    if scale1 * clamp < scale2:
        print('extremely small scale clamped')
        return 1/clamp
    return scale1 / scale2


class Patch:
    def __init__(self, nodes, coordinates):
        """Initialise a patch from a list of nodes and corresponding coordinates

        :param nodes: Iterable of integer node indeces for patch
        :param coordinates: Array-like of node coordinates of shape (len(nodes), dim)
        """
        self.index = {int(n): i for i, n in enumerate(nodes)}
        self.coordinates = np.array(coordinates)

    @property
    def shape(self):
        """get shape of patch coordinates, where `shape[0]` is the number of nodes in the patch
        and `shape[1]` is the embedding dimension"""
        return self.coordinates.shape

    def get_coordinates(self, nodes):
        """get coordinates for a list of nodes

        :param nodes: Iterable of node indeces"""
        return self.coordinates[[self.index[node] for node in nodes], :]

    def get_coordinate(self, node):
        """get coordinate for a single node
        :param node: Integer node index"""
        return self.coordinates[self.index[node], :]

    def __copy__(self):
        """return a copy of patch"""
        instance = self.__new__(type(self))
        instance.index = dict(self.index)
        instance.coordinates = np.array(self.coordinates)
        return instance


class AlignmentProblem:
    """
    Implements the standard local2global algorithm using an unweighted patch graph

    :ivar n_nodes: total number of nodes

    """
    def weight(self, i, j):
        """Compute the weighting factor for a pair of patches

        :param i: First patch index
        :param j: Second patch index

        Override this in subclasses for weighted alignment
        """
        return 1

    def __init__(self, patches: List[Patch], patch_edges=None,
                 min_overlap=None, copy_data=True, self_loops=False, verbose=False):
        """Initialise the alignment problem with a list of patches

        :param patches: List of patches to synchronise
        :param int min_overlap: minimum number of points in the overlap required for two patches to be considered
         connected (defaults to `dim+1`) where `dim` is the embedding dimension of the patches
        :param bool copy_data: if True, input patches are copied (default: True)
        :param bool self_loops: if True, self-loops from a patch to itself are included in the synchronisation problem
        :param verbose: if True print diagnostic information (default: False)
        (default: False)
        """
        if copy_data:
            self.patches = [copy.copy(patch) for patch in patches]
        else:
            self.patches = patches
        self.verbose = verbose
        self.n_nodes = max(max(patch.index.keys()) for patch in self.patches) + 1
        self.n_patches = len(self.patches)
        self.dim = self.patches[0].shape[1]
        self.scales = np.ones(self.n_patches)
        self.rotations = np.tile(np.eye(self.dim), (self.n_patches, 1, 1))
        self.shifts = np.zeros((self.n_patches, self.dim))
        self._aligned_embedding = None
        if min_overlap is None:
            min_overlap = self.dim + 1

        # create an index for the patch membership of each node
        self.patch_index = [[] for _ in range(self.n_nodes)]
        for i, patch in enumerate(self.patches):
            for node in patch.index.keys():
                self.patch_index[node].append(i)

        # find patch overlaps
        self.patch_overlap = defaultdict(list)
        for i, patch in enumerate(self.patches):
            for node in patch.index:
                for j in self.patch_index[node]:
                    if self_loops or i != j:
                        self.patch_overlap[i, j].append(node)

        # restrict to patch edges if provided
        if patch_edges is not None:
            self.patch_overlap = {e: self.patch_overlap[e] for e in patch_edges}

        # remove small overlaps
        keys = list(self.patch_overlap.keys())
        for e in keys:
            if len(self.patch_overlap[e]) < min_overlap:
                if patch_edges is None:
                    del self.patch_overlap[e]
                else:
                    raise RuntimeError("Patch edges do not satisfy minimum overlap")

        # find patch degrees
        self.patch_degrees = [0] * self.n_patches
        for i, j in self.patch_overlap.keys():
            self.patch_degrees[i] += 1

        patch_graph = nx.Graph()
        patch_graph.add_edges_from(self.patch_overlap.keys())
        if nx.number_connected_components(patch_graph) > 1:
            raise RuntimeError("patch graph is not connected")

        if self.verbose:
            print(f'mean patch degree: {np.mean(self.patch_degrees)}')

    def scale_patches(self, scale_factors=None):
        """Synchronise scales of the embeddings for each patch

        :param scale_factors: if provided apply the given scales instead of synchronising"""
        if scale_factors is None:
            scale_factors = [1 / x for x in self.calc_synchronised_scales()]

        for i, scale in enumerate(scale_factors):
            self.patches[i].coordinates *= scale
            # track transformations
            self.scales[i] *= scale
            self.shifts[i] *= scale

    def calc_synchronised_scales(self, max_scale=1e8):
        scaling_mat = self._transform_matrix(lambda ov1, ov2: relative_scale(ov1, ov2, max_scale), 1)
        vec = self._synchronise(scaling_mat, 1)
        vec = vec.flatten()
        vec = np.abs(vec)
        vec /= vec.mean()
        vec = np.clip(vec, a_min=1/max_scale, a_max=max_scale, out=vec)  # avoid blow-up
        return vec

    def rotate_patches(self, rotations=None):
        """align the rotation/reflection of all patches

        :param rotations: If provided, apply the given transformations instead of synchronizing patch rotations
        """
        if rotations is None:
            rotations = (rot.T for rot in self.calc_synchronised_rotations())

        for i, rot in enumerate(rotations):
            self.patches[i].coordinates = self.patches[i].coordinates @ rot.T
            # track transformations
            self.rotations[i] = self.rotations[i] @ rot.T
            self.shifts[i] = self.shifts[i] @ rot.T

    def calc_synchronised_rotations(self):
        """Synchronise the all orthogonal pairwise transformations"""
        rots = self._transform_matrix(relative_orthogonal_transform, self.dim, symmetric_weights=True)
        vecs = self._synchronise(rots, blocksize=self.dim, symmetric=True)
        for mat in vecs:
            mat[:] = nearest_orthogonal(mat)
        return vecs

    def translate_patches(self, translations=None):
        """Translate patches"""
        if translations is None:
            translations = self.calc_synchronised_translations()

        for i, t in enumerate(translations):
            self.patches[i].coordinates += t
            # keep track of transformations
            self.shifts[i] += t

    def calc_synchronised_translations(self):
        """Compute patch translations"""
        b = np.empty((len(self.patch_overlap), self.dim))
        row = []
        col = []
        val = []
        for i, ((p1, p2), overlap) in enumerate(self.patch_overlap.items()):
            row.append(i)
            col.append(p1)
            val.append(-1)
            row.append(i)
            col.append(p2)
            val.append(1)
            b[i, :] = np.mean(self.patches[p1].get_coordinates(overlap)
                              - self.patches[p2].get_coordinates(overlap), axis=0)
        A = ss.coo_matrix((val, (row, col)), shape=(len(self.patch_overlap), self.n_patches), dtype=np.int8)
        A = A.tocsr()
        translations = np.empty((self.n_patches, self.dim))
        for d in range(self.dim):
            translations[:, d] = lsmr(A, b[:, d], atol=1e-16, btol=1e-16)[0]
            # TODO: probably doesn't need to be that accurate, this is for testing
        return translations

    def mean_embedding(self):
        embedding = np.empty((self.n_nodes, self.dim))
        for node, patch_list in enumerate(self.patch_index):
            embedding[node] = np.mean([self.patches[p].get_coordinate(node) for p in patch_list], axis=0)
        return embedding

    def get_aligned_embedding(self, scale=False, realign=False):
        """Return the aligned embedding

        :param scale: Set scale=True to rescale patches
        :param realign: Set realign=True to recompute aligned embedding even if it already exists
        :return: n_nodes x dim numpy array of embedding coordinates
        :rtype: np.array
        """
        if realign or self._aligned_embedding is None:
            if scale:
                self.scale_patches()
            self.rotate_patches()
            self.translate_patches()
            self._aligned_embedding = self.mean_embedding()
        return self._aligned_embedding

    def save_patches(self, filename):
        filename = ensure_extension(filename, '.json')
        patch_dict = {str(i): {int(node): [float(c) for c in coord]
                               for node, coord in zip(patch.index, patch.coordinates)}
                      for i, patch in enumerate(self.patches)}
        with open(filename, 'w') as f:
            json.dump(patch_dict, f)

    @classmethod
    def load(cls, filename):
        filename = ensure_extension(filename, '.json')
        with open(filename) as f:
            patch_dict = json.load(f)
        patch_list = [None] * len(patch_dict)
        for i, patch_data in patch_dict.items():
            nodes = (int(n) for n in patch_data.keys())
            coordinates = list(patch_data.values())
            patch_list[int(i)] = Patch(nodes, coordinates)
        return cls(patch_list)

    def save_embedding(self, filename):
        filename = ensure_extension(filename, '.json')
        embedding = {str(i): c for i, c in enumerate(self.get_aligned_embedding())}
        with open(filename, 'w') as f:
            json.dump(embedding, f)

    def __copy__(self):
        """return a copy of the alignment problem where all patches are copied."""
        instance = self.__new__(type(self))
        for key, value in self.__dict__.items():
            instance.__dict__[key] = copy.copy(value)
        instance.patches = [copy.copy(patch) for patch in self.patches]
        return instance

    def _synchronise(self, matrix: ss.spmatrix, blocksize=1, symmetric=False):
        dim = matrix.shape[0]
        if symmetric:
            matrix = matrix + ss.eye(dim) # shift to ensure matrix is positive semi-definite for buckling mode
            eigs, vecs = ss.linalg.eigsh(matrix, k=blocksize, v0=rg.normal(size=dim), which='LM',
                                         sigma=2, mode='buckling')
            # eigsh unreliable with multiple (clustered) eigenvalues, only buckling mode seems to help reliably

        else:
            # scaling is not symmetric but Perron-Frobenius applies
            eigs, vecs = ss.linalg.eigs(matrix, k=blocksize, v0=rg.normal(size=dim))
            eigs = eigs.real
            vecs = vecs.real

        order = np.argsort(eigs)
        vecs = vecs[:, order[-1:-blocksize-1:-1]]
        if self.verbose:
            print(f'eigenvalues: {eigs}')
        vecs.shape = (dim//blocksize, blocksize, blocksize)
        return vecs

    def _transform_matrix(self, transform: Callable[[np.ndarray, np.ndarray], Any], dim, symmetric_weights=False):
        """Calculate matrix of relative transformations between patches

        :param Callable[[np.ndarray, np.ndarray], Any] transform: function to compute the relative transformation
        :param int dim: output dimension of transform should be `(dim, dim)`
        :param bool symmetric_weights: if true use symmetric weighting (default: False)
        :return: sparse matrix
        """
        n = self.n_patches  # number of patches
        if dim != 1:
            # construct matrix of rotations as a block-sparse-row matrix
            data = np.empty(shape=(len(self.patch_overlap), dim, dim))
        else:
            data = np.empty(shape=(len(self.patch_overlap),))
        weights = np.zeros(n)
        indptr = np.zeros((n + 1,), dtype=int)
        np.cumsum(self.patch_degrees, out=indptr[1:])
        index = np.empty(shape=(len(self.patch_overlap),), dtype=int)

        keys = sorted(self.patch_overlap.keys())
        # TODO: this could be sped up by a factor of two by not computing rotations twice
        for count, (i, j) in enumerate(keys):
            if i == j:
                element = np.eye(dim)
            else:
                overlap_idxs = self.patch_overlap[i, j]
                # find positions of overlapping nodes in the two reference frames
                overlap1 = self.patches[i].get_coordinates(overlap_idxs)
                overlap2 = self.patches[j].get_coordinates(overlap_idxs)
                element = transform(overlap1, overlap2)
            weight = self.weight(i, j)
            weights[i] += weight
            element *= weight
            data[count] = element
            index[count] = j

        # computed weighted average based on error weights
        if symmetric_weights:
            for i in range(n):
                for ind in range(indptr[i], indptr[i + 1]):
                    data[ind] /= np.sqrt(weights[i]*weights[index[ind]])
        else:
             for i in range(n):
                 data[indptr[i]:indptr[i+1]] /= weights[i]
        if dim == 1:
            matrix = ss.csr_matrix((data, index, indptr), shape=(n, n))
        else:
            matrix = ss.bsr_matrix((data, index, indptr), shape=(dim*n, dim*n), blocksize=(dim, dim))
        return matrix


class WeightedAlignmentProblem(AlignmentProblem):
    """
    Variant of the local2global algorithm where patch edges are weighted according to the number of nodes in the overlap.
    """
    def weight(self, i, j):
        ov = len(self.patch_overlap[i, j])
        return ov


