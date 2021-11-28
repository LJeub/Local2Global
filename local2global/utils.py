"""Implementation of local2global algorithm"""
#  Copyright (c) 2021. Lucas G. S. Jeub
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import sys

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

import ilupp

from tqdm.auto import tqdm

from .patch import Patch

rg = np.random.default_rng()
eps = np.finfo(float).eps


def seed(new_seed):
    """
    Change seed of random number generator.

    Args:
        new_seed: New seed value

    """
    global rg
    rg = np.random.default_rng(new_seed)


def ensure_extension(filename, extension):
    """
    check filename for extension and add it if necessary

    Args:
        filename: input filename
        extension: desired extension (including `.`)

    Returns:
        filename with extension added

    Raises:
        ValueError: if filename has the wrong extension

    """
    filename = Path(filename)
    if filename.suffix == "":
        filename = filename.with_suffix(extension)
    elif filename.suffix != extension:
        raise ValueError(f"filename should have extension {extension}, not {filename.suffix}")
    return filename


def procrustes_error(coordinates1, coordinates2):
    """
    compute the procrustes alignment error between two sets of coordinates

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape.
    """
    return procrustes(coordinates1, coordinates2)[2]


def local_error(patch: Patch, reference_coordinates):
    """
    compute the euclidean distance between patch coordinate and reference
    coordinate for each node in patch

    Args:
        patch:
        reference_coordinates:

    Returns:
        vector of error values
    """
    return np.linalg.norm(reference_coordinates[patch.nodes, :] - patch.coordinates, axis=1)


def transform_error(transforms):
    """
    Compute the recovery error based on tracked transformations.

    After recovery, all transformations should be constant across patches
    as we can recover the embedding only up to a global scaling/rotation/translation.
    The error is computed as the mean over transformation elements of the standard deviation over patches.

    Args:
        transforms: list of transforms
    """
    return np.mean(np.std(transforms, axis=0))


def orthogonal_MSE_error(rots1, rots2):
    """
    Compute the MSE between two sets of orthogonal transformations up to a global transformation

    Args:
        rots1: First list of orthogonal matrices
        rots2: Second list of orthogonal matrices

    """
    dim = len(rots1[0])
    rots1 = np.asarray(rots1)
    rots1 = rots1.transpose((0, 2, 1))
    rots2 = np.asarray(rots2)
    combined = np.mean(rots1 @ rots2, axis=0)
    _, s, _ = sp.linalg.svd(combined)
    return 2*(dim - np.sum(s))


def _cov_svd(coordinates1: np.ndarray, coordinates2: np.ndarray):
    """
    Compute SVD of covariance matrix between two sets of coordinates

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape.
    """
    coordinates1 = coordinates1 - coordinates1.mean(axis=0)
    coordinates2 = coordinates2 - coordinates2.mean(axis=0)
    cov = coordinates1.T @ coordinates2
    return sp.linalg.svd(cov)


def relative_orthogonal_transform(coordinates1, coordinates2):
    """
    Find the best orthogonal transformation aligning two sets of coordinates for the same nodes

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

    Note that the two sets of coordinates need to have the same shape.
    """
    # Note this is completely equivalent to the approach in
    # "Closed-Form Solution of Absolute Orientation using Orthonormal Matrices"
    # Journal of the Optical Society of America A · July 1988
    U, s, Vh = _cov_svd(coordinates1, coordinates2)
    return U @ Vh


def nearest_orthogonal(mat):
    """
    Compute nearest orthogonal matrix to a given input matrix

    Args:
        mat: input matrix
    """
    U, s, Vh = sp.linalg.svd(mat)
    return U @ Vh


def relative_scale(coordinates1, coordinates2, clamp=1e8):
    """
    compute relative scale of two sets of coordinates for the same nodes

    Args:
        coordinates1: First set of coordinates (array-like)
        coordinates2: Second set of coordinates (array-like)

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


class AlignmentProblem:
    """
    Implements the standard local2global algorithm using an unweighted patch graph
    """
    n_nodes = None
    """total number of nodes"""

    n_patches = None
    """number of patches"""

    dim = None
    """embedding dimension"""

    scales = None
    """tracks scale transformations applied to patches (updated by :meth:`scale_patches`)"""

    rotations = None
    """tracks orthogonal transformations applied to patches (updated by :meth:`rotate_patches`)"""

    shifts = None
    """tracks translation transformations applied to patches (updated by :meth:`scale_patches`, 
       :meth:`rotate_patches`, and :meth:`translate_patches`)"""

    verbose = False
    """print debug output if `True`"""

    def weight(self, i, j):
        """Compute the weighting factor for a pair of patches

        Args:
            i: First patch index
            j: Second patch index

        Returns:
            1

        Override this in subclasses for weighted alignment
        """
        return 1

    def __init__(self, patches: List[Patch], patch_edges=None,
                 min_overlap=None, copy_data=True, self_loops=False, verbose=False):
        """
        Initialise the alignment problem with a list of patches

        Args:
            patches: List of patches to synchronise
            patch_edges: if provided, only compute relative transformations for given patch edges (all pairs of patches
                         with at least ``min_overlap`` points in common are included by default)
            min_overlap (int): minimum number of points in the overlap required for two patches to be considered
                               connected (defaults to `dim+1`) where `dim` is the embedding dimension of the patches
            copy_data (bool): if ``True``, input patches are copied (default: ``True``)
            self_loops (bool): if ``True``, self-loops from a patch to itself are included in the synchronisation problem
                               (default: ``False``)
            verbose(bool): if True print diagnostic information (default: ``False``)

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
            for node in patch.nodes:
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
            if self_loops or e[0] != e[1]:
                if len(self.patch_overlap[e]) < min_overlap:
                    if patch_edges is None:
                        del self.patch_overlap[e]
                    else:
                        raise RuntimeError("Patch edges do not satisfy minimum overlap")
            else:
                del self.patch_overlap[e]  # remove spurious self-loops

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
        """
        Synchronise scales of the embeddings for each patch

        Args:
            scale_factors: if provided apply the given scales instead of synchronising
        """
        if scale_factors is None:
            scale_factors = [1 / x for x in self.calc_synchronised_scales()]

        for i, scale in enumerate(scale_factors):
            self.patches[i].coordinates *= scale
            # track transformations
            self.scales[i] *= scale
            self.shifts[i] *= scale
        return self

    def calc_synchronised_scales(self, max_scale=1e8):
        """
        Compute the scaling transformations that best align the patches

        Args:
            max_scale: maximum allowed scale (all scales are clipped to the range [``1/max_scale``, ``max_scale``])
                       (default: 1e8)

        Returns:
            list of scales

        """
        scaling_mat = self._transform_matrix(lambda ov1, ov2: relative_scale(ov1, ov2, max_scale), 1)
        vec = self._synchronise(scaling_mat, 1)
        vec = vec.flatten()
        vec = np.abs(vec)
        vec /= vec.mean()
        vec = np.clip(vec, a_min=1/max_scale, a_max=max_scale, out=vec)  # avoid blow-up
        return vec

    def rotate_patches(self, rotations=None):
        """align the rotation/reflection of all patches

        Args:
            rotations: If provided, apply the given transformations instead of synchronizing patch rotations
        """
        if rotations is None:
            rotations = (rot.T for rot in self.calc_synchronised_rotations())

        for i, rot in enumerate(rotations):
            self.patches[i].coordinates = self.patches[i].coordinates @ rot.T
            # track transformations
            self.rotations[i] = self.rotations[i] @ rot.T
            self.shifts[i] = self.shifts[i] @ rot.T
        return self

    def calc_synchronised_rotations(self):
        """Compute the orthogonal transformations that best align the patches"""
        rots = self._transform_matrix(relative_orthogonal_transform, self.dim, symmetric_weights=True)
        vecs = self._synchronise(rots, blocksize=self.dim, symmetric=True)
        for mat in vecs:
            mat[:] = nearest_orthogonal(mat)
        return vecs

    def translate_patches(self, translations=None):
        """align the patches by translation

        Args:
            translations: If provided, apply the given translations instead of synchronizing

        """
        if translations is None:
            translations = self.calc_synchronised_translations()

        for i, t in enumerate(translations):
            self.patches[i].coordinates += t
            # keep track of transformations
            self.shifts[i] += t
        return self

    def calc_synchronised_translations(self):
        """Compute translations that best align the patches"""
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

    def mean_embedding(self, out=None):
        """
        Compute node embeddings as the centroid over patch embeddings

        Args:
            out: numpy array to write results to (supply a memmap for large-scale problems that do not fit in ram)
        """
        if out is None:
            embedding = np.zeros((self.n_nodes, self.dim))
        else:
            embedding = out  # important: needs to be zero-initialised

        count = np.array([len(patch_list) for patch_list in self.patch_index])
        for patch in tqdm(self.patches, smoothing=0, desc='Compute mean embedding'):
            embedding[patch.nodes] += patch.coordinates

        embedding /= count[:, None]

        return embedding

    def median_embedding(self, out=None):
        if out is None:
            out = np.full((self.n_nodes, self.dim), np.nan)

        for i, pids in tqdm(enumerate(self.patch_index), total=self.n_nodes, desc='Compute median embedding for node'):
            if pids:
                points = np.array([self.patches[pid].get_coordinate(i) for pid in pids])
                out[i] = np.median(points, axis=0)
        return out

    def align_patches(self, scale=False):
        if scale:
            self.scale_patches()
        self.rotate_patches()
        self.translate_patches()
        return self

    def get_aligned_embedding(self, scale=False, realign=False, out=None):
        """Return the aligned embedding

        Args:
            scale (bool): if ``True``, rescale patches (default: ``False``)
            realign (bool): if ``True``, recompute aligned embedding even if it already exists (default: ``False``)

        Returns:
            n_nodes x dim numpy array of embedding coordinates
        """
        if realign or self._aligned_embedding is None:
            self._aligned_embedding = self.align_patches(scale).mean_embedding(out)
        return self._aligned_embedding

    def save_patches(self, filename):
        """
        save patch embeddings to json file
        Args:
            filename: path to output file


        """
        filename = ensure_extension(filename, '.json')
        patch_dict = {str(i): {int(node): [float(c) for c in coord]
                               for node, coord in zip(patch.index, patch.coordinates)}
                      for i, patch in enumerate(self.patches)}
        with open(filename, 'w') as f:
            json.dump(patch_dict, f)

    @classmethod
    def load(cls, filename):
        """
        restore ``AlignmentProblem`` from patch file

        Args:
            filename: path to patch file

        """
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
        """
        save aligned embedding to json file

        Args:
            filename: output filename

        """
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

        Args:
            transform: function to compute the relative transformation
            dim: output dimension of transform should be `(dim, dim)`
            symmetric_weights: if true use symmetric weighting (default: False)
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
        for count, (i, j) in tqdm(enumerate(keys), total=len(keys),
                                  desc='Compute relative transformations'):
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
        """
        compute weight for pair of patches

        Args:
            i: first patch index
            j: second patch index

        Returns:
            number of shared nodes between patches `i` and `j`
        """
        ov = len(self.patch_overlap[i, j])
        return ov


class SVDAlignmentProblem(WeightedAlignmentProblem):
    def __init__(self, *args, tol=1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.tol = tol

    def calc_synchronised_rotations(self):
        """Compute the orthogonal transformations that best align the patches"""
        rots = self._transform_matrix(relative_orthogonal_transform, self.dim, symmetric_weights=False)
        vecs = self._synchronise(rots, blocksize=self.dim, symmetric=True)
        for mat in vecs:
            mat[:] = nearest_orthogonal(mat)
        return vecs

    @staticmethod
    def _preconditioner(matrix, noise_level=1e-8):
        dim = matrix.shape[0]

        if dim < 20000:
            ilu = ss.linalg.spilu(matrix + noise_level*ss.rand(dim, dim, 1/dim, format='csc', random_state=rg))

            def cond_solve(x):
                return ilu.solve(ilu.solve(x), 'T')

            M = ss.linalg.LinearOperator((dim, dim), matvec=cond_solve, matmat=cond_solve)
        else:
            print('using ILU0')
            ilu = ilupp.ILU0Preconditioner(matrix + noise_level*ss.rand(dim, dim, 1/dim, format='csc', random_state=rg))

            def cond_solve(x):
                y = x.copy()
                ilu.apply(y)
                ilu.apply_trans(y)
                return y

            M = ss.linalg.LinearOperator((dim, dim), matvec=cond_solve)
        return M

    def _synchronise(self, matrix: ss.spmatrix, blocksize=1, symmetric=False):
        """Compute synchronised group elements from matrix
        :param matrix: matrix to synchronise
        :param blocksize: size of group element blocks
        """
        dim = matrix.shape[0]
        if blocksize == 1:
            # leading eigenvector is much easier to compute in this case
            eigs, vecs = ss.linalg.eigs(matrix, k=1, v0=rg.normal(size=dim))
            eigs = eigs.real
            vecs = vecs.real
        else:
            if dim < 5*blocksize:
                matrix = matrix.T - ss.identity(dim)
                matrix = matrix @ matrix.T
                matrix += ss.identity(dim)
                # this uses a lot of memory for large matrices due to computing full LU factorisation
                eigs, vecs = ss.linalg.eigsh(matrix, which='LM', k=blocksize, v0=np.ones(dim), maxiter=10000, sigma=0.9,
                                             mode='buckling')
            else:
                matrix = matrix.tocsc(copy=False) - ss.identity(dim)


                # v0 = np.tile(np.eye(blocksize), (dim // blocksize, 1))

                def matmat(x):
                    x1 = matrix.T @ x
                    return matrix @ x1
                B_op = ss.linalg.LinearOperator((dim, dim), matvec=matmat, matmat=matmat, rmatmat=matmat, rmatvec=matmat)

                if self.verbose:
                    print('computing ilu')
                # ILU helps but maybe could do better
                # fill_in = 100

                #
                # TODO: could use pytorch implementation to run this on GPU
                if self.verbose:
                    print('finding eigenvectors')
                #
                v0 = rg.normal(size=(dim, blocksize))
                M = self._preconditioner(matrix, self.tol)
                max_tries = 10
                max_iter = max(blocksize, 10)
                for _ in range(max_tries):
                    try:
                        eigs, vecs, res = ss.linalg.lobpcg(B_op, v0, M=M, largest=False, maxiter=max_iter,
                                                  verbosityLevel=self.verbose, tol=self.tol, retResidualNormsHistory=True)
                    except ValueError as e:
                        print(f'LOBPCG failed with error {e}, retrying with noise in preconditioner')
                        M = self._preconditioner(matrix, self.tol)
                        v0 += rg.normal(size=v0.shape, scale=self.tol)
                        eigs, vecs, res = ss.linalg.lobpcg(B_op, v0, largest=False, maxiter=max_iter,
                                                           verbosityLevel=self.verbose, tol=self.tol,
                                                           retResidualNormsHistory=True)
                    if res[-1].max() > self.tol:
                        v0 = vecs + rg.normal(size=vecs.shape, scale=self.tol)
                    else:
                        break
                else:  # LOBPCG still failed after max_tries
                    raise RuntimeError(f'LOBPCG still failed after {max_tries=} initialisations')

                # eigs, vecs = ss.linalg.lobpcg(B_op, v0, largest=False, maxiter=500,
                #                               verbosityLevel=self.verbose, tol=tol)
        if self.verbose:
            print(f"eigs: {eigs}")
        order = np.argsort(np.abs(eigs))
        vecs = vecs[:, order[:blocksize]].real
        vecs.shape = (dim//blocksize, blocksize, blocksize)
        return vecs


