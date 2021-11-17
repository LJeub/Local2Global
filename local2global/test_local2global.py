"""Test local2global reconstruction"""

import pytest
import numpy as np
import local2global.utils as ut
import local2global.example as ex
from statistics import mean
from copy import copy
from pathlib import Path
import csv
import subprocess
import sys


# fixed seed for testing
# _seed = np.random.SeedSequence(148047762894979172857694243677519903461)

_seed = np.random.SeedSequence()
ut.seed(_seed)
print(_seed.entropy)


@pytest.fixture(autouse=True)
def seed():
    ut.seed(_seed)
    print(f'seed: {_seed.entropy}')


def iter_seed(it):
    it_seed = np.random.SeedSequence(entropy=_seed.entropy, n_children_spawned=it)
    ut.seed(it_seed.spawn(1)[0])


# test parameters
noise_scales = np.linspace(0, 0.1, 11)[1:]
test_classes = [ut.AlignmentProblem,  ut.WeightedAlignmentProblem, ut.SVDAlignmentProblem]
dim = [2, 4, 8]
min_overlap = [d + 1 for d in dim]
tol = 1e-5  # Note LOBPCG is not that accurate

# test data
points_list = [ex.generate_data(n_clusters=5, scale=1, std=0.2, max_size=2000, dim=d) for d in dim]
patches_list = [ex.Voronoi_patches(points=points, sample_size=100, min_overlap=2*mo, min_degree=10,
                                   eps=1+1/d**0.5, return_graph=False, kmeans=True)
                for points, d, mo in zip(points_list, dim, min_overlap)]


@pytest.mark.parametrize("it", range(100))
@pytest.mark.parametrize("problem_cls", test_classes)
@pytest.mark.parametrize("patches,min_overlap", zip(patches_list, min_overlap))
def test_stability(it, problem_cls, patches, min_overlap):
    """Test stability of eigenvector calculations"""
    iter_seed(it)
    problem = problem_cls(patches, min_overlap=min_overlap, verbose=True)
    # ex.add_noise(problem, 1e-8)
    rotations = ex.rand_rotate_patches(problem)
    recovered_rots = problem.calc_synchronised_rotations()
    error = ut.orthogonal_MSE_error(rotations, recovered_rots)
    print(f"Mean error is {error}")
    assert error < tol


@pytest.mark.parametrize("problem_cls", test_classes)
@pytest.mark.parametrize("patches,min_overlap", zip(patches_list, min_overlap))
def test_calc_synchronised_rotations(problem_cls, patches, min_overlap):
    problem = problem_cls(patches, min_overlap=min_overlap)
    rotations = ex.rand_rotate_patches(problem)
    ex.rand_shift_patches(problem)
    recovered_rots = problem.calc_synchronised_rotations()
    error = ut.orthogonal_MSE_error(rotations, recovered_rots)
    print(f"Mean error is {error}")
    assert error < tol


@pytest.mark.xfail(reason="Noisy tests may fail, though many failures are a bad sign")
@pytest.mark.parametrize("test_class", test_classes)
@pytest.mark.parametrize("noise", noise_scales)
@pytest.mark.parametrize("patches,min_overlap", zip(patches_list, min_overlap))
def test_noisy_calc_synchronised_rotations(noise, test_class, patches, min_overlap):
    problem = test_class(patches, min_overlap=min_overlap)
    rotations = ex.rand_rotate_patches(problem)
    ex.add_noise(problem, noise)
    ex.rand_shift_patches(problem)
    relative_error = [np.linalg.norm(rotations[i] @ rotations[j].T
                                     - ut.relative_orthogonal_transform(problem.patches[i].get_coordinates(ov_indx),
                                                                        problem.patches[j].get_coordinates(ov_indx))
                                     )
                      for (i, j), ov_indx in problem.patch_overlap.items()]

    max_err = max(relative_error)
    min_err = min(relative_error)
    mean_err = mean(relative_error)

    recovered_rots = problem.calc_synchronised_rotations()
    problem.rotate_patches(rotations=[r.T for r in recovered_rots])
    error = ut.orthogonal_MSE_error(rotations, recovered_rots)
    print(f"Mean rotation error is {error}")
    print(f"Error of relative rotations is min: {min_err}, mean: {mean_err}, max: {max_err}")
    assert error < max(max_err, tol)


@pytest.mark.parametrize("problem_cls", test_classes)
@pytest.mark.parametrize("patches,min_overlap", zip(patches_list, min_overlap))
def test_calc_synchronised_scales(problem_cls, patches, min_overlap):
    problem = problem_cls(patches, min_overlap=min_overlap)
    scales = ex.rand_scale_patches(problem)
    ex.rand_shift_patches(problem)
    recovered_scales = problem.calc_synchronised_scales()
    rel_scales = scales / recovered_scales
    error = ut.transform_error(rel_scales)
    print(f"Mean error is {error}")
    assert error < tol


@pytest.mark.xfail(reason="Noisy tests may fail, though many failures are a bad sign")
@pytest.mark.parametrize("problem_cls", test_classes)
@pytest.mark.parametrize("noise", noise_scales)
@pytest.mark.parametrize("patches,min_overlap", zip(patches_list, min_overlap))
def test_noisy_calc_synchronised_scales(problem_cls, noise, patches, min_overlap):
    problem = problem_cls(patches, min_overlap=min_overlap)
    scales = ex.rand_scale_patches(problem)
    ex.add_noise(problem, noise, scales)
    ex.rand_shift_patches(problem)
    relative_error = []
    for (i, j), ov_indx in problem.patch_overlap.items():
        ratio = scales[i] / scales[j]
        relative_error.append(np.linalg.norm(ratio
                                             - ut.relative_scale(problem.patches[i].get_coordinates(ov_indx),
                                                                 problem.patches[j].get_coordinates(ov_indx))))
    max_err = max(relative_error)
    min_err = min(relative_error)
    mean_err = mean(relative_error)
    recovered_scales = problem.calc_synchronised_scales()
    rel_scales = scales / recovered_scales
    error = ut.transform_error(rel_scales)
    print(f"Mean error is {error}")
    print(f"Error of relative rotations is min: {min_err}, mean: {mean_err}, max: {max_err}")
    assert error < max_err + tol


@pytest.mark.parametrize("problem_cls", test_classes)
@pytest.mark.parametrize("patches,min_overlap", zip(patches_list, min_overlap))
def test_calc_synchronised_translations(problem_cls, patches, min_overlap):
    problem = problem_cls(patches, min_overlap=min_overlap)
    translations = ex.rand_shift_patches(problem)
    recovered_translations = problem.calc_synchronised_translations()
    error = ut.transform_error(translations+recovered_translations)
    print(f"Mean error is {error}")
    assert error < tol


@pytest.mark.xfail(reason="Noisy tests may fail, though many failures are a bad sign")
@pytest.mark.parametrize("noise", noise_scales)
@pytest.mark.parametrize("patches,min_overlap", zip(patches_list, min_overlap))
def test_noisy_calc_synchronised_translations(noise, patches, min_overlap):
    problem = ut.AlignmentProblem(patches, min_overlap=min_overlap)
    translations = ex.rand_shift_patches(problem)
    ex.add_noise(problem, noise)
    recovered_translations = problem.calc_synchronised_translations()
    error = ut.transform_error(translations+recovered_translations)
    print(f"Mean error is {error}")
    assert error < noise + tol


@pytest.mark.parametrize("problem_cls", test_classes)
@pytest.mark.parametrize("patches,min_overlap,points", zip(patches_list, min_overlap, points_list))
def test_get_aligned_embedding(problem_cls, patches, min_overlap, points):
    problem = problem_cls(patches, min_overlap=min_overlap)
    ex.rand_shift_patches(problem)
    ex.rand_rotate_patches(problem)
    ex.rand_scale_patches(problem)
    recovered = problem.get_aligned_embedding(scale=True)
    error = ut.procrustes_error(points, recovered)
    print(f"Procrustes error is {error}")
    assert error < tol


@pytest.mark.xfail(reason="Noisy tests may fail, though many failures are a bad sign")
@pytest.mark.parametrize("problem_cls", test_classes)
@pytest.mark.parametrize("it", range(3))
@pytest.mark.parametrize("noise", noise_scales)
@pytest.mark.parametrize("patches,min_overlap,points", zip(patches_list, min_overlap, points_list))
def test_noisy_get_aligned_embedding(problem_cls, noise, it, patches, min_overlap,points):
    iter_seed(it)
    problem = problem_cls(patches, min_overlap=min_overlap)
    ex.add_noise(problem, noise)
    shifts = ex.rand_shift_patches(problem)
    rotations = ex.rand_rotate_patches(problem)
    # scales = ex.rand_scale_patches(problem)
    # problem.save_patches("test_patches.json")
    noise_error = max(ut.procrustes_error(patch.coordinates, noisy_patch.coordinates)
                      for patch, noisy_patch in zip(patches, problem.patches))
    problem_before = copy(problem)
    recovered = problem.get_aligned_embedding(scale=False)
    error = ut.procrustes_error(points, recovered)
    print(f"Procrustes error is {error}, patch max: {noise_error}")

    relative_error = [np.linalg.norm(rotations[i] @ rotations[j].T
                                     - ut.relative_orthogonal_transform(problem_before.patches[i].get_coordinates(ov_indx),
                                                                        problem_before.patches[j].get_coordinates(ov_indx))
                                     )
                      for (i, j), ov_indx in problem_before.patch_overlap.items()]
    print(f"rotation error: {ut.transform_error(problem.rotations)}, input max: {max(relative_error)}")
    print(f"translation error: {ut.transform_error(problem.shifts)}")

    # store configuration for failed tests
    # if error > max(noise_error, tol):
    #     folder = Path().cwd()
    #     folder /= f"test_noisy_get_aligned_embedding[{noise}_{type(problem).__name__}_{it}]"
    #     folder.mkdir(exist_ok=True)
    #     problem_before.save_patches(folder / "noisy_patches.json")
    #
    #     with open(folder / 'gt_positions.csv', 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerows(points)
    #
    #     folder /= 'gt_rotations'
    #     folder.mkdir(exist_ok=True)
    #     for i, rot in enumerate(rotations):
    #         with open(folder / f"gt_rot_patch_{i+1}.csv", 'w', newline='') as f:
    #             writer = csv.writer(f)
    #             writer.writerows(rot)
    assert error < max(noise_error, tol)


if __name__ == '__main__':
    # run integration test as script (e.g. for profiling)
    import sys

    pytest.main(sys.argv)
