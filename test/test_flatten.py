import hiermodelutils as hmu
import numpy as np
from copy import copy
import pytest

def test_flatten_and_condense():
    mask = np.array([[True, True, False], [True, False, True]])

    array = np.arange(4).reshape(4, 1, 1)
    # original_masked_data = np.array([[[-1, -1, 0], [-1, 0, -1]], [[-1, -1, 1], [-1, 1, -1]], [[-1, -1, 2], [-1, 2, -1]], [[-1, -1, 3], [-1, 3, -1]]])
    compare_with = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    condensed = hmu.flatten_and_condense(array, mask=mask)
    assert np.all(condensed == compare_with)

    array = np.arange(4).reshape(4, 1, 1, 1)
    compare_with = np.array([[[0, 0]], [[1, 1]], [[2, 2]], [[3, 3]]])
    condensed = hmu.flatten_and_condense(array, mask=mask)
    assert np.all(condensed == compare_with)

    array = np.array([[1, 2, 3]])
    compare_with = np.array([[3, 2]])
    condensed = hmu.flatten_and_condense(array, mask=mask)
    assert np.all(condensed == compare_with)

    array = np.arange(2).reshape(1, 2, 1)
    compare_with = np.array([[0, 1]])
    condensed = hmu.flatten_and_condense(array, mask=mask)
    assert np.all(condensed == compare_with)

    array = np.arange(3)
    compare_with = np.array([[2, 1]])
    condensed = hmu.flatten_and_condense(array, mask=mask)
    assert np.all(condensed == compare_with)

    array = np.arange(1,25).reshape(4, 1, 2, 3)
    condensed = hmu.flatten_and_condense(array, mask, ignore_ndim=1)
    reconstructed = hmu.unflatten_and_expand(condensed, mask)
    assert reconstructed.shape == (4, 1, 2, 3)
    assert np.all(reconstructed[:, :, mask] == 0)

    array = np.arange(1,7).reshape(1, 2, 3)
    condensed = hmu.flatten_and_condense(array, mask, ignore_ndim=1)
    reconstructed = hmu.unflatten_and_expand(condensed, mask)
    assert reconstructed.shape == (1, 2, 3)
    assert np.all(reconstructed[:, mask] == 0)

    array = np.array([[1, 1, 3]])  # shape: (1, 3)
    condensed = hmu.flatten_and_condense(array, mask, ignore_ndim=2)
    reconstructed = hmu.unflatten_and_expand(condensed, mask)
    assert reconstructed.shape == (1, 3, 2, 3)
    assert np.all(reconstructed[:, :, mask] == 0)

def test_unflatten_and_expand():
    mask = np.array([[True, True, False], [True, False, True]])

    array = np.arange(1, 2+1)  # Original array shape was: (2, 3)
    compare_with = np.array([[0, 0, 1], [0, 2, 0]])
    reconstructed = hmu.unflatten_and_expand(array, mask=mask)
    assert np.all(reconstructed == compare_with)

    array = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # Original array shape was: (4, 2, 3)
    compare_with = np.array([[[0, 0, 1], [0, 2, 0]], [[0, 0, 3], [0, 4, 0]], [[0, 0, 5], [0, 6, 0]], [[0, 0, 7], [0, 8, 0]]])
    reconstructed = hmu.unflatten_and_expand(array, mask=mask)
    assert np.all(reconstructed == compare_with)

    array = np.arange(1, 2*1*4*2+1).reshape(2, 1, 4, 2)  # Original array shape was: (2, 1, 4, 2, 3)
    compare_with = np.array([[[[[0, 0, 1], [0, 2, 0]], [[0, 0, 3], [0, 4, 0]], [[0, 0, 5], [0, 6, 0]], [[0, 0, 7], [0, 8, 0]]]], [[[[0, 0, 9], [0, 10, 0]], [[0, 0, 11], [0, 12, 0]], [[0, 0, 13], [0, 14, 0]], [[0, 0, 15], [0, 16, 0]]]]])
    reconstructed = hmu.unflatten_and_expand(array, mask=mask)
    assert np.all(reconstructed == compare_with)

def test_flatten_unflatten_integration():
    def do_test(array, mask):
        condensed = hmu.flatten_and_condense(array, mask=mask)
        reconstructed = hmu.unflatten_and_expand(condensed, mask=mask)

        if array.ndim <= mask.ndim:
            masked_array = copy(np.broadcast_to(array, mask.shape))
            masked_array[mask] = 0
            assert condensed.ndim == 1
            
        else: # array.ndim > mask.ndim
            masked_array = copy(array)
            masked_array[np.broadcast_to(mask, array.shape)] = 0
            assert condensed.ndim == np.abs(array.ndim - mask.ndim) + 1

        assert np.all(reconstructed == masked_array)

    # mask = np.array(np.random.randint(0, 2, (4, 2, 3)), dtype=bool)
    mask = \
        np.array([[[ True,  True,  True],
                   [ True,  True, False]],
                  [[ True,  True, False],
                   [False,  True, False]],
                  [[False, False,  True],
                   [False, False,  True]],
                  [[ True, False,  True],
                   [False, False, False]]])
    
    # mask = \
    #     np.array([[[ True1,  True2,  True3],
    #                [ True4,  True5, False6]],
    #               [[ True7,  True8, False9],
    #                [False10,  True11, False12]],
    #               [[False13, False14,  True15],
    #                [False16, False17,  True18]],
    #               [[ True19, False20,  True21],
    #                [False22, False23, False24]]])
    
    array = np.arange(24).reshape(4, 2, 3)
    do_test(array, mask)

    array = np.arange(48).reshape(2, 4, 2, 3)
    do_test(array, mask)

    array = np.arange(336).reshape(7, 2, 4, 2, 3)
    do_test(array, mask)

    array = np.arange(6).reshape(2, 3)
    do_test(array, mask)

    array = np.arange(3)
    do_test(array, mask)

    array = np.array(1.5)
    do_test(array, mask)

    # These need custom test code (do_test won't work as is)
    array = np.arange(1, 4*2*3+1).reshape(1, 1, 4, 2, 3)
    compare_with = \
        np.array([[[ 0,  0,  0],
                   [ 0,  0,  6]],
                  [[ 0,  0,  9],
                   [10,  0, 12]],
                  [[13, 14,  0],
                   [16, 17,  0]],
                  [[ 0, 20,  0],
                   [22, 23, 24]]])
    reconstructed = hmu.unflatten_and_expand(hmu.flatten_and_condense(array, mask), mask)
    assert reconstructed.shape == (1, 1, 4, 2, 3)
    assert np.all(reconstructed == compare_with)

    array = np.arange(1, 12+1).reshape(1, 1, 4, 1, 3)
    compare_with = \
        np.array([[[ 0,  0,  0],
                   [ 0,  0,  3]],
                  [[ 0,  0,  6],
                   [ 4,  0,  6]],
                  [[ 7,  8,  0],
                   [ 7,  8,  0]],
                  [[ 0, 11,  0],
                   [10, 11, 12]]])
    reconstructed = hmu.unflatten_and_expand(hmu.flatten_and_condense(array, mask), mask)
    assert reconstructed.shape == (1, 1, 4, 2, 3)
    assert np.all(reconstructed == compare_with)

    array = np.array([[[[[1], [2]]]], [[[[3], [4]]]]])  # Original array shape was: (2, 1, 1, 2, 1)
    compare_with = \
      np.array([[[[[ 0,  0,  0],
                   [ 0,  0,  2]],
                  [[ 0,  0,  1],
                   [ 2,  0,  2]],
                  [[ 1,  1,  0],
                   [ 2,  2,  0]],
                  [[ 0,  1,  0],
                   [ 2,  2,  2]]]],
                [[[[ 0,  0,  0],
                   [ 0,  0,  4]],
                  [[ 0,  0,  3],
                   [ 4,  0,  4]],
                  [[ 3,  3,  0],
                   [ 4,  4,  0]],
                  [[ 0,  3,  0],
                   [ 4,  4,  4]]]]])
    reconstructed = hmu.unflatten_and_expand(hmu.flatten_and_condense(array, mask), mask)
    assert reconstructed.shape == (2, 1, 4, 2, 3)
    assert np.all(reconstructed == compare_with)

#     # # TODO: Add more tests here.

#     # # TODO: Add tests for unflatten with ignore_ndim.


def test_get_full_and_extra_shape():

    array_shape = (4, 2, 3)
    mask_shape = (4, 2, 3)
    full_shape, extra_shape = hmu._flatten._get_full_and_extra_shape(array_shape, mask_shape)
    assert full_shape == (4, 2, 3)
    assert extra_shape == ()

    array_shape = (2, 4, 2, 3)
    mask_shape = (4, 2, 3)
    full_shape, extra_shape = hmu._flatten._get_full_and_extra_shape(array_shape, mask_shape)
    assert full_shape == (2, 4, 2, 3)
    assert extra_shape == (2,)

    array_shape = (7, 2, 4, 2, 3)
    mask_shape = (4, 2, 3)
    full_shape, extra_shape = hmu._flatten._get_full_and_extra_shape(array_shape, mask_shape)
    assert full_shape == (7, 2, 4, 2, 3)
    assert extra_shape == (7, 2)

    array_shape = (2, 3)
    mask_shape = (4, 2, 3)
    full_shape, extra_shape = hmu._flatten._get_full_and_extra_shape(array_shape, mask_shape)
    assert full_shape == (4, 2, 3)
    assert extra_shape == ()

    array_shape = (3,)
    mask_shape = (4, 2, 3)
    full_shape, extra_shape = hmu._flatten._get_full_and_extra_shape(array_shape, mask_shape)
    assert full_shape == (4, 2, 3)
    assert extra_shape == ()

    array_shape = (1, 1, 1, 1, 3)
    mask_shape = (4, 2, 3)
    full_shape, extra_shape = hmu._flatten._get_full_and_extra_shape(array_shape, mask_shape)
    assert full_shape == (1, 1, 4, 2, 3)
    assert extra_shape == (1, 1)

    array_shape = (10, 1, 1, 1, 3)
    mask_shape = (4, 2, 3)
    full_shape, extra_shape = hmu._flatten._get_full_and_extra_shape(array_shape, mask_shape)
    assert full_shape == (10, 1, 4, 2, 3)
    assert extra_shape == (10, 1)

def test_insert_dims_into_array():
    array = np.zeros((4, 2, 3))
    mask_shape = (4, 2, 3)
    ignore_ndim = 0
    array = hmu._flatten._insert_dims_into_array(array, mask_shape, ignore_ndim)
    assert array.shape == (4, 2, 3)

    array = np.zeros((4, 2, 3))
    mask_shape = (4, 2, 3)
    ignore_ndim = 2
    array = hmu._flatten._insert_dims_into_array(array, mask_shape, ignore_ndim)
    assert array.shape == (4, 2, 1, 1, 3)

    array = np.zeros((4, 2, 3))
    mask_shape = (4, 2, 3)
    ignore_ndim = 3
    array = hmu._flatten._insert_dims_into_array(array, mask_shape, ignore_ndim)
    assert array.shape == (4, 2, 3, 1, 1, 1)

    with pytest.raises(ValueError):
        array = np.zeros((4, 2, 3))
        mask_shape = (4, 2, 3)
        array = hmu._flatten._insert_dims_into_array(array, mask_shape, -1)
        array = hmu._flatten._insert_dims_into_array(array, mask_shape, 4)
    
    array = np.zeros((2, 2, 4, 2, 3))
    mask_shape = (4, 2, 3)
    ignore_ndim = 1
    array = hmu._flatten._insert_dims_into_array(array, mask_shape, ignore_ndim)
    assert array.shape == (2, 2, 4, 2, 3)

    array = np.zeros((2, 2, 4, 2, 3))
    mask_shape = (4, 2, 3)
    ignore_ndim = 4
    array = hmu._flatten._insert_dims_into_array(array, mask_shape, ignore_ndim)
    assert array.shape == (2, 2, 4, 2, 1, 1, 3)

    array = np.zeros((10,))
    mask_shape = (4, 2, 3)
    ignore_ndim = 1
    array = hmu._flatten._insert_dims_into_array(array, mask_shape, ignore_ndim)
    assert array.shape == (10, 1, 1, 1)