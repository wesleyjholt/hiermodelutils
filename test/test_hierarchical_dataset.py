import hiermodelutils as hmu
import jax.tree_util as jtu
import numpy as np
import pandas as pd
from collections import OrderedDict
import equinox as eqx
from copy import deepcopy
from test.cases import *

case_data = [get_case_data(i) for i in range(1, 5)]

def _array_equal(a, b, equal_nan=True):
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            return np.allclose(a, b, equal_nan=equal_nan) & (a.dtype == b.dtype)
        else:
            return np.array_equal(a, b) & ((a.dtype == b.dtype) | (np.issubdtype(a.dtype, str) & np.issubdtype(b.dtype, str)))
    else:
        raise ValueError('a and b must be numpy arrays')

def _tree_equal(a, b, equal_nan=True):
    are_structures_equal = (jtu.tree_structure(a) == jtu.tree_structure(b))
    are_leaves_equal = []
    for leaf1, leaf2 in zip(jtu.tree_leaves(a), jtu.tree_leaves(b)):
        are_leaves_equal.append(_array_equal(np.array(leaf1), np.array(leaf2), equal_nan=equal_nan))
    are_leaves_equal = np.all(are_leaves_equal)
    return are_structures_equal & are_leaves_equal

# TODO: Make it so that this method detects when there is a response with None and handles it properly. (i.e., doesn't make an entry for that response in that response dictionary)
# def test_from_dataframe():
#     for d in case_data:
#         tmp = hmu.HierarchicalDataset.from_dataframe(
#             d['dfs'],
#             d['attribute_names'],
#             d['response_names']
#         )
#         # print()
#         # print(tmp.data[0])
#         # print()
#         # print(d['flattened_dicts'][0])
#         print()
#         print(tmp.data)
#         print(d['flattened_dicts'])
#         assert _tree_equal(tmp.data, d['flattened_dicts'])

def test_from_hierarchical_dict():
    for d in case_data:
        tmp = hmu.HierarchicalDataset.from_hierarchical_dict(
            d['hierarchical_dicts'],
            d['attribute_names'],
            d['response_names']
        )
        # print()
        # print(tmp.data[0])
        # print()
        # print(d['flattened_dicts'][0])
        assert _tree_equal(tmp.data, d['flattened_dicts'])

def test_get_attribute_categories():
    for d in case_data:
        for i in range(len(d['attribute_names'])):
            tmp = hmu.HierarchicalDataset(d['flattened_dicts'], d['attribute_names'], d['response_names'], share_attribute_categories_to_depth=i)
            assert _tree_equal(tmp.attribute_categories, d['attribute_categories_all_depths'][i])

def test_get_max_replicates():
    for d in case_data:
        tmp = hmu.HierarchicalDataset(d['flattened_dicts'], d['attribute_names'], d['response_names'])
        for i, di in enumerate(tmp.data):
            assert tmp._get_max_replicates(di) == tmp.max_replicates[i]

def test_get_hierarchical_lists_recursive():
    d = case_data[3]
    hier_dataset = hmu.HierarchicalDataset(d['flattened_dicts'], d['attribute_names'], d['response_names'], share_attribute_categories_to_depth=2)
    n_a = len(hier_dataset.attribute_names) + 1
    n_r = len(hier_dataset.response_names)
    dataset_index = 0
    main_attribute, mask_attribute, main_response, mask_response = \
        hier_dataset._get_hierarchical_lists_recursive(
            dataset_index,
            [[] for _ in range(n_a)],
            [[] for _ in range(n_a)],
            [[] for _ in range(n_r)],
            [[] for _ in range(n_r)],
            hier_dataset.data[0],
            0,
            len(hier_dataset.attribute_names)
        )
    main_attribute_ref = [arr.data for arr in d['hierarchical_attribute_arrays'][dataset_index]]
    mask_attribute_ref = [arr.mask for arr in d['hierarchical_attribute_arrays'][dataset_index]]
    main_response_ref = [arr.data for arr in d['hierarchical_response_arrays'][dataset_index]]
    mask_response_ref = [arr.mask for arr in d['hierarchical_response_arrays'][dataset_index]]
    for i in range(n_a):
        print(np.array(main_attribute[i]))
        print(main_attribute_ref[i])
        print(np.array(main_attribute[i]).dtype)
        print(main_attribute_ref[i].dtype)
        print(np.array(mask_attribute[i]))
        print(mask_attribute_ref[i])
        print(np.array(mask_attribute[i]).dtype)
        print(mask_attribute_ref[i].dtype)
        print(_array_equal(np.array(main_attribute[i]), main_attribute_ref[i]))
        print(_array_equal(np.array(mask_attribute[i]), mask_attribute_ref[i]))
        assert _array_equal(np.array(main_attribute[i]), main_attribute_ref[i])
        assert _array_equal(np.array(mask_attribute[i]), mask_attribute_ref[i])
    for i in range(n_r):
        assert _array_equal(np.array(main_response[i]), main_response_ref[i])
        assert _array_equal(np.array(mask_response[i]), mask_response_ref[i])


# schools = [1, 2, 3]
# classes = [['A', 'B'], 
#            ['A'], 
#            ['A', 'B', 'C']]
# students = [[['a', 'b', 'c'], 
#              ['a', 'b']],
#             [['a', 'b', 'c', 'd', 'e']],
#             [['a', 'b'], 
#              ['a', 'b', 'c', 'd'], 
#              ['a']]]
# test_score_raw = [[[1, 2, 3],
#                    [4, 5]],
#                   [[6, 7, 8, 9, 10]],
#                   [[11, 12],
#                    [13, 14, 15, 16],
#                    [17]]]
# test_score_percent = [[[1/20, 2/20, 3/20],
#                        [4/20, 5/20]],
#                       [[6/20, 7/20, 8/20, 9/20, 10/20]],
#                       [[11/20, 12/20],
#                        [13/20, 14/20, 15/20, 16/20],
#                        [17/20]]]
# long_data = []
# for i, school in enumerate(schools):
#     for j, cls in enumerate(classes[i]):
#         for k, student in enumerate(students[i][j]):
#             long_data.append([school, cls, student, test_score_raw[i][j][k], test_score_percent[i][j][k]])
# df = pd.DataFrame(long_data, columns=['school', 'class', 'student', 'test_score_raw', 'test_score_percent'])

# def test_create_nested_dataset():
#     nested_data_1 = hmu._long_to_wide.create_nested_dataset(df, ['school', 'class', 'student'])
#     nested_data_2 = hmu._long_to_wide._nest_data_recursive(OrderedDict(), df, ['school', 'class', 'student'])
#     nested_data_ref = OrderedDict(
#         {
#             1: OrderedDict(
#                 {
#                     'A': OrderedDict(
#                         {
#                             'a': np.array([1, 1/20]),
#                             'b': np.array([2, 2/20]),
#                             'c': np.array([3, 3/20])
#                         }
#                     ),
#                     'B': OrderedDict(
#                         {
#                             'a': np.array([4, 4/20]),
#                             'b': np.array([5, 5/20])
#                         }
#                     )
#                 }
#             ),
#             2: OrderedDict(
#                 {
#                     'A': OrderedDict(
#                         {
#                             'a': np.array([6, 6/20]),
#                             'b': np.array([7, 7/20]),
#                             'c': np.array([8, 8/20]),
#                             'd': np.array([9, 9/20]),
#                             'e': np.array([10, 10/20])
#                         }
#                     )
#                 }
#             ),
#             3: OrderedDict(
#                 {
#                     'A': OrderedDict(
#                         {
#                             'a': np.array([11, 11/20]),
#                             'b': np.array([12, 12/20])
#                         }
#                     ),
#                     'B': OrderedDict(
#                         {
#                             'a': np.array([13, 13/20]),
#                             'b': np.array([14, 14/20]),
#                             'c': np.array([15, 15/20]),
#                             'd': np.array([16, 16/20])
#                         }
#                     ),
#                     'C': OrderedDict(
#                         {
#                             'a': np.array([17, 17/20])
#                         }
#                     )
#                 }
#             )
#         }
#     )
#     assert eqx.tree_equal(nested_data_1, nested_data_2)
#     assert eqx.tree_equal(nested_data_1, nested_data_ref)

# def test_get_array_shapes():
#     # TODO: Test more cases
#     array_shape = hmu._long_to_wide.get_array_shapes([df], ['school', 'class', 'student'], 0, [None, None, None])
#     array_shape_ref = [[3, 3, 5, 1]]
#     assert array_shape == array_shape_ref

# def test_init_arrays():
#     array_shape = [3, 3, 5, 1]
#     arrays = hmu._long_to_wide._init_arrays(
#         array_shape, 
#         [True, 0, '', np.nan], 
#         dtypes=[bool, int, str, float]
#     )
#     arrays_ref = [
#         np.full((3,), True),
#         np.full((3, 3), 0),
#         np.full((3, 3, 5), ''),
#         np.full((3, 3, 5, 1), np.nan)
#     ]
#     for i in range(len(arrays)):
#         assert _array_equal(arrays[i], arrays_ref[i])

# def test_populate_arrays_in_place():
#     arrays = [[
#         np.full((3,), 0),
#         np.full((3, 3), '', dtype='<U100'),
#         np.full((3, 3, 5), '', dtype='<U100'),
#         np.full((3, 3, 5, 1), np.nan)
#     ]]
#     masks = [[
#         np.full((3,), False),
#         np.full((3, 3), False),
#         np.full((3, 3, 5), False),
#         np.full((3, 3, 5, 1), False)
#     ]]
#     nested_data = OrderedDict(
#         {
#             1: OrderedDict(
#                 {
#                     'A': OrderedDict(
#                         {
#                             'a': np.array([1/20]),
#                             'b': np.array([2/20]),
#                             'c': np.array([3/20])
#                         }
#                     ),
#                     'B': OrderedDict(
#                         {
#                             'a': np.array([4/20]),
#                             'b': np.array([5/20])
#                         }
#                     )
#                 }
#             ),
#             2: OrderedDict(
#                 {
#                     'A': OrderedDict(
#                         {
#                             'a': np.array([6/20]),
#                             'b': np.array([7/20]),
#                             'c': np.array([8/20]),
#                             'd': np.array([9/20]),
#                             'e': np.array([10/20])
#                         }
#                     )
#                 }
#             ),
#             3: OrderedDict(
#                 {
#                     'A': OrderedDict(
#                         {
#                             'a': np.array([11/20]),
#                             'b': np.array([12/20])
#                         }
#                     ),
#                     'B': OrderedDict(
#                         {
#                             'a': np.array([13/20]),
#                             'b': np.array([14/20]),
#                             'c': np.array([15/20]),
#                             'd': np.array([16/20])
#                         }
#                     ),
#                     'C': OrderedDict(
#                         {
#                             'a': np.array([17/20])
#                         }
#                     )
#                 }
#             )
#         }
#     )
#     arrays_ref = [
#         np.array([1, 2, 3], dtype=int),
#         np.array([['A', 'B', ''], ['A', '', ''], ['A', 'B', 'C']], dtype='<U100'),
#         np.array([[['a', 'b', 'c', '', ''], ['a', 'b', '', '', ''], ['', '', '', '', '']], 
#                   [['a', 'b', 'c', 'd', 'e'], ['', '', '', '', ''], ['', '', '', '', '']], 
#                   [['a', 'b', '', '', ''], ['a', 'b', 'c', 'd', ''], ['a', '', '', '', '']]], dtype='<U100'),
#         np.array([[[1/20, 2/20, 3/20, np.nan, np.nan], [4/20, 5/20, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan]],
#                   [[6/20, 7/20, 8/20, 9/20, 10/20], [np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan, np.nan]],
#                   [[11/20, 12/20, np.nan, np.nan, np.nan], [13/20, 14/20, 15/20, 16/20, np.nan], [17/20, np.nan, np.nan, np.nan, np.nan]]], dtype=float)[..., None]
#     ]
#     masks_ref = [
#         np.array([True, True, True]),
#         np.array([[True, True, False], [True, False, False], [True, True, True]]),
#         np.array([[[True, True, True, False, False], [True, True, False, False, False], [False, False, False, False, False]], 
#                   [[True, True, True, True, True], [False, False, False, False, False], [False, False, False, False, False]], 
#                   [[True, True, False, False, False], [True, True, True, True, False], [True, False, False, False, False]]]),
#         np.array([[[True, True, True, False, False], [True, True, False, False, False], [False, False, False, False, False]],
#                   [[True, True, True, True, True], [False, False, False, False, False], [False, False, False, False, False]],
#                   [[True, True, False, False, False], [True, True, True, True, False], [True, False, False, False, False]]])[..., None]
#     ]
#     hmu._long_to_wide._populate_arrays_in_place(arrays, masks, [nested_data], 1, lambda x: x, ['', 0, 0, np.nan], 0, [None, None, None])
#     a = arrays[0]
#     m = masks[0]
#     for i in range(len(a)):
#         assert _array_equal(a[i], arrays_ref[i])
#         assert _array_equal(m[i], masks_ref[i])
            

if __name__=='__main__':
    test_get_attribute_categories()