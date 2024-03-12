import numpy as np
import pandas as pd
from collections import OrderedDict

get_case_data = lambda c: globals()[f"get_case_{c}_data"]()

# CASE 1:
def get_case_1_data():
    attr = np.array(["1"])
    val = np.array([1])
    df = pd.DataFrame({'attr': attr, 'val': val})
    flattened_dicts = [OrderedDict({("1",): (OrderedDict({"val": 1}),)})]
    tabular_attribute_arrays = [[attr]]
    tabular_response_arrays = [[val]]
    hierarchical_dict = OrderedDict({"1": (OrderedDict({"val": 1}),)})
    hierarchical_attribute_arrays = [[np.array(["1"])]]  # One dataset, one attribute, one category
    hierarchical_response_arrays = [[np.array([[1]])]]  # One dataset, one attribute, one replicate, one response type
    attribute_categories = [[["1"]]]
    attribute_categories_all_depths = {0: attribute_categories}
    max_replicates_each_dataset = [1]
    return dict(dfs=[df], 
                flattened_dicts=flattened_dicts,
                tabular_attribute_arrays=tabular_attribute_arrays,
                tabular_response_arrays=tabular_response_arrays,
                hierarchical_dicts=[hierarchical_dict], 
                hierarchical_attribute_arrays=hierarchical_attribute_arrays,
                hierarchical_response_arrays=hierarchical_response_arrays,
                attribute_names=["attr"], 
                response_names=["val"],
                attribute_categories_all_depths=attribute_categories_all_depths,
                max_replicates_each_dataset=max_replicates_each_dataset)

# CASE 2:
def get_case_2_data():
    attr = np.array([""])
    val = np.array([np.nan])
    df = pd.DataFrame({'attr': attr, 'val': val})
    flattened_dicts = [OrderedDict({("",): (OrderedDict({"val": np.nan}),)})]
    tabular_attribute_arrays = [[attr]]  # One dataset, one attribute, one category
    tabular_response_arrays = [[val]]  # One dataset, one attribute, one replicate, one response type
    hierarchical_dict = OrderedDict({"": (OrderedDict({"val": np.nan}),)})
    hierarchical_attribute_arrays = [[np.array([""])]]
    hierarchical_response_arrays = [[np.array([[np.nan]])]]
    attribute_categories = [[[""]]]
    attribute_categories_all_depths = {0: attribute_categories}
    max_replicates_each_dataset = [1]
    return dict(dfs=[df], 
                flattened_dicts=flattened_dicts,
                tabular_attribute_arrays=tabular_attribute_arrays,
                tabular_response_arrays=tabular_response_arrays,
                hierarchical_dicts=[hierarchical_dict], 
                hierarchical_attribute_arrays=hierarchical_attribute_arrays,
                hierarchical_response_arrays=hierarchical_response_arrays,
                attribute_names=["attr"], 
                response_names=["val"],
                attribute_categories_all_depths=attribute_categories_all_depths,
                max_replicates_each_dataset=max_replicates_each_dataset)

# CASE 3:
def get_case_3_data():
    attr = np.array([True, False])
    val = np.array([1, 2])
    df = pd.DataFrame({'attr': attr, 'val': val})
    flattened_dicts = [OrderedDict({(False,): (OrderedDict({"val": 2}),), (True,): (OrderedDict({"val": 1}),)})]
    tabular_attribute_arrays = [[attr]]  # One dataset, one attribute, two categories
    tabular_response_arrays = [[val]]  # One dataset, one attribute, one replicate, one response type
    hierarchical_dict = OrderedDict({False: (OrderedDict({"val": 2}),), True: (OrderedDict({"val": 1}),)})
    hierarchical_attribute_arrays = [[np.array([True, False])]]
    hierarchical_response_arrays = [[np.array([[1], [2]])]]
    attribute_categories = [[[False, True]]]
    attribute_categories_all_depths = {0: attribute_categories}
    max_replicates_each_dataset = [1]
    return dict(dfs=[df], 
                flattened_dicts=flattened_dicts,
                tabular_attribute_arrays=tabular_attribute_arrays,
                tabular_response_arrays=tabular_response_arrays,
                hierarchical_dicts=[hierarchical_dict], 
                hierarchical_attribute_arrays=hierarchical_attribute_arrays,
                hierarchical_response_arrays=hierarchical_response_arrays,
                attribute_names=["attr"], 
                response_names=["val"],
                attribute_categories_all_depths=attribute_categories_all_depths,
                max_replicates_each_dataset=max_replicates_each_dataset)

# CASE 4:
def get_case_4_data():
    dataset1_attr1 = np.array(["1", "1", "2", "2", "3", "3", "4"])
    dataset1_attr2 = np.array(["A", "B", "A", "A", "B", "B", "C"])
    dataset1_vals1 = np.linspace(0, 1, 7)
    dataset1_vals2 = np.arange(7)
    df1 = pd.DataFrame({'attr1': dataset1_attr1, 'attr2': dataset1_attr2, 'vals1': dataset1_vals1, 'vals2': dataset1_vals2})

    dataset2_attr1 = np.array(["5", "1", "2", "3"])
    dataset2_attr2 = np.array(["A", "B", "A", "Z"])
    dataset2_vals1 = np.linspace(0, 1, 4)
    dataset2_vals2 = np.arange(4)
    df2 = pd.DataFrame({'attr1': dataset2_attr1, 'attr2': dataset2_attr2, 'vals1': dataset2_vals1, 'vals2': dataset2_vals2})

    dataset3_attr1 = np.array(["0"])
    dataset3_attr2 = np.array(["0"])
    dataset2_vals1 = np.array([0.0])
    dataset2_vals2 = np.array([0])
    df3 = pd.DataFrame({'attr1': dataset3_attr1, 'attr2': dataset3_attr2, 'vals1': dataset2_vals1, 'vals2': dataset2_vals2})

    tabular_attribute_arrays = [[dataset1_attr1, dataset1_attr2], [dataset2_attr1, dataset2_attr2], [dataset3_attr1, dataset3_attr2]]
    tabular_response_arrays = [[dataset1_vals1, dataset1_vals2], [dataset2_vals1, dataset2_vals2], [dataset2_vals1, dataset2_vals2]]

    hierarchical_dict_1 = \
        {
            "1": {
                "A": (
                    {
                        "vals1": 0.0,
                    },
                ),
                "B": (
                    {
                        "vals1": 1/6,
                        "vals2": 1
                    },
                )
            },
            "2": {
                "A": (
                    {
                        "vals1": 2/6,
                        "vals2": 2
                    },
                    {
                        "vals1": 3/6,
                        "vals2": 3
                    }
                )
            },
            "3": {
                "B": (
                    {
                        "vals1": 4/6,
                        "vals2": 4
                    },
                    {
                        "vals1": 5/6,
                        "vals2": 5
                    }
                )
            },
            "4": {
                "C": (
                    {
                        "vals1": 6/6,
                        "vals2": 6
                    },
                )
            }
        }
    
    hierarchical_dict_2 = \
        {
            "5": {
                "A": (
                    {
                        "vals1": 0.0,
                        "vals2": 0
                    },
                ),
            },
            "1": {
                "B": (
                    {
                        "vals1": 1/3,
                        "vals2": 1
                    },
                )
            },
            "2": {
                "A": (
                    {
                        "vals1": 2/3,
                        "vals2": 2
                    },
                )
            },
            "3": {
                "Z": (
                    {
                        "vals1": 1.0,
                        "vals2": 3
                    },
                )
            }
        }
    
    hierarchical_dict_3 = \
        {
            "0": {
                "0": (
                    {
                        "vals1": 0.0,
                        "vals2": 0
                    },
                ),
            }
        }

    flattened_dicts = [
        OrderedDict({
            ("1", "A"): (OrderedDict({"vals1": 0.0}),),
            ("1", "B"): (OrderedDict({"vals1": 1/6, "vals2": 1}),),
            ("2", "A"): (OrderedDict({"vals1": 2/6, "vals2": 2}), OrderedDict({"vals1": 3/6, "vals2": 3}),),
            ("3", "B"): (OrderedDict({"vals1": 4/6, "vals2": 4}), OrderedDict({"vals1": 5/6, "vals2": 5}),),
            ("4", "C"): (OrderedDict({"vals1": 6/6, "vals2": 6}),)
        }),
        OrderedDict({
            ("1", "B"): (OrderedDict({"vals1": 1/3, "vals2": 1}),),
            ("2", "A"): (OrderedDict({"vals1": 2/3, "vals2": 2}),),
            ("3", "Z"): (OrderedDict({"vals1": 1.0, "vals2": 3}),),
            ("5", "A"): (OrderedDict({"vals1": 0.0, "vals2": 0}),)
        }),
        OrderedDict({
            ("0", "0"): (OrderedDict({"vals1": 0.0, "vals2": 0}),)
        })
    ]
    
    _fill = np.nan
    _ifill = -1
    hierarchical_attribute_arrays = [
        [
            np.ma.array(
                data = ["", "1", "2", "3", "4", ""],
                mask = [True, False, False, False, False, True],
                # dtype = '<U100'
            ),
            np.ma.array(
                data = [["",  "",  "",  "", ""],
                        ["", "A", "B",  "", ""],
                        ["", "A",  "",  "", ""],
                        ["",  "", "B",  "", ""],
                        ["",  "",  "", "C", ""],
                        ["",  "",  "",  "", ""]],
                mask = [[True]*5,
                        [True, False, False, True, True],
                        [True, False, True, True, True],
                        [True, True, False, True, True],
                        [True, True, True, False, True],
                        [True]*5],
                # dtype = '<U100'
            ),
            np.ma.array(
                data = [[[_ifill, _ifill], [_ifill, _ifill], [_ifill, _ifill], [_ifill, _ifill], [_ifill, _ifill]],
                        [[_ifill, _ifill], [     0, _ifill], [     0, _ifill], [_ifill, _ifill], [_ifill, _ifill]],
                        [[_ifill, _ifill], [     0,      1], [_ifill, _ifill], [_ifill, _ifill], [_ifill, _ifill]],
                        [[_ifill, _ifill], [_ifill, _ifill], [     0,      1], [_ifill, _ifill], [_ifill, _ifill]],
                        [[_ifill, _ifill], [_ifill, _ifill], [_ifill, _ifill], [     0, _ifill], [_ifill, _ifill]],
                        [[_ifill, _ifill], [_ifill, _ifill], [_ifill, _ifill], [_ifill, _ifill], [_ifill, _ifill]]],
                mask = [[[True, True], [ True,  True], [ True,  True], [ True, True], [True, True]],
                        [[True, True], [False,  True], [False,  True], [ True, True], [True, True]],
                        [[True, True], [False, False], [ True,  True], [ True, True], [True, True]],
                        [[True, True], [ True,  True], [False, False], [ True, True], [True, True]],
                        [[True, True], [ True,  True], [ True,  True], [False, True], [True, True]],
                        [[True, True], [ True,  True], [ True,  True], [ True, True], [True, True]]],
                # dtype = int
            )
        ],
        [
            np.ma.array(
                data = ["", "1", "2", "3", "", "5"],
                mask = [True, False, False, False, True, False],
                # dtype = '<U100'
            ),
            np.ma.array(
                data = [["",  "",  "", "",  ""],
                        ["",  "", "B", "",  ""],
                        ["", "A",  "", "",  ""],
                        ["",  "",  "", "", "Z"],
                        ["",  "",  "", "",  ""],
                        ["", "A",  "", "",  ""]],
                mask = [[True]*5,
                        [True, True, False, True, True],
                        [True, False, True, True, True],
                        [True, True, True, True, False],
                        [True]*5,
                        [True, False, True, True, True]],
                # dtype = '<U100'
            ),
            np.ma.array(
                data = [[[_ifill], [_ifill], [_ifill], [_ifill], [_ifill]],
                        [[_ifill], [_ifill], [     0], [_ifill], [_ifill]],
                        [[_ifill], [     0], [_ifill], [_ifill], [_ifill]],
                        [[_ifill], [_ifill], [_ifill], [_ifill], [     0]],
                        [[_ifill], [_ifill], [_ifill], [_ifill], [_ifill]],
                        [[_ifill], [     0], [_ifill], [_ifill], [_ifill]]],
                mask = [[[True], [ True], [ True], [True], [ True]],
                        [[True], [ True], [False], [True], [ True]],
                        [[True], [False], [ True], [True], [ True]],
                        [[True], [ True], [ True], [True], [False]],
                        [[True], [ True], [ True], [True], [ True]],
                        [[True], [False], [ True], [True], [ True]]],
                # dtype = int
            )
        ],
        [
            np.ma.array(
                data = ["0", "", "", "", "", ""],
                mask = [False, True, True, True, False, False],
                # dtype = '<U100'
            ),
            np.ma.array(
                data = [["0", "", "", "", ""],
                        [ "", "", "", "", ""],
                        [ "", "", "", "", ""],
                        [ "", "", "", "", ""],
                        [ "", "", "", "", ""],
                        [ "", "", "", "", ""]],
                mask = [[False, True, True, True, True],
                        [True]*5,
                        [True]*5,
                        [True]*5,
                        [True]*5,
                        [True]*5],
                # dtype = '<U100'
            ),
            np.ma.array(
                data = [[[     0], [_ifill], [_ifill], [_ifill], [_ifill]],
                        [[_ifill], [_ifill], [_ifill], [_ifill], [_ifill]],
                        [[_ifill], [_ifill], [_ifill], [_ifill], [_ifill]],
                        [[_ifill], [_ifill], [_ifill], [_ifill], [_ifill]],
                        [[_ifill], [_ifill], [_ifill], [_ifill], [_ifill]],
                        [[_ifill], [_ifill], [_ifill], [_ifill], [_ifill]]],
                mask = [[[False], [True], [True], [True], [True]],
                        [[ True], [True], [True], [True], [True]],
                        [[ True], [True], [True], [True], [True]],
                        [[ True], [True], [True], [True], [True]],
                        [[ True], [True], [True], [True], [True]],
                        [[ True], [True], [True], [True], [True]]],
                # dtype = int
            )
        ]
    ]
    hierarchical_response_arrays = [
        [
            np.ma.array(
                data = [[[_fill, _fill], [_fill, _fill], [_fill, _fill], [_fill, _fill], [_fill, _fill]],
                        [[_fill, _fill], [  0.0, _fill], [  1/6, _fill], [_fill, _fill], [_fill, _fill]],
                        [[_fill, _fill], [  2/6,   3/6], [_fill, _fill], [_fill, _fill], [_fill, _fill]],
                        [[_fill, _fill], [_fill, _fill], [  4/6,   5/6], [_fill, _fill], [_fill, _fill]],
                        [[_fill, _fill], [_fill, _fill], [_fill, _fill], [  6/6, _fill], [_fill, _fill]]], 
                mask = [[[True, True], [ True,  True], [ True,  True], [ True, True], [True, True]],
                        [[True, True], [False,  True], [False,  True], [ True, True], [True, True]],
                        [[True, True], [False, False], [ True,  True], [ True, True], [True, True]],
                        [[True, True], [ True,  True], [False, False], [ True, True], [True, True]],
                        [[True, True], [ True,  True], [ True,  True], [False, True], [True, True]]],
            ),
            np.ma.array(
                data = [[[_fill, _fill], [_fill, _fill], [_fill, _fill], [_fill, _fill], [_fill, _fill]],
                        [[_fill, _fill], [_fill, _fill], [    1, _fill], [_fill, _fill], [_fill, _fill]],
                        [[_fill, _fill], [    2,     3], [_fill, _fill], [_fill, _fill], [_fill, _fill]],
                        [[_fill, _fill], [_fill, _fill], [    4,     5], [_fill, _fill], [_fill, _fill]],
                        [[_fill, _fill], [_fill, _fill], [_fill, _fill], [    6, _fill], [_fill, _fill]]], 
                mask = [[[True, True], [ True,  True], [ True,  True], [ True, True], [True, True]],
                        [[True, True], [False,  True], [False,  True], [ True, True], [True, True]],
                        [[True, True], [False, False], [ True,  True], [ True, True], [True, True]],
                        [[True, True], [ True,  True], [False, False], [ True, True], [True, True]],
                        [[True, True], [ True,  True], [ True,  True], [False, True], [True, True]]],
            )
        ],
        [
            np.ma.array(
                data = [[[_fill], [_fill], [_fill], [_fill], [_fill]],
                        [[_fill], [_fill], [  1/3], [_fill], [_fill]],
                        [[_fill], [  2/3], [_fill], [_fill], [_fill]],
                        [[_fill], [_fill], [_fill], [_fill], [  1.0]],
                        [[_fill], [_fill], [_fill], [_fill], [_fill]],
                        [[_fill], [  0.0], [_fill], [_fill], [_fill]]],
                mask = [[[True], [ True], [ True], [True], [ True]],
                        [[True], [ True], [False], [True], [ True]],
                        [[True], [False], [ True], [True], [ True]],
                        [[True], [ True], [ True], [True], [False]],
                        [[True], [ True], [ True], [True], [ True]],
                        [[True], [False], [ True], [True], [ True]]],
            ),
            np.ma.array(
                data = [[[_fill], [_fill], [_fill], [_fill], [_fill]],
                        [[_fill], [_fill], [    1], [_fill], [_fill]],
                        [[_fill], [    2], [_fill], [_fill], [_fill]],
                        [[_fill], [_fill], [_fill], [_fill], [    3]],
                        [[_fill], [_fill], [_fill], [_fill], [_fill]],
                        [[_fill], [    0], [_fill], [_fill], [_fill]]],
                mask = [[[True], [ True], [ True], [True], [ True]],
                        [[True], [ True], [False], [True], [ True]],
                        [[True], [False], [ True], [True], [ True]],
                        [[True], [ True], [ True], [True], [False]],
                        [[True], [ True], [ True], [True], [ True]],
                        [[True], [False], [ True], [True], [ True]]],
            ),
        ],
        [
            np.ma.array(
                data = [[[  0.0], [_fill], [_fill], [_fill], [_fill]],
                        [[_fill], [_fill], [_fill], [_fill], [_fill]],
                        [[_fill], [_fill], [_fill], [_fill], [_fill]],
                        [[_fill], [_fill], [_fill], [_fill], [_fill]],
                        [[_fill], [_fill], [_fill], [_fill], [_fill]],
                        [[_fill], [_fill], [_fill], [_fill], [_fill]]],
                mask = [[[False], [True], [True], [True], [True]],
                        [[ True], [True], [True], [True], [True]],
                        [[ True], [True], [True], [True], [True]],
                        [[ True], [True], [True], [True], [True]],
                        [[ True], [True], [True], [True], [True]],
                        [[ True], [True], [True], [True], [True]]]
            ),
            np.ma.array(
                data = [[[    0], [_fill], [_fill], [_fill], [_fill]],
                        [[_fill], [_fill], [_fill], [_fill], [_fill]],
                        [[_fill], [_fill], [_fill], [_fill], [_fill]],
                        [[_fill], [_fill], [_fill], [_fill], [_fill]],
                        [[_fill], [_fill], [_fill], [_fill], [_fill]],
                        [[_fill], [_fill], [_fill], [_fill], [_fill]]],
                mask = [[[False], [True], [True], [True], [True]],
                        [[True], [True], [True], [True], [True]],
                        [[True], [True], [True], [True], [True]],
                        [[True], [True], [True], [True], [True]],
                        [[True], [True], [True], [True], [True]],
                        [[True], [True], [True], [True], [True]]]
            ),
        ]
    ]

    attribute_categories = [
        [["1", "2", "3", "4"],
         ["A", "B", "C"]],
        [["1", "2", "3", "5"],
         ["A", "B", "Z"]],
        [["0"],
         ["0"]]
    ]
    attribute_categories_share_depth_1 = [
        [["0", "1", "2", "3", "4", "5"],
         ["A", "B", "C"]],
        [["0", "1", "2", "3", "4", "5"],
         ["A", "B", "Z"]],
        [["0", "1", "2", "3", "4", "5"],
         ["0"]]
    ]
    attribute_categories_share_depth_2 = [
        [["0", "1", "2", "3", "4", "5"],
         ["0", "A", "B", "C", "Z"]],
        [["0", "1", "2", "3", "4", "5"],
         ["0", "A", "B", "C", "Z"]],
        [["0", "1", "2", "3", "4", "5"],
         ["0", "A", "B", "C", "Z"]],
    ]
    attribute_categories_all_depths = {
        0: attribute_categories,
        1: attribute_categories_share_depth_1,
        2: attribute_categories_share_depth_2
    }

    return dict(dfs=[df1, df2, df3], 
                hierarchical_dicts=[hierarchical_dict_1, hierarchical_dict_2, hierarchical_dict_3], 
                flattened_dicts=flattened_dicts,
                tabular_attribute_arrays=tabular_attribute_arrays,
                tabular_response_arrays=tabular_response_arrays,
                hierarchical_attribute_arrays=hierarchical_attribute_arrays,
                hierarchical_response_arrays=hierarchical_response_arrays,
                attribute_names=["attr1", "attr2"], 
                response_names=["vals1", "vals2"],
                attribute_categories_all_depths=attribute_categories_all_depths,
                max_replicates_each_dataset=[2, 1, 1]
                )



## PROBABLY WILL DELETE THIS:

# # CASE 1:
# def get_case_1_data():
#     attr1 = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3])
#     attr2 = np.array(['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B', 'B', 'B', 'C'])
#     attr3 = np.array(['a', 'a', 'b', 'b', 'b', 'c', 'd', 'e', 'e', 'e', 'a', 'b', 'c', 'a', 'b', 'a', 'b', 'a', 'b', 'c', 'd', 'a'])
#     test_score_raw = np.array([6, np.nan, 7, 7, np.nan, 8, 9, 10, 10, 10, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, np.nan])
#     test_score_percent = np.array([6, 6.1, 7, np.nan, 7.1, 8, 9, 10, 10.1, 10.2, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17])
#     attribute_arrays = [[attr1, attr2, attr3]]
#     response_arrays = [[test_score_raw, test_score_percent]]
#     dfs = [pd.DataFrame({'attr1': attr1, 'attr2': attr2, 'attr3': attr3, 'test_score_raw': test_score_raw, 'test_score_percent': test_score_percent})]

#     hierarchical_dicts = \
#         [OrderedDict(
#             {
#                 2: OrderedDict(
#                     {
#                         'A': OrderedDict(
#                             {
#                                 'a': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 6,
#                                             'test_score_percent': 6/20
#                                         }
#                                     ),
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 999999,
#                                             'test_score_percent': 6.1/20
#                                         }
#                                     ),
#                                 ),
#                                 'b': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 7,
#                                             'test_score_percent': 7/20
#                                         }
#                                     ),
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 7,
#                                             'test_score_percent': 1e+20
#                                         }
#                                     ),
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 999999,
#                                             'test_score_percent': 7.1/20
#                                         }
#                                     )
#                                 ),
#                                 'c': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 8,
#                                             'test_score_percent': 8/20
#                                         }
#                                     ),
#                                 ),
#                                 'd': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 9,
#                                             'test_score_percent': 9/20
#                                         }
#                                     ),
#                                 ),
#                                 'e': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 10,
#                                             'test_score_percent': 10/20
#                                         }
#                                     ),
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 10,
#                                             'test_score_percent': 10.1/20
#                                         }
#                                     ),
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 10,
#                                             'test_score_percent': 10.2/20
#                                         }
#                                     ),
#                                 )
#                             }
#                         )
#                     }
#                 ),
#                 1: OrderedDict(
#                     {
#                         'A': OrderedDict(
#                             {
#                                 'a': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 1,
#                                             'test_score_percent': 1/20
#                                         }
#                                     ),
#                                 ),
#                                 'b': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 2,
#                                             'test_score_percent': 2/20
#                                         }
#                                     ),
#                                 ),
#                                 'c': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 3,
#                                             'test_score_percent': 3/20
#                                         }
#                                     ),
#                                 ),
#                             }
#                         ),
#                         'B': OrderedDict(
#                             {
#                                 'a': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 4,
#                                             'test_score_percent': 4/20
#                                         }
#                                     ),
#                                 ),
#                                 'b': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 5,
#                                             'test_score_percent': 5/20
#                                         }
#                                     ),
#                                 )
#                             }
#                         )
#                     }
#                 ),
#                 3: OrderedDict(
#                     {
#                         'A': OrderedDict(
#                             {
#                                 'a': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 11,
#                                             'test_score_percent': 11/20
#                                         }
#                                     ),
#                                 ),
#                                 'b': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 12,
#                                             'test_score_percent': 12/20
#                                         }
#                                     ),
#                                 )
#                             }
#                         ),
#                         'B': OrderedDict(
#                             {
#                                 'a': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 13,
#                                             'test_score_percent': 13/20
#                                         }
#                                     ),
#                                 ),
#                                 'b': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 14,
#                                             'test_score_percent': 14/20
#                                         }
#                                     ),
#                                 ),
#                                 'c': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 15,
#                                             'test_score_percent': 15/20
#                                         }
#                                     ),
#                                 ),
#                                 'd': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 16,
#                                             'test_score_percent': 16/20
#                                         }
#                                     ),
#                                 )
#                             }
#                         ),
#                         'C': OrderedDict(
#                             {
#                                 'a': (
#                                     OrderedDict(
#                                         {
#                                             'test_score_raw': 17,
#                                             'test_score_percent': 17/20
#                                         }
#                                     ),
#                                 )
#                             }
#                         )
#                     }
#                 )
#             }
#         )]

#     return dict(dfs=dfs, 
#                 hierarchical_dicts=hierarchical_dicts,
#                 attribute_arrays=attribute_arrays,
#                 response_arrays=response_arrays,
#                 attribute_names=["attr1", "attr2", "attr3"], 
#                 response_names=["test_score_raw", "test_score_percent"])
