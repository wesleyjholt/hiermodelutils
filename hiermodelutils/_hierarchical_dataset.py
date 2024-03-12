__all__ = ["HierarchicalDataset"]

from collections import OrderedDict
from copy import copy
import numpy as np
import pandas as pd
import jax.tree_util as jtu
from typing import Callable, Union, Optional
from jaxtyping import Float, Int
from dataclasses import dataclass
from functools import partial

# How to handle replicates in the tabular dataframe (if there is no "replicates" identifier)? 
# Strategy: Make each data value a tuple containing the replicates
# If there are more than one response variable, and there are two rows with all the same attributes values,
# but the responses are (1, None) and (None, 1), then we can't tell if these two entries belong to the same
# replicate or not. So we add the option on whether to treat these as the same replicate or not. The option
# is called "infer_replicates". Default is True.

def _infer_dtype_from_example(example):
    pandas_dtype = pd.api.types.infer_dtype(example)
    if pandas_dtype == 'floating':
        return float
    elif pandas_dtype == 'integer':
        return int
    elif pandas_dtype == 'boolean':
        return bool
    else:
        return object

def _get_default_value(dtype):
    if dtype == float:
        return 0.0
    elif dtype == int:
        return -1
    elif dtype == bool:
        return False
    elif dtype == object:
        return ''
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

def _infer_dtypes_and_fill_values_from_examples(examples, fixed_dtypes=None, fixed_fill_values=None):
    dtypes = ()
    fill_values = ()
    for i, x in enumerate(examples):
        if fixed_dtypes is not None:
            if fixed_dtypes[i] is not None:
                dtype = fixed_dtypes[i]
        else:
            dtype = _infer_dtype_from_example(x)
        if fixed_fill_values is not None:
            if fixed_fill_values[i] is not None:
                fill_value = fixed_fill_values[i]
        else:
            fill_value = _get_default_value(dtype)
        dtypes += (dtype,)
        fill_values += (fill_value,)
    return dtypes, fill_values

def _is_category_in_dataset(category, data):
    data_categories = list(map(lambda x: x[0], list(data.keys())))
    return category in data_categories

def _get_category_mask_counter(data, categories, counter, fill_value):
    category = categories[counter]
    if _is_category_in_dataset(category, data):
        return category, False, counter + 1
    else:
        return fill_value, True, counter + 1

def _get_replicate_response_mask_counter(data, response_names, replicate_counter, replicate_fill_value, response_fill_values):
    if len(data) > 0:  # Data is empty
        d = data[()]
        if replicate_counter < len(d):
            response = []
            response_mask = []
            for i, response_name in enumerate(response_names):
                if response_name in d[replicate_counter]:
                    response.append(d[replicate_counter][response_name])
                    response_mask.append(False)
                else:
                    response.append(response_fill_values[i])
                    response_mask.append(True)
            return replicate_counter, response, False, response_mask, replicate_counter + 1
    return replicate_fill_value, [response_fill_values[i] for i, _ in enumerate(response_names)], True, [True for _ in response_names], replicate_counter + 1

def _get_data_subset(data, category):
    new_data = OrderedDict()
    for k, v in data.items():
        if k[0] == category:
            new_data[k[1:]] = v
    return new_data

def _append_many_to_many_inplace(parent: list, child: list):
    """Append each element of a child list to a parent list."""
    if len(child) > 0:
        for i in range(len(parent)):
            parent[i].append(child[i])

def _concat_many_to_many_inplace(parent: list, child: list):
    """Concatenate each element of a child list to a parent list."""
    if len(child) > 0:
        for i in range(len(parent)):
            parent[i] += child[i]

# @dataclass(frozen=True)
class HierarchicalDataset:
    """A class for representing hierarchical datasets.

    The core underlying data structure for the HierarchicalDataset class is comprised of:
        1. An ordered dictionary with the following structure:
            {
                (attribute1_value, ..., attributeN_value): (replicate_1, ..., replicate_N1),  # This is a unique combination of attribute values
                (attribute1_value, ..., attributeN_value): (replicate_1, ..., replicate_N2),  # This is another unique combination of attribute values
                ...
            }
        where each replicate is an ordered dictionary with the following structure:
            {
                response1_name: response1_value,
                response2_name: response2_value,
                ...
            }
            NOTE: Not every replicate has to have the same number of responses.
        2. Other "metadata" (e.g., attribute names, response names, data types, fill values, etc.)
    
    From these, we can uniquely express the data in every other supported form (e.g., tabular, hierarchical dict, hierarchical arrays)
    """
    data: OrderedDict
    attribute_names: list[str]
    response_names: list[str]
    attribute_fill_value: Optional[list] = None
    response_fill_value: Optional[list] = None
    attribute_dtypes: Optional[list[type]] = None
    response_dtypes: Optional[list[type]] = None
    _user_specified_attribute_categories: Optional[list[bool]] = None
    share_attribute_categories_to_depth: Optional[int] = None
    # equal_ndim: Optional[bool] = None,

    def __init__(
        self,
        data: OrderedDict,
        attribute_names: list[str],
        response_names: list[str],
        attribute_fill_values: Optional[list] = None,
        response_fill_values: Optional[list] = None,
        attribute_dtypes: Optional[list[type]] = None,
        response_dtypes: Optional[list[type]] = None,
        fixed_attribute_categories: Optional[list[bool]] = None,
        share_attribute_categories_to_depth: Optional[int] = None,
    ):
        """Initialize the hierarchical dataset."""
        self.data = data
        self.attribute_names = attribute_names
        self.response_names = response_names

        # Set default dtypes and fill values
        attribute_dtypes, attribute_fill_values = _infer_dtypes_and_fill_values_from_examples(list(data[0].keys())[0], attribute_dtypes, attribute_fill_values)
        self.attribute_dtypes = attribute_dtypes + (int,)  # Add int for the replicates
        self.attribute_fill_values = attribute_fill_values + (-1,)
        self.response_fill_values = tuple(0.0 for _ in response_names) if response_fill_values is None else response_fill_values
        self.response_dtypes = tuple(float for _ in response_names) if response_dtypes is None else response_dtypes

        if fixed_attribute_categories is None:
            fixed_attribute_categories = [None]*len(attribute_names)
        if share_attribute_categories_to_depth is None:
            share_attribute_categories_to_depth = 0
        self.attribute_categories = self._get_attribute_categories(fixed_attribute_categories, share_attribute_categories_to_depth)
        self.max_replicates = [self._get_max_replicates(d) for d in data]
    
    def _get_attribute_categories(
        self,
        fixed_attribute_categories: Optional[list[bool]] = None,
        share_attribute_categories_to_depth: Optional[int] = None
    ):
        """Return the attribute categories for all datasets.

        The output is a list (1 entry per dataset) of lists (1 entry per attribute) of 
        lists (1 entry per category).
        """
        categories = []
        for i_data, data in enumerate(self.data):  # Iterate through each dataset
            categories.append([])
            for i_attr in range(len(self.attribute_names)):  # Iterate through each attribute
                if fixed_attribute_categories[i_attr] is not None:
                    # Here, we put in the user-specified categories
                    categories[i_data].append(list(categories[i_attr]))
                elif i_attr < share_attribute_categories_to_depth:
                    # Here, we put in the shared categories (across all datasets)
                    categories[i_data].append(list(np.unique(np.hstack([[k[i_attr] for k in data_.keys()] for data_ in self.data]))))
                else:
                    # Here, we put in the categories only for the current dataset
                    categories[i_data].append(list(np.unique([k[i_attr] for k in data])))
        return categories
    
    @staticmethod
    def _get_max_replicates(data):
        """Return the maximum number of replicates for a dataset."""
        return max(list(map(len, data.values())))

    # def __call__(self):
    #     return self.to_hierarchical_dict()
    
    # def __repr__(self):
    #     return f"HierarchicalDataset({self.data})"

    @classmethod
    def from_dataframe(
        cls,
        data: list[pd.DataFrame],
        attribute_names: list[str],
        response_names: list[str],
        **kwargs
    ):
        # TODO: We need to know what to do with duplicates
        """Construct a hierarchical dataset from a tabular, long-format pandas dataframe.

        For example, consider the following dataframe:
        A    B    R
        A_1  B_1  1.0
        A_1  B_2  1.5
        A_2  B_1  2.0
        A_2  B_1  2.5

        With labels = ['A', 'B'], the corresponding hierarchical dataset would look like this:
        {
            'A_1': {
                'B_1': [1.0],
                'B_2': [1.5]
            },
            'A_2': {
                'B_1': [2.0, 2.5]
            }
        }
        
        :param data: A tabular dataset in "long" format.
        :type data: pd.DataFrame
        :param labels: A list of labels.
        :type labels: list[str]
        :return: A nested dataset.
        :rtype: OrderedDict
        """
        data_reformatted = []
        for d in data:  # Iterate through each dataset
            d = d.reset_index()
            grouped = d.groupby(attribute_names)

            # Here, we make a dictionary where each key marks a different "treatment group"
            # (i.e., a unique combination of attributes). The data are encoded in the lowest
            # level. Each node in the lowest level is a dictionary whose keys are the data
            # response names, and whose values are arrays storing the data values (each element
            # is a replicate).
            hier_repr_implicit_rep = OrderedDict(grouped.apply(lambda x: OrderedDict({v: x[v].values for v in response_names}), include_groups=False).to_dict())
            
            # If there is only one attribute, pandas groupby will set the key to be the 
            # level for that attribute. But we want the the key to be a *tuple* containing 
            # the level for that attribute as its onnly element. This makes the representation
            # consistent with the multi-attribute case. Hence, the following:
            if len(attribute_names) == 1:
                hier_repr_implicit_rep = OrderedDict({(k,): v for k, v in hier_repr_implicit_rep.items()})
            
            # Here, we make a similar dictionary, except that the lowest level is a tuple of 
            # dictionaries, where each dictionary represents a single replicate. This is the
            # desired underlying representation of the data.
            hier_repr_explicit_rep = OrderedDict()
            for group_attributes, group_responses_all_implicit_rep in hier_repr_implicit_rep.items():  
                # NOTE:
                #   - group attributes = (attr1, attr2, ...)
                #   - group_responses_all_implicit_rep = {response1: [rep1, rep2, ...], response2: [rep1, rep2, ...], ...}
                group_responses_all_explicit_rep = ()
                for group_responses_single_rep in zip(*group_responses_all_implicit_rep.values()):
                    # NOTE:
                    #   - group_responses_single_rep = (rep1_from_response1, rep1_from_response2, ...)
                    tmp = OrderedDict()
                    for k, v in zip(response_names, group_responses_single_rep):
                        # NOTE:
                        #   - k = response_name
                        #   - v = replicate for response_name
                        tmp.update({k: v})
                    group_responses_all_explicit_rep += (tmp,)
                hier_repr_explicit_rep[group_attributes] = group_responses_all_explicit_rep
            
            # TODO: Remove NaNs (and other values based on a null_value argument)
            data_reformatted.append(hier_repr_explicit_rep)
        return cls(data_reformatted, attribute_names, response_names, **kwargs)
        # data_hierarchical = cls._nest_data_recursive(OrderedDict(), data, attribute_names)
        # return cls(data_hierarchical, attribute_names, response_names, **kwargs)

    @classmethod
    def from_tabular_arrays(
        cls, 
        attributes: list[list[np.ndarray]],  # TODO: actually, np.ndarray should be a list or 1d array
        responses: list[list[np.ndarray]],
        attribute_names: list[str],
        response_names: list[str],
        **kwargs
    ):
        """Initialize the hierarchical dataset from lists of arrays (in long format)."""
        # TODO: Add docstring
        # TODO: Add tests
        dataframes = []
        for attr, val in zip(attributes, responses):  # Iterate through each dataset
            tmp = {k: v for k, v in zip(attribute_names, attr)}
            tmp.update({k: v for k, v in zip(response_names, val)})
            dataframes.append(pd.DataFrame(tmp))
        return cls.from_dataframe(dataframes, attribute_names, response_names, **kwargs)

    @classmethod
    def from_hierarchical_dict(
        cls, 
        data: list[OrderedDict],
        attribute_names: list[str],
        response_names: list[str],
        **kwargs
    ):
        """Initialize the hierarchical dataset from an ordered dictionary."""
        # TODO: Add test
        def _extract_key(x):
            if isinstance(x, jtu.DictKey):
                return x.key
            elif isinstance(x, jtu.SequenceKey):
                return x.idx
            else:
                # raise ValueError(f"Key type must be one of {jtu.DictKey} or {jtu.SequenceKey}, but got {type(x)}.")
                return x
        def _dict_to_ordered_dict(x):
            return OrderedDict({k: v for k, v in x.items()})
        data_reformatted = []
        for d in data:  # Iterate through datasets
            data_reformatted.append(OrderedDict())
            keypath_leaves, _ = jtu.tree_flatten_with_path(d, is_leaf=lambda x: not isinstance(x, dict))  # stops traversing the pytree when it encounters the replicates tuple
            for keypath, leaf in keypath_leaves:  # Iterate through each "treatment group"
                attributes = tuple(map(_extract_key, keypath))  # Get attributes as a tuple
                responses = tuple(map(_dict_to_ordered_dict, leaf))  # get responses as a tuple of replicates, where each replicate's data is stored in a dictionary
                data_reformatted[-1][attributes] = responses  # Add this treatment group's data to the dataset under construction
        return cls(data_reformatted, attribute_names, response_names, **kwargs)
    
    @classmethod
    def from_hierarchical_arrays(
        cls, 
        data: list[np.ndarray], 
        attribute_names: list[str],
        response_names: list[str],
        masks: list[np.ndarray] = None,
        **kwargs
    ):
        """Initialize the hierarchical dataset from lists of arrays, labels, and masks."""
        if masks is None:
            masks = [None]*len(data)
        masked_arrays = [np.ma.array(d, mask=m) for d, m in zip(data, masks)]
        return cls.from_hierarchical_masked_arrays(masked_arrays, attribute_names)
    
    @classmethod
    def from_hierarchical_masked_arrays(
        cls, 
        data: list[np.ma.MaskedArray], 
        labels: list[str]
    ):
        """Initialize the hierarchical dataset from lists of masked arrays and labels."""
        pass

    def to_tabular(self):
        # TODO: Implement this
        # TODO: Add test
        # Plan: 
        # 1. Convert to tabular array format.
        # 2. Convert to dataframe.
        pass

    def to_hierarchical_dict(self):
        # TODO: Implement this (jtu.tree_map_with_path?)
        # TODO: Add test
        pass

    def to_hierarchical_masked_arrays(self):
        """Return the hierarchical dataset as collections of masked arrays."""
        # TODO: Add test
        n_d = len(self.data)
        n_a = len(self.attribute_names) + 1
        n_r = len(self.response_names)
        attribute_arrays = [[] for _ in range(n_d)]
        response_arrays = [{} for _ in range(n_d)]
        for i, d in enumerate(self.data):
            attr_main_lists, attr_mask_lists, resp_main_lists, resp_mask_lists = \
                self._get_hierarchical_lists_recursive(
                    i,
                    [[] for _ in range(n_a)],
                    [[] for _ in range(n_a)],
                    [[] for _ in range(n_r)],
                    [[] for _ in range(n_r)],
                    d, 
                    0, 
                    len(self.attribute_names)
                )
            for j, main, mask in zip(range(n_a), attr_main_lists, attr_mask_lists):
                attribute_arrays[i].append(
                    np.ma.array(main, mask=mask, fill_value=self.attribute_fill_values[j], dtype=self.attribute_dtypes[j])
                )
            for j, main, mask in zip(range(n_r), resp_main_lists, resp_mask_lists):
                response_arrays[i][self.response_names[j]] = \
                    np.ma.array(main, mask=mask, fill_value=self.response_fill_values[j], dtype=self.response_dtypes[j])
        return attribute_arrays, response_arrays
    

    def _get_hierarchical_lists_recursive(
        self,
        dataset_index,
        main_attribute, 
        mask_attribute, 
        main_response, 
        mask_response, 
        data, 
        depth, 
        max_depth
    ):
        """Return the hierarchical lists for a dataset."""
        # TODO: Write docstrings and inline comments
        if depth < max_depth:
            categories = self.attribute_categories[dataset_index][depth]
            for counter in range(len(categories)):
                more_main_attribute = [[] for _ in range(len(main_attribute))]
                more_mask_attribute = [[] for _ in range(len(mask_attribute))]
                more_main_response = [[] for _ in range(len(main_response))]
                more_mask_response = [[] for _ in range(len(mask_response))]
                category, mask_, counter = _get_category_mask_counter(data, categories, counter, self.attribute_fill_values[depth])
                more_main_attribute[0].append(category)
                more_mask_attribute[0].append(mask_)
                data_subset = _get_data_subset(data, category)
                (
                    even_more_main_attribute, 
                    even_more_mask_attribute, 
                    even_more_main_response, 
                    even_more_mask_response
                ) = \
                    self._get_hierarchical_lists_recursive(
                        dataset_index,
                        [[] for _ in range(len(main_attribute) - 1)], 
                        [[] for _ in range(len(main_attribute) - 1)], 
                        [[] for _ in range(len(main_response))],
                        [[] for _ in range(len(main_response))],
                        data_subset, 
                        depth + 1, 
                        max_depth
                    )
                _append_many_to_many_inplace(more_main_attribute[1:], even_more_main_attribute)
                _append_many_to_many_inplace(more_mask_attribute[1:], even_more_mask_attribute)
                _append_many_to_many_inplace(more_main_response, even_more_main_response)
                _append_many_to_many_inplace(more_mask_response, even_more_mask_response)
                _concat_many_to_many_inplace(main_attribute, more_main_attribute)
                _concat_many_to_many_inplace(mask_attribute, more_mask_attribute)
                _concat_many_to_many_inplace(main_response, more_main_response)
                _concat_many_to_many_inplace(mask_response, more_mask_response)
        elif depth == max_depth:
            max_rep = self.max_replicates[dataset_index]
            response_names = self.response_names
            more_main_response = [[] for _ in range(len(main_response))]
            more_mask_response = [[] for _ in range(len(mask_response))]
            for counter in range(max_rep):
                more_main_attribute = [[] for _ in range(len(main_attribute))]
                more_mask_attribute = [[] for _ in range(len(mask_attribute))]
                replicate, response, replicate_mask, response_mask, counter = _get_replicate_response_mask_counter(data, response_names, counter, self.attribute_fill_values[depth], self.response_fill_values)
                more_main_attribute[0].append(replicate)
                more_mask_attribute[0].append(replicate_mask)
                _concat_many_to_many_inplace(main_attribute, more_main_attribute)
                _concat_many_to_many_inplace(mask_attribute, more_mask_attribute)
                _append_many_to_many_inplace(more_main_response, response)
                _append_many_to_many_inplace(more_mask_response, response_mask)
            _concat_many_to_many_inplace(main_response, more_main_response)
            _concat_many_to_many_inplace(mask_response, more_mask_response)
        return main_attribute, mask_attribute, main_response, mask_response
