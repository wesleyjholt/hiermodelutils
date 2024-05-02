"""Utilities for defining and working with hierarchical models.

This is the intended pattern to follow:
- A full hierarchical model is made up of a collection of (potentially nested) callable pytrees (e.g., equinox models).
- The hyperparameters (i.e., things that don't depend on any upstream parameters) are stored as attributes in the
    equinox models.
- The parameters (i.e., things that do depend on upstream values) are everything else that is included in the 
    equinox models' methods.
- If there is any parameter that will be of interest for analysis, then one should make an equinox model that directly
    returns that parameter.
"""

__all__ = ["HyperParameter", "get_path", "get_hyperparameter_and_path", "get_hyperparameter_and_update_model", "default_names"]

import equinox as eqx
import numpyro
import numpyro.distributions as dist
from typing import NamedTuple, Optional, Union
from jaxtyping import PyTree, Float, Array

class HyperParameter(NamedTuple):
    """Container for storing information about a hyperparameter in a model."""
    path: callable
    is_stochastic: bool
    fn: Optional[dist.Distribution] = None
    init_value: Optional[Union[Array, callable]] = None

def get_path(
    name: str,
    all_hyperparams: dict[str, HyperParameter]
):
    """Retrieves a hyperparameter's path.
    
    :param path: The path to retrieve.
    :type path: str
    """
    return all_hyperparams[name].path

def get_hyperparameter_and_path(
    name: str,
    all_hyperparams: dict[str, HyperParameter],
    **kwargs
):
    """Retrieves/samples a hyperparameter from a dictionary of hyperparameters.
    
    :param name: The name of the hyperparameter.
    :param type: str
    :param all_hyperparams: The dictionary of all hyperparameters.
    :param type: dict[str, HyperParameter]
    """
    param = all_hyperparams[name]
    if 'suffix' in kwargs:
        suffix = kwargs.pop('suffix')
        name = f"{name}_{suffix}"
    if param.is_stochastic:
        return numpyro.sample(name, param.fn, **kwargs), param.path
    else:
        return numpyro.param(name, param.init_value, **kwargs), param.path

def get_hyperparameter_and_update_model(
    name: str,
    model: PyTree,
    all_hyperparams: dict[str, HyperParameter],
    **kwargs
):
    """Retrieves/samples a hyperparameter from a dictionary of hyperparameters and stores it in the model.
    
    :param name: The name of the hyperparameter.
    :param type: str
    :param model: The model to update.
    :param type: PyTree
    :param all_hyperparams: The dictionary of all hyperparameters.
    :param type: dict[str, HyperParameter]
    """
    val, path = get_hyperparameter_and_path(name, all_hyperparams, **kwargs)
    model = eqx.tree_at(path, model, val)
    return model

def default_names(names: str | tuple[str]):
    """Decorator for giving a model a default name set.
    
    :param names: The default names for the outputs of the stochastic model.
    :type name: str
    """

    def model_with_default_names(cls):
        """Constructs a stochastic model with default names.
        
        :param cls: The class to be decorated.
        :type cls: class
        """
    
        class ModelWithDefaultNames(cls):
            """A stochastic model with default names for its outputs."""
            names: tuple[str]

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.names = names
            
            def __call__(self, *args, names: Optional[str] = None, **kwargs):
                if names is None:
                    names = self.names
                return super().__call__(*args, names=names, **kwargs)

        return type(cls.__name__, (ModelWithDefaultNames,), {})

    return model_with_default_names