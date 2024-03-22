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

__all__ = ["HyperParameter", "get_hyperparameter_and_path", "get_hyperparameter_and_update_model"]

import equinox as eqx
import numpyro
import numpyro.distributions as dist
from typing import NamedTuple, Optional
from jaxtyping import PyTree

class HyperParameter(NamedTuple):
    """Container for storing information about a hyperparameter in a model."""
    path: callable
    stochastic_fn: Optional[dist.Distribution] = None

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
    if param.stochastic_fn is not None:
        return numpyro.sample(name, param.stochastic_fn, **kwargs), param.path
    else:
        return numpyro.param(name, **kwargs), param.path

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