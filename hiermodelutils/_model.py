"""Utilities for defining and working with hierarchical models.

This is the intended pattern to follow:
- A model is a collection of smaller models, each of which is implemented as an equinox class.
- How do you decide how "finely" to break up your overall model? Well, create separate model if
    1. It makes sense conceptually.
    2. You would like to have it isolated for postprocessing/plotting purposes.
    3. It is a stochastic model.
- For stochastic models, use the stochasticmodel decorator, which
    1. Defines the __call__ method as sampling the specified distribution (up until now, you have only implemented the get_distribution method)
    2. Adds optional kwarg `samples` to the __call__ method which allows you to override the numpyro.sample statement and instead return the e.g. posterior samples (as read off the `samples` dict)
- The parameters for the model show up as class attributes in each equinox model/pytree.
- Create a priors/default dict, where each entry corresponds to a different parameter in the model and contains:
    1. The name of the parameter
    2. The prior/default distribution of the parameter
- To set a new value for a certain parameter (e.g., after a sample step of MCMC) then one must use eqx.tree_at.
"""

__all__ = ["Parameter", 
           "stochasticmodel", 
           "draw_sample",
           "draw_sample_and_get_path",
           "draw_sample_and_update_model"]

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpyro
import numpyro.distributions as dist
from typing import NamedTuple, Optional
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray, PyTree

def stochasticmodel(name: str):
    """Decorator for stochastic models (e.g., in the likelihood).

    The class to be decorated should have a get_distribution method which outputs a numpyro distribution.
    
    :param name: The name for the output of the stochastic model.
    :type name: str
    """

    def stochasticmodel_constructor(cls):
        """Constructs a stochastic model.
        
        :param cls: The class to be decorated. Should have a get_distribution method which outputs a numpyro distribution.
        :type cls: class
        """
    
        class StochasticModel(cls):
            """A generic class for stochastic models (in the likelihood)."""
            name: str
            # samples: Optional[dict[str, Array]]
            def __init__(
                self, 
                *args, 
                # samples: Optional[dict] = None, 
                **kwargs
            ):
                super().__init__(*args, **kwargs)
                self.name = name
                # self.samples = samples

            def __call__(
                self, 
                *args, 
                obs: Optional[ArrayLike] = None, 
                suffix: Optional[str] = None,
                **kwargs
            ):
                # TODO: Remove samples kwarg from __call__ method. This same functionality can be achieved with numpyro's Predictive class.
                # if self.samples is not None:
                #     full_name = self.name if suffix is None else f'{self.name}_{suffix}'
                #     if full_name in self.samples:
                #         return self.samples[full_name]
                name = self.name if suffix is None else f"{self.name}_{suffix}"
                distribution = super().get_distribution(*args, **kwargs)
                return numpyro.sample(name, distribution, obs=obs)

        return type(cls.__name__, (StochasticModel,), {})

    return stochasticmodel_constructor

class Parameter(NamedTuple):
    """Container for storing information about a parameter in a model."""
    path: callable
    distribution: dist.Distribution

def draw_sample(
    name: str, 
    paths_and_distributions: dict[str, Parameter]
) -> Array:
    """Draws a sample."""
    return numpyro.sample(name, paths_and_distributions[name].distribution)

def draw_sample_and_get_path(
    name: str, 
    paths_and_distributions: dict[str, Parameter]
) -> tuple[Array, callable]:
    """Draws a sample and gives the "location" where it belongs within the model."""
    val = draw_sample(name, paths_and_distributions)
    path = paths_and_distributions[name].path
    return val, path

def draw_sample_and_update_model(
    names: str, 
    model: PyTree, 
    paths_and_distributions: dict[str, Parameter]
) -> PyTree:
    """Draws a sample and puts it into the model."""
    val, path = draw_sample_and_get_path(names, paths_and_distributions)
    model = eqx.tree_at(path, model, val)
    return model

# TODO: Delete this function. The same functionality can be achieved with numpyro's Predictive class.
# def set_all_model_parameters_from_samples(
#     model: PyTree, 
#     *, 
#     paths_and_distributions: dict,
#     samples: dict
# ):
#     """Takes samples and puts them into the models.
    
#     For each parameter in paths_and_distributions, we first check if there are corresponding samples. 
#     If there are, then we put them into `model`. If not, then we 
#     1. take the (empirical) median of the prior
#     2. create an array of the same size as the samples and whose elements are all equal to the median
#     3. put this new array into `model`
#     The resulting `model` pytree can then be evaluated using the parameter samples.
#     """
#     sample_sizes = [value.shape[0] for value in samples.values()]
#     if all([size == sample_sizes[0] for size in sample_sizes]):
#         sample_size = sample_sizes[0]
#     else:
#         raise ValueError("All samples must have the same size.")
#     for name, value in paths_and_distributions.items():
#         if name in samples:
#             model = eqx.tree_at(value.path, model, samples[name])
#         else:
#             print('Warning: using prior for', name, 'because no sample was found.')
#             median = jnp.median(value.distribution.sample(jr.PRNGKey(0), (20,)), axis=0)  # This doesn't need to be accurate -- really we just need a value that won't throw an error downstream.
#             model = eqx.tree_at(value.path, model, jnp.full((sample_size, *jnp.asarray(median).shape), median))
#     return model