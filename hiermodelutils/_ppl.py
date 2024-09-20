__all__ = ['get_logdensities_and_transforms_from_numpyro', 'get_generative_model']

import jax.random as jr
from numpyro.infer.util import initialize_model
from typing import NamedTuple
import numpy as np

def get_logdensities_and_transforms_from_numpyro(
    model, 
    model_args=(), 
    model_kwargs=dict(), 
    is_conditional_model=False, 
    prior_postprocess_fn=None,
    prior_init_params=None
):
    key = jr.PRNGKey(np.random.randint(0, 2**32 - 1))
    if is_conditional_model:
        model_args_ = (prior_postprocess_fn(prior_init_params),) + model_args
    else:
        model_args_ = model_args
    
    (
        init_params, 
        potential_fn_gen, 
        postprocess_fn_gen, 
        model_trace
    ) = initialize_model(
        key, 
        model, 
        model_args=model_args_, 
        model_kwargs=model_kwargs, 
        dynamic_args=True
    )

    if is_conditional_model:
        logdensity_fn = lambda position: -potential_fn_gen(prior_postprocess_fn(position), *model_args, **model_kwargs)(position)
        postprocess_fn = lambda position: postprocess_fn_gen(prior_postprocess_fn(position), *model_args, **model_kwargs)(position)
    else:
        logdensity_fn = lambda position: -potential_fn_gen(*model_args, **model_kwargs)(position)
        postprocess_fn = lambda position: postprocess_fn_gen(*model_args, **model_kwargs)(position)
    
    class DistributionInfo(NamedTuple):
        init_params: dict
        logdensity_fn: callable
        postprocess_fn: callable
        model_trace: any
    
    return DistributionInfo(init_params, logdensity_fn, postprocess_fn, model_trace)

def get_generative_model(
    prior: callable,
    likelihood: callable,
    prior_args=(),
    prior_kwargs=dict(),
    likelihood_args=(),
    likelihood_kwargs=dict()
):
    """Get the log densities and transforms from a prior and likelihood function.
    
    To work properly, the prior and likelihood functions must follow the pattern:
    
    ```python
    def prior(*args, **kwargs) -> dict:
        # Sample parameters
        x = numpyro.sample(...)
        y = numpyro.sample(...)
        ...

        # Return all parameters in a dictionary
        return latents
    
    def likelihood(params, *args, **kwargs):
        # Retrieve prior samples of parameters
        x = params["x"]
        y = params["y"]
        ...

        # Do computations for the likelihood using numpyro statements (sample, plate, factor, etc.)
        # ... (stuff using x, y, ..., and args)

        # Return all parameters in a dictionary
        return observed_variables
    ```

    Parameters
    ----------
    prior : callable
        The prior function.
    likelihood : callable
        The likelihood function.
    prior_args : tuple, optional
    prior_kwargs : dict, optional
    likelihood_args : tuple, optional
    likelihood_kwargs : dict, optional
    
    Returns
    -------
    GenerativeModel
        A named tuple containing the prior, likelihood, and joint log densities and transforms.
    """
    prior_info = get_logdensities_and_transforms_from_numpyro(prior, prior_args, prior_kwargs)
    # likelihood_info = get_logdensities_and_transforms_from_numpyro(likelihood, likelihood_args, likelihood_kwargs, is_conditional_model=True, prior_postprocess_fn=prior_info.postprocess_fn, prior_init_params=prior_info.init_params.z)
    # prior_output = prior(*prior_args, **prior_kwargs)
    likelihood_info = get_logdensities_and_transforms_from_numpyro(likelihood, likelihood_args, likelihood_kwargs, is_conditional_model=True, prior_postprocess_fn=prior_info.postprocess_fn, prior_init_params=prior_info.init_params.z)
    def joint(*args, **kwargs):
        params = prior(*args[0], **args[1])
        return likelihood(params, *args[2], **args[3])
    joint_info = get_logdensities_and_transforms_from_numpyro(joint, (prior_args, prior_kwargs, likelihood_args, likelihood_kwargs))
    class GenerativeModel(NamedTuple):
        prior: object
        likelihood: object
        joint: object
    return GenerativeModel(prior_info, likelihood_info, joint_info)

# # TODO: Test this with variable transforms automatically created by numpyro.
# def get_logdensities_and_transforms_from_numpyro(
#     prior: callable, 
#     likelihood: callable,
#     prior_extra_args = (),
#     likelihood_extra_args = (),
#     *,
#     key
# ):
#     """Get the log densities and transforms from a prior and likelihood function."""
#     key_prior, key_likelihood = jr.split(key)

#     # Prior
#     prior_init_params, prior_potential_fn_gen, *_ = initialize_model(
#         key_prior,
#         prior,
#         model_args=prior_extra_args,
#         dynamic_args=True,
#     )

#     def prior_logdensity_fn(position):
#         """The prior log density function.
        
#         :param position: the position at which to evaluate the log density
#         """
#         return -prior_potential_fn_gen()(position)
    
#     # Likelihood
#     likelihood_init_params, likelihood_potential_fn_gen, *_ = initialize_model(
#         key_likelihood,
#         likelihood,
#         model_args=(prior_init_params.z,) + likelihood_extra_args,
#         dynamic_args=True,
#     )

#     def likelihood_logdensity_fn(position):
#         """The likelihood log density function.
        
#         :param position: the position at which to evaluate the log density
#         """
#         return -likelihood_potential_fn_gen(position)({})
    
#     class Problem(NamedTuple):
#         prior_logdensity_fn: callable
#         likelihood_logdensity_fn: callable
#         prior_init_params: dict
#         likelihood_init_params: dict
    
#     return Problem(
#         prior_logdensity_fn, 
#         likelihood_logdensity_fn,
#         prior_init_params,
#         likelihood_init_params
#     )
