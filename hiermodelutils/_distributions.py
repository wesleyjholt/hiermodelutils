__all__ = ['LogNormalMeanStd', 
           'LogNormalMedianScatter',
           'GammaMeanStd', 
           'BetaMeanStd', 
           'RealLineToUnitTransform',
           'UniformToBetaTransform',
        #    'AffineBeta',
        #    'AffineBetaMeanStd',
           'softplus']

import jax
from jax import lax, grad
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import Transform, AffineTransform

def softplus(x, k=1.0):
    """Softplus function with a "hardness" factor k. The function is defined as log(1 + exp(k*x))/k.
    
    See https://www.johndcook.com/blog/2023/08/06/swish-mish-and-serf/.
    """
    return jnp.logaddexp(0, k*x)/k


class LogNormalMeanStd(dist.LogNormal):
    arg_constraints = {"mean_x": dist.constraints.positive, "std_x": dist.constraints.positive}
    support = dist.constraints.positive

    def __init__(self, mean, std, **kwargs):
        self.mean_x = mean
        self.std_x = std
        params = self.get_pyro_params_from_mean_std(mean, std)
        super().__init__(**params, **kwargs)
    
    @staticmethod
    def get_pyro_params_from_mean_std(mean, std):
        mean2 = mean**2
        var = std**2
        mean_logx = jnp.log(mean2/jnp.sqrt(mean2 + var))
        std_logx = jnp.sqrt(jnp.log(1 + var/mean2))
        return dict(
            loc=mean_logx,
            scale=std_logx
        )

class LogNormalMedianScatter(dist.LogNormal):
    arg_constraints = {"median_x": dist.constraints.positive, "scatter_x": dist.constraints.positive}
    support = dist.constraints.positive

    def __init__(self, median, scatter, **kwargs):
        self.median_x = median
        self.scatter_x = scatter
        loc = jnp.log(median)
        scale = jnp.log(scatter)
        super().__init__(loc, scale, **kwargs)

class RealLineToUnitTransform(Transform):
    domain = dist.constraints.real
    codomain = dist.constraints.unit_interval
    event_dim = 0

    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        self.loc = loc
        self.scale = scale

    def __call__(self, x):
        return dist.Normal(self.loc, self.scale).cdf(x)  # R -> [0, 1]

    def inv(self, y):
        return dist.Normal(self.loc, self.scale).icdf(y)  # [0, 1] -> R

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return dist.Normal(self.loc, self.scale).log_prob(x)
    
    def tree_flatten(self):
        return (self.loc, self.scale), (("loc", "scale"), dict())
    
    def __eq__(self, other):
        if not isinstance(other, RealLineToUnitTransform):
            return False
        return self.domain == other.domain

class UniformToBetaTransform(Transform):
    domain = dist.constraints.unit_interval
    codomain = dist.constraints.unit_interval
    event_dim = 0

    def __init__(self, concentration1: float = 1.0, concentration0: float = 1.0):
        self.concentration1 = concentration1
        self.concentration0 = concentration0

    def __call__(self, x):
        return self.create_beta_dist().icdf(x)

    def inv(self, y):
        return self.create_beta_dist().cdf(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.log(jnp.abs(grad(self.create_beta_dist().icdf)(x)))
    
    def tree_flatten(self):
        return (self.concentration1, self.concentration0), (("concentration1", "concentration0"), dict())

    def create_beta_dist(self):
        return dist.Beta(concentration1=self.concentration1, concentration0=self.concentration0)

    def __eq__(self, other):
        if not isinstance(other, UniformToBetaTransform):
            return False
        return self.domain == other.domain

class GammaMeanStd(dist.Gamma):
    def __init__(self, mean, std, **kwargs):
        self.mean_x = mean
        self.std_x = std
        params = self.get_pyro_params_from_mean_std(mean, std)
        super().__init__(**params, **kwargs)
    
    @staticmethod
    def get_pyro_params_from_mean_std(mean, std):
        var = std**2
        alpha = mean**2/var
        beta = mean/var
        return dict(
            concentration=alpha,
            rate=beta
        )

class BetaMeanStd(dist.Beta):

    def __init__(self, mean, std, **kwargs):
        self.mean_x = mean
        self.std_x = std
        params = self.get_pyro_params_from_mean_std(mean, std)
        super().__init__(**params, **kwargs)
    
    @staticmethod
    def get_pyro_params_from_mean_std(mean, std):
        eps = 0.0
        safe_mean = jnp.clip(mean, eps, 1 - eps)
        safe_std = jnp.clip(std, eps, 0.5 - eps)
        var = std**2
        # jax.debug.print('mean: {x}', x=mean)
        # jax.debug.print('var: {x}', x=var)
        alpha = mean*(mean*(1-mean)/var - 1)
        beta = alpha*(1/mean - 1)
        jax.debug.print('alpha: {x}', x=alpha)
        jax.debug.print('beta: {x}', x=beta)
        return dict(
            concentration1=alpha,
            concentration0=beta
        )

# class AffineBeta(dist.TransformedDistribution):
#     def __init__(self, concentration1, concentration0, loc, scale, validate_args=None, eps=1e-10):
#         base_dist = dist.Beta(concentration1, concentration0, validate_args=validate_args)
#         transforms = [AffineTransform(loc=loc, scale=scale)]
#         super().__init__(base_dist, transforms, validate_args=validate_args)
#         self.eps = eps

#     @staticmethod
#     def infer_shapes(concentration1, concentration0, loc, scale):
#         batch_shape = lax.broadcast_shapes(jnp.shape(concentration1), jnp.shape(concentration0),
#                                             jnp.shape(loc), jnp.shape(scale))
#         event_shape = ()
#         return batch_shape, event_shape

#     def sample(self, key, sample_shape=()):
#         x = self.base_dist.sample(key, sample_shape)
#         for transform in self.transforms:
#             x = transform(x)
#         eps = self.eps * self.scale
#         x = jnp.clip(x, self.low + eps, self.high - eps)
#         return x

#     def rsample(self, key, sample_shape=()):
#         x = self.base_dist.rsample(key, sample_shape)
#         for transform in self.transforms:
#             x = transform(x)
#         eps = self.eps * self.scale
#         x = jnp.clip(x, self.low + eps, self.high - eps)
#         return x

#     @property
#     def concentration1(self):
#         return self.base_dist.concentration1

#     @property
#     def concentration0(self):
#         return self.base_dist.concentration0

#     @property
#     def sample_size(self):
#         return self.concentration1 + self.concentration0

#     @property
#     def loc(self):
#         return jnp.asarray(self.transforms[0].loc)

#     @property
#     def scale(self):
#         return jnp.asarray(self.transforms[0].scale)

#     @property
#     def low(self):
#         return self.loc

#     @property
#     def high(self):
#         return self.loc + self.scale

#     @property
#     def mean(self):
#         return self.loc + self.scale * self.base_dist.mean

#     @property
#     def variance(self):
#         return self.scale ** 2 * self.base_dist.variance

# class AffineBetaMeanStd(AffineBeta):
#     def __init__(self, mean, std, loc, scale, validate_args=None, eps=1e-6):
#         # mean and std are of the original distribution (i.e., the one with support [0, 1])
#         # The mean and std of the final distribution are loc*mean and scale*std, respectively.
#         p = BetaMeanStd.get_pyro_params_from_mean_std(mean, std)
#         # jax.debug.print('conc1: {x}', x=p['concentration1'])
#         super().__init__(p['concentration1'], p['concentration0'], loc, scale, validate_args=validate_args, eps=eps)