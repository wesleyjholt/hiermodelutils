import jax.numpy as jnp
import numpy as np
from typing import Optional
from jaxtyping import Shaped, Bool, ArrayLike, Array, jaxtyped
# from beartype import beartype

# TODO: We need ignore_ndim for the following scenario: We have array shape (3, 2) and mask shape (1, 1, 2), and we want the output shape to be (3, 1, 1, 2)
# SOLUTION: Just insert enough axes after the ignored ones until the non-ignored axes match the mask shape.

def _get_full_and_extra_shape(
    array_shape: tuple[int, ...],
    mask_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """Get the full shape of the array and mask."""
    if len(array_shape) < len(mask_shape):
        return mask_shape, ()
    elif len(array_shape) == len(mask_shape):
        return mask_shape, ()
    if len(array_shape) > len(mask_shape):
        extra_ndim = len(array_shape) - len(mask_shape)
        extra_shape = array_shape[:extra_ndim]
        return (*extra_shape, *mask_shape), extra_shape

def _insert_dims_into_array(
    array: Shaped[ArrayLike, "..."],
    mask_shape: tuple[int, ...],
    ignore_ndim: int
) -> Shaped[Array, "..."]:
    if (ignore_ndim < 0) or (ignore_ndim > array.ndim):
        raise ValueError(f"ignore_ndim must be between 0 and {array.ndim}.")
    """Insert dimensions into the array to match the mask shape."""
    if ignore_ndim > 0:
        non_ignored_shape = array.shape[ignore_ndim:]
        if len(non_ignored_shape) <= len(mask_shape):
            n_insert = len(mask_shape) - len(non_ignored_shape)
            array = jnp.expand_dims(array, axis=tuple(range(ignore_ndim, ignore_ndim + n_insert)))
    return array

def flatten_and_condense(
    array: Shaped[ArrayLike, "..."], 
    mask: Bool[np.ndarray, "..."], 
    ignore_ndim: Optional[int] = 0
) -> Shaped[Array, "..."]:
    """Flattens an array (up to certain dim) and removes all values where mask=True.
    
    This is especially useful when the hierarchical arrays are inputs into an expensive
    model. If you naively pass the arrays into the (vectorized) expensive model, then the 
    model will be run for all elements in the array, even the meaningless masked values 
    (which are only there to preserve the array shape). By flattening and condensing the 
    input arrays, you avoid wasting computation on masked values.

    Follows these rules: 
    - Let array have shape (*A, *B) and mask have shape (*C, *D).
    - Define the equivalence relation ~ as:
        - X ~ Y if the len(X) == len(Y)
    - The shapes are compatible if:
        - There B ~ C, where ~ 
    - Then the full shape of the array is
    
    Note: Fills `array` to match the shape of `mask`, if necessary.

    :param array: Array to flatten and condense.
    :param mask: Mask to apply to the array.
    :return: Condensed array.
    """
    array = _insert_dims_into_array(array, mask.shape, ignore_ndim)
    full_shape, extra_shape = _get_full_and_extra_shape(array.shape, mask.shape)
    array = jnp.broadcast_to(array, full_shape)
    mask = np.broadcast_to(mask, full_shape)
    return array[~np.full(full_shape, mask)].reshape(*extra_shape, -1)

def unflatten_and_expand(
    array: Shaped[ArrayLike, "..."], 
    mask: Bool[np.ndarray, "..."], 
) -> Shaped[Array, "..."]:
    """Expands and unflattens an array.
    
    Restores `array` back to its original shape (i.e., the `mask` shape). This is
    essentially the inverse of flatten_and_condense (except the last dimensions will
    have sizes equal to the mask shape, not necessarily equal to the original array
    shape).

    :param array: Array to expand and unflatten.
    :param mask: Mask to apply to the array.
    :return: Expanded array.
    """
    # Create a matrix M_ij, where column j is a one-hot vector encoding the index of the
    # non-condensed (flat) array to which the jth element of array belongs.
    M = np.diag(~mask.ravel())
    M = M[np.sum(M, axis=0) > 0]

    # Get the shape of the output array
    if array.ndim == 1:
        new_shape = mask.shape
    elif array.ndim > 1:
        new_shape = (*array.shape[:array.ndim - 1], *mask.shape)
    
    return jnp.einsum('ij,...i->...j', M, array).reshape(new_shape)