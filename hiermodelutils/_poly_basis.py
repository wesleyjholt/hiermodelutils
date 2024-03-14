__all__ = ['compute_polynomial_basis_exponents']

import numpy as np

# TODO: Fix bug - degrees has one extra term.

def compute_polynomial_basis_exponents(num_dim, total_degree, degrees=None):
    if degrees is None:
        degrees = [total_degree for _ in range(num_dim)]
    _, total_degree, _ = _compute_polynomial_basis_exponents(num_dim, total_degree, degrees)
    return total_degree

def _compute_polynomial_basis_exponents(num_dim, total_degree, degrees):
    """Compute the basis terms.

    The following is taken from Stokhos.

    The approach here for ordering the terms is inductive on the total
    order p.  We get the terms of total order p from the terms of total
    order p-1 by incrementing the orders of the first dimension by 1.
    We then increment the orders of the second dimension by 1 for all of the
    terms whose first dimension order is 0.  We then repeat for the third
    dimension whose first and second dimension orders are 0, and so on.
    How this is done is most easily illustrated by an example of dimension 3:

    Order  terms   cnt      Order  terms   cnt
      0    0 0 0              4    4 0 0  15 5 1
                                   3 1 0
      1    1 0 0  3 2 1            3 0 1
           0 1 0                   2 2 0
           0 0 1                   2 1 1
                                   2 0 2
      2    2 0 0  6 3 1            1 3 0
           1 1 0                   1 2 1
           1 0 1                   1 1 2
           0 2 0                   1 0 3
           0 1 1                   0 4 0
           0 0 2                   0 3 1
                                   0 2 2
      3    3 0 0  10 4 1           0 1 3
           2 1 0                   0 0 4
           2 0 1
           1 2 0
           1 1 1
           1 0 2
           0 3 0
           0 2 1
           0 1 2
           0 0 3
    """
    # Temporary array of terms grouped in terms of same order
    terms_order = [[] for i in range(total_degree + 1)]

    # Store number of terms up to each order
    num_terms = np.zeros(total_degree + 2, dtype='i')

    # Set order zero
    terms_order[0] = ([np.zeros(num_dim, dtype='i')])
    num_terms[0] = 1

    # The array cnt stores the number of terms we need to
    # increment for each dimension.
    cnt = np.zeros(num_dim, dtype='i')
    for j, degree in zip(range(num_dim), degrees):
        if degree >= 1:
            cnt[j] = 1

    cnt_next = np.zeros(num_dim, dtype='i')
    term = np.zeros(num_dim, dtype='i')

    # Number of basis functions
    num_basis = 1

    # Loop over orders
    for k in range(1, total_degree + 1):
        num_terms[k] = num_terms[k - 1]
        # Stores the inde of the term we are copying
        prev = 0
        # Loop over dimensions
        for j, degree in zip(range(num_dim), degrees):
            # Increment orders of cnt[j] terms for dimension j
            for i in range(cnt[j]):
                if terms_order[k - 1][prev + i][j] < degree:
                    term = terms_order[k - 1][prev + i].copy()
                    term[j] += 1
                    terms_order[k].append(term)
                    num_basis += 1
                    num_terms[k] += 1
                    for l in range(j + 1):
                        cnt_next[l] += 1
            if j < num_dim - 1:
                prev += cnt[j] - cnt[j + 1]
        cnt[:] = cnt_next
        cnt_next[:] = 0
    num_terms[total_degree + 1] = num_basis
    # Copy into final terms array
    terms = []
    for k in range(total_degree + 1):
        num_k = len(terms_order[k])
        for j in range(num_k):
            terms.append(terms_order[k][j])
    terms = np.array(terms, dtype='i')
    return num_basis, terms, num_terms