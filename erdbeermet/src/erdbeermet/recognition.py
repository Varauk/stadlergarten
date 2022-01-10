# -*- coding: utf-8 -*-

from itertools import combinations, permutations
import numpy as np

from erdbeermet.tools.Tree import Tree, TreeNode
from numpy.core.fromnumeric import sort


__author__ = 'David Schaller'


def is_pseudometric(D, rtol=1e-05, atol=1e-08, print_info=False, V=None,
                    return_info=False):
    """Check whether a given distance matrix is a pseudometric.

    Parameters
    ----------
    D : 2-dimensional numpy array
        Distance matrix
    rtol : float, optional
        Relative tolerance for equality. The default is 1e-05.
    atol : float, optional
        Absolute tolerance for equality. The default is 1e-08.
    print_info : bool, optional
        If True, print first encountered violation of the triangle inequality
        if any.
    V : list, optional
        List of items (used for info output).
    return_info : bool, optional
        If True, return an info string as a second return value. The default
        is False.

    Return
    ------
    bool or tuple of bool and str
        True if D is a pseudometric and optionally an info string.
    """

    N = D.shape[0]

    # check whether all entries are non-negative
    if not np.all(np.logical_or(np.isclose(D, 0.0, rtol=rtol, atol=atol),
                                D > 0.0)):
        return False if not return_info else (False, 'negative distances')

    # check whether all diagonal entries are zero
    if np.any(np.diagonal(D)):
        return False if not return_info else (False, 'non-zero diagonal')

    # check whether the matrix is symmetric
    if not np.allclose(D, D.T, rtol=rtol, atol=atol):
        return False if not return_info else (False, 'not symmetric')

    # check the triangle inequality
    for i in range(N-1):
        for j in range(i+1, N):
            minimum = np.min(D[i, :] + D[:, j])
            if minimum < D[i, j] and not np.isclose(minimum, D[i, j],
                                                    rtol=rtol, atol=atol):
                if print_info or return_info:
                    argmin = np.argmin(D[i, :] + D[:, j])
                    if not V:
                        info = f'triangle inequality violation: D[{i},'\
                               f'{j}]={D[i,j]} > {minimum} over {argmin}'
                    else:
                        info = f'triangle inequality violation: D[v{V[i]},'\
                               f'v{V[j]}]={D[i,j]} > {minimum} over v{V[argmin]}'
                        if print_info:
                            print(info)
                return False if not return_info else (False, info)

    return True if not return_info else (True, 'passed')


def distance_sums_matrix(D, x, y, z, u):

    xy_zu = D[x,y] + D[z,u]
    xz_yu = D[x,z] + D[y,u]
    xu_yz = D[x,u] + D[y,z]

    return xy_zu, xz_yu, xu_yz


def restrict_matrix(D, indices):

    if min(indices) < 0 or max(indices) >= D.shape[0]:
        raise IndexError("List contains index that is out of range!")

    n = len(indices)
    D_new = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            D_new[i, j] = D[indices[i], indices[j]]

    return D_new


def _recognize4_parent_xy(D, x, y, z, u):

    left = D[x,y] * (D[x,y] + 2 * D[z,u] - D[x,z] - D[y,u] - D[x,u] - D[y,z])
    right = (D[x,z] - D[y,z]) * (D[y,u] - D[x,u])

    return np.isclose(left, right) or left < right

def _recognize4_xy_zu(D, x, y, z, u):

    return (_recognize4_parent_xy(D, x, y, z, u) or
            _recognize4_parent_xy(D, z, u, x, y))


def recognize4_new(D, x, y, z, u):

    if not is_pseudometric(restrict_matrix(D, [x, y, z, u])):
        return False

    dsums = distance_sums_matrix(D, x, y, z, u)

    if dsums[0] == max(dsums):
        return _recognize4_xy_zu(D, x, y, z, u)
    elif dsums[1] == max(dsums):
        return _recognize4_xy_zu(D, x, z, y, u)
    else:
        return _recognize4_xy_zu(D, x, u, y, z)


def recognize4_matrix_only(D):

    return recognize4_new(D, 0, 1, 2, 3)


def _compute_delta_x(alpha, xz, d_xy, delta_z):

    return xz - (1-alpha) * d_xy - delta_z


def _compute_delta_y(alpha, yz, d_xy, delta_z):

    return yz - alpha * d_xy - delta_z


def _compute_delta_z(xy, xz, yz):

    return 0.5 * (xz + yz - xy)


def _compute_d_xy(alpha, xz, yz, ux, uy, uz, delta_z):

    return (   (uz - alpha * ux - (1-alpha) * uy
                - 2 * delta_z + alpha * xz + (1-alpha) * yz)
            / (2 * alpha * (1-alpha))   )


def _close_to_equal(a):

    if np.isclose(a, 0.0):
        return 0.0
    elif np.isclose(a, 1.0):
        return 1.0
    else:
        return a


def _non_negative(a):

    return np.isclose(a, 0.0) or a > 0.0


def _all_non_negative(a):

    for val in a:
        if not _non_negative(val):
            return False

    return True


def _compute_alpha(V, D, x, y, z, u, v):

    x = V.index(x)
    y = V.index(y)
    z = V.index(z)
    u = V.index(u)
    v = V.index(v)

    numerator   = (D[u,z] + D[v,y]) - (D[v,z] + D[u,y])
    denominator = (D[u,x] + D[v,y]) - (D[v,x] + D[u,y])

    if not np.isclose(denominator, 0.0):
        return numerator / denominator
    else:
        return np.nan


def _find_candidates(D, V, print_info, B=None):
    candidates = []
    n = len(V)

    if print_info: print(f'-----> n = {n}, V = {V} ---> Candidates')

    for x, y, z in permutations(V, 3):
        # If z is in the list of original leafes, throw this combination away.
        if B is not None:
            if z in B:
                continue
        # considering x < y suffices
        if x > y:
            continue

        alpha = np.zeros(( (n-3) * (n-4) // 2 ,))

        pos = 0
        u_witness = None
        for u, v in combinations(V, 2):
            if u in  (x, y, z) or v in (x, y, z):
                continue

            alpha[pos] = _compute_alpha(V, D, x, y, z, u, v)
            if not u_witness and not np.isnan(alpha[pos]):
                u_witness = u

            pos += 1

        nan_mask = np.isnan(alpha)

        if not np.any(nan_mask) and np.allclose(alpha, alpha[0]):
            alpha[0] = _close_to_equal(alpha[0])

            if alpha[0] >= 0.0 and alpha[0] <= 1.0:
                candidates.append((x, y, z, u_witness, alpha[0]))
                deltas = _compute_deltas(V, D, alpha[0], x, y, z, u_witness)

                if print_info:
                    print(f'({x}, {y}: {z}) alpha={alpha}', end='   ')
                    print('δx = {:.3f}, δy = {:.3f}, '\
                          'δz = {:.3f}, dxy = {:.3f}'.format(deltas[2],
                                                             deltas[3],
                                                             deltas[0],
                                                             deltas[1]))

        elif not np.all(nan_mask):

            ref_alpha = alpha[ np.argmin(nan_mask) ]
            masked_alpha = np.ma.array(alpha, mask=nan_mask)

            if np.ma.allclose(masked_alpha, ref_alpha, masked_equal=True):
                ref_alpha = _close_to_equal(ref_alpha)
                if ref_alpha >= 0.0 and ref_alpha <= 1.0:
                    candidates.append((x, y, z, u_witness, ref_alpha))

        else:
            # choose an arbitrary alpha (e.g. 0.5) and witness u (?)
            ref_alpha, u_witness = 0.5, None
            for u in V:
                if u not in (x, y, z):
                    u_witness = u
                    break
            candidates.append((x, y, z, u_witness, ref_alpha))

    return candidates


def _compute_deltas(V, D, alpha, x, y, z, u):

    x = V.index(x)
    y = V.index(y)
    z = V.index(z)
    u = V.index(u)

    delta_z = _compute_delta_z(D[x,y], D[x,z], D[y,z])

    # handle alpha in {0, 1}
    if alpha == 0.0 or alpha == 1.0:
        return delta_z, D[x,y], 0.0, 0.0

    d_xy = _compute_d_xy(alpha, D[x,z], D[y,z], D[u,x], D[u,y], D[u,z], delta_z)
    delta_x = _compute_delta_x(alpha, D[x,z], d_xy, delta_z)
    delta_y = _compute_delta_y(alpha, D[y,z], d_xy, delta_z)

    return delta_z, d_xy, delta_x, delta_y


def _update_matrix(V, D, x, y, delta_x, delta_y):

    x = V.index(x)
    y = V.index(y)

    if delta_x:             # if not 0.0
        D[:, x] -= delta_x
        D[x, :] -= delta_x
        D[x, x] = 0.0

    if delta_y:             # if not 0.0
        D[:, y] -= delta_y
        D[y, :] -= delta_y
        D[y, y] = 0.0


def _matrix_without_index(D, index):

    n = D.shape[0]

    if index < 0 or index >= n:
        raise IndexError(f"Index {index} is out of range!")

    D_new = np.zeros((n-1, n-1))

    indices = [i for i in range(n) if i != index]

    for i in range(n-1):
        for j in range(n-1):
            D_new[i, j] = D[indices[i], indices[j]]

    return D_new


def _finalize_tree(recognition_tree):

    def _sort_children(v):
        v.children.sort(key=lambda c: c.R_step)
        for c in v.children:
            _sort_children(c)

    for v in recognition_tree.postorder():
        if v.valid_ways and v.parent:
            v.parent.valid_ways += v.valid_ways

    recognition_tree.valid_ways = recognition_tree.root.valid_ways
    recognition_tree.successes = recognition_tree.root.valid_ways

    _sort_children(recognition_tree.root)


def recognize(D, first_candidate_only=False, print_info=False, B=None, use_spike_length=False, use_erdbeermet_computation=False):
    """Recognition of type R matrices.

    Parameters
    ----------
    D : 2-dimensional numpy array
        A distance matrix.
    first_candidate_only : bool, optional
        If True, only consider the first found candidate for a merge event.
        The default is False.
    print_info : bool, True
        If True, print the recognition history. The default is False.
    B : Set of ints
        A set of leaf identifiers for the first 3-4 leafes of the simulation

    Returns
    -------
    Tree
        The recognition tree.

    See also
    --------
    tools.Tree
    """

    n = D.shape[0]
    V = [i for i in range(n)]

    recognition_tree = Tree(TreeNode(n, V, D=D))
    stack = []

    # trivial failure if not a pseudometric
    if not is_pseudometric(D):
        if print_info: print('no pseudometric')
        recognition_tree.root.info = 'no pseudometric'

    # every pseudometric is additve and thus also an R matrix
    elif n <= 3:
        if print_info: print(print(f'SUCCESS on {V}'))
        recognition_tree.root.valid_ways = 1

    # otherwise start the recognition algorithm
    else:
        stack.append(recognition_tree.root)

    while stack:

        parent = stack.pop()
        V, D = parent.V, parent.D
        n = len(V)

        if n > 4:

            candidates = _find_candidates(D, V, print_info, B)

            # Spikelength-wise reordering of candidates
            if use_spike_length and len(candidates) > 1:
                candidates = reorderBySpikelength(candidates=candidates, V=V, D=D, use_erdbeermet_computation=use_erdbeermet_computation)

            # Continue algorithm
            found_valid = False

            if print_info:
                print(f'-----> n = {n}, V = {V} ---> R-steps actually carried out')
            for x, y, z, u_witness, alpha in candidates:

                V_copy = V.copy()
                V_copy.remove(z)

                child = TreeNode(n-1, V_copy, R_step=(x, y, z, alpha))
                parent.add_child(child)

                deltas = _compute_deltas(V, D, alpha, x, y, z, u_witness)

                if print_info:
                    print('({}, {}: {}) alpha={:.5f}'.format(x, y, z, alpha),
                          end='   ')
                    print('δx = {:.3f}, δy = {:.3f}, '\
                          'δz = {:.3f}, dxy = {:.3f}'.format(deltas[2],
                                                             deltas[3],
                                                             deltas[0],
                                                             deltas[1]))

                if not _all_non_negative(deltas):
                    if print_info: print('         |___ negative δ/dxy')
                    child.info = 'negative delta/dxy'
                    continue

                D_copy = _matrix_without_index(D, V.index(z))
                _update_matrix(V_copy, D_copy, x, y, deltas[2], deltas[3])
                child.D = D_copy

                still_metric, metric_info = is_pseudometric(D_copy,
                                                            return_info=True,
                                                            V=V_copy)

                if not still_metric:
                    if print_info: print( '         |___ no pseudometric')
                    if print_info: print(f'         |___ {metric_info}')
                    child.info = 'no pseudometric'
                    continue

                found_valid = True
                if print_info: print(f'         |___ STACKED {V_copy}')
                stack.append(child)

                # for n = 5 always check all candidates
                if first_candidate_only and n > 5:
                    break

            if not candidates or not found_valid:
                parent.info = 'no candidate'

        else:
            if print_info: print(f'-----> n = {n} R-map test')
            if recognize4_matrix_only(D):
                if print_info: print(f'SUCCESS on {V}')
                parent.valid_ways = 1
            else:
                if print_info: print(f'NO R-MAP on {V}')
                parent.info = 'spikes too short'

    _finalize_tree(recognition_tree)
    return recognition_tree

def reorderBySpikelength(candidates, V, D, use_erdbeermet_computation): 
    spikelength_dict = {}
    for current_candidate in candidates:

        if use_erdbeermet_computation:
            # Erbeermet approach by using their function
            (delta_z, _ , delta_x, delta_y) = _compute_deltas(V=V,
                                                            D=D,
                                                            alpha=current_candidate[4],
                                                            x=current_candidate[0],
                                                            y=current_candidate[1],
                                                            z=current_candidate[2],
                                                            u=current_candidate[3])

            spikelength_dict[current_candidate] = (delta_x, delta_y, delta_z)
        
        else:
            # Computational approach from WP4
            # Copied from compute_alpha - if we don't use it, the index is outOfBounds for D.
            x = V.index(current_candidate[0])
            y = V.index(current_candidate[1])
            z = V.index(current_candidate[2])

            # Calculate spike lengths fur arbitrary u in V
            delta_z = _delta_z(D=D, x=x, y=y, z=z)
            delta_x = 0.0
            delta_y = 0.0
            counter = 0
            for u in V:
                if u in [x,y,z]:
                    continue
                else:
                    u = V.index(u)
                    delta_x += _delta_x(D=D, x=x, y=y, z=z, u=u, alpha=current_candidate[4])
                    delta_y += _delta_y(D=D, x=x, y=y, z=z, u=u, alpha=current_candidate[4])
                    counter += 1
            # Collect resulting deltas
            if counter != 0:
                final_delta_x = delta_x / counter
                final_delta_y = delta_y / counter
            else:
                final_delta_x = 0.0
                final_delta_y = 0.0
            spikelength_dict[current_candidate] = (final_delta_x, final_delta_y, delta_z)
    
    # Now compare candidates 
    temp_dict = {}
    for key1 in spikelength_dict:
        smaller_counter = 0
        for key2 in spikelength_dict:
            if key1 == key2:
                 continue
            else:
                if (spikelength_dict[key1][0] < spikelength_dict[key2][0] 
                    or spikelength_dict[key1][1] < spikelength_dict[key2][1] 
                    or spikelength_dict[key1][2] < spikelength_dict[key2][2]):
                    smaller_counter += 1

            temp_dict[key1] = smaller_counter

    # Order the dict descending because a higher smaller_counter means its smaller then the others and construct the list therefore. 
    sorted_list = sorted(temp_dict.items(), key = lambda item: item[1], reverse = True)

    # Delete the smaller_count entry in the lists and return
    return [a for (a,b) in sorted_list]

# Helper functions for Spike-Length Computation
def _delta_abc(D, a, b, c):
    return 0.5 * (D[a,c] + D[b,c] - D[a,b]) 

def _delta_x(D, x, y, z, u, alpha):
    return _delta_abc(D=D, a=u, b=y, c=x) - ((1/(2*alpha)) * (D[x,y] + D[z,u] - D[x,z] - D[y,u]))


def _delta_y(D, x, y, z, u, alpha):
    return _delta_abc(D=D, a=u, b=x, c=y) - ((1/(2*(1-alpha))) * (D[x,y] + D[z,u] - D[x,u] - D[y,z]))

def _delta_z(D, x, y, z):
    return _delta_abc(D=D, a=x, b=y, c=z)