import warnings

import numpy as np
from numba import errors, njit, objmode
from numba.types import complex128
from scipy.linalg import expm

warnings.simplefilter('ignore', category=errors.NumbaPerformanceWarning)

COMPLEX_MATRIX_TYPE = complex128[:, :]


@njit
def t_dot(a, b):
    c = np.zeros_like(a)
    n, _ = a.shape
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i, j] += a[k, i] * b[k, j]
    return c


@njit
def det_f_with_derivative(u, t, v, beta, q, m, r):
    n, _ = v.shape
    iu = 1j * u

    z = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    z[:n, :n] = m
    z[:n, -n:] = -2 * t_dot(q, q)
    z[-n:, :n] = (iu * (iu - 1) / 2) * np.eye(n)
    z[-n:, -n:] = -(m.T + 2 * iu * t_dot(r, q))

    dz = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    dz[-n:, :n] = -(u + 0.5j) * np.eye(n)
    dz[-n:, -n:] = -2j * t_dot(r, q)

    with objmode(exp_tz=COMPLEX_MATRIX_TYPE):
        exp_tz = expm(t * z)
    f = exp_tz[-n:, -n:]
    df = 0.5 * t * (exp_tz @ dz + dz @ exp_tz)[-n:, -n:]

    det_f = np.linalg.det(f)
    d_det_f = det_f * np.trace(np.linalg.solve(f, df))

    return det_f, d_det_f


@njit
def sine_angle_with_derivative(x, y):
    l2_sq = x ** 2 + y ** 2
    l2 = np.sqrt(l2_sq)
    sin = y / l2
    d_dx = -x * y / (l2 * l2_sq)
    d_dy = x ** 2 / (l2 * l2_sq)
    return sin, d_dx, d_dy


@njit
def sine_det_f_with_derivative(u, t, v, beta, q, m, r):
    det_f, d_det_f = det_f_with_derivative(u, t, v, beta, q, m, r)
    sin, d_dx, d_dy = sine_angle_with_derivative(det_f.real, det_f.imag)
    d_sin = d_det_f.real * d_dx + d_det_f.imag * d_dy
    return sin, d_sin


@njit
def cosine_det_f_with_derivative(u, t, v, beta, q, m, r):
    det_f, d_det_f = det_f_with_derivative(u, t, v, beta, q, m, r)
    cos, d_dy, d_dx = sine_angle_with_derivative(det_f.imag, det_f.real)
    d_cos = d_det_f.real * d_dx + d_det_f.imag * d_dy
    return cos, d_cos


@njit
def find_next_quadrant(u, curr_quadrant, u_max, eps, delta, curr_n_iter, max_n_iter, t, v, beta, q, m, r):
    sign = -1 if (curr_quadrant == 0) or (curr_quadrant == 3) else 1
    for curr_n_iter in range(curr_n_iter, max_n_iter):
        if (curr_quadrant == 0) or (curr_quadrant == 2):
            sin, d_sin = sine_det_f_with_derivative(u, t, v, beta, q, m, r)
            target, d_target = sign * sin + 1, sign * d_sin
        else:
            cos, d_cos = cosine_det_f_with_derivative(u, t, v, beta, q, m, r)
            target, d_target = sign * cos + 1, sign * d_cos
        step = -target / d_target
        u += step
        converged = target <= eps
        if curr_quadrant == 1:
            converged &= step <= delta
        if converged:
            return u, curr_n_iter, True
        if u.real > u_max.real:
            return u, curr_n_iter, False
    return u, curr_n_iter + 1, False


@njit
def count_rotations(u, t, v, beta, q, m, r, rot_locs, cached_u, cached_quadrant):
    u_max = u
    u, = cached_u
    curr_quadrant, = cached_quadrant
    assert u.imag == u_max.imag
    if u.real <= u_max.real:
        eps = 1e-2
        delta = 1e-3
        max_n_iter = 1_000
        curr_n_iter = 0
        not_converged = True
        while not_converged:
            u, curr_n_iter, not_converged = find_next_quadrant(
                u, curr_quadrant, u_max, eps, delta, curr_n_iter, max_n_iter,
                t, v, beta, q, m, r
            )
            if curr_n_iter == max_n_iter:
                break
            curr_quadrant = (curr_quadrant + 1) % 4
            if not_converged and (curr_quadrant == 2):
                rot_locs.append(u)
        cached_u[0] = u
        cached_quadrant[0] = curr_quadrant
    rots = 0
    for rot_loc in rot_locs:
        if rot_loc.real < u_max.real:
            rots += 1
    return rots
