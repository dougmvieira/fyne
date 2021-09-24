import cmath

import numpy as np
from numba import njit, objmode, typed
from numba.types import ListType, Tuple, UniTuple, boolean, complex128, float64, int64
from scipy.integrate import quad
from scipy.linalg import expm


COMPLEX_MATRIX_TYPE = complex128[:, :]


#@njit
def characteristic_function(u, t, v, beta, q, m, r, rot_locs, cached_u, cached_quadrant):
    n = 2
    iu = 1j * u

    z = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    z[:n, :n] = m
    z[:n, -n:] = -2 * q.T @ q
    z[-n:, :n] = (iu * (iu  - 1) / 2) * np.eye(n)
    z[-n:, -n:] = -(m.T + 2 * iu * (r.T @ q))

    with objmode(exp_tz=COMPLEX_MATRIX_TYPE):
        exp_tz = expm(t * z)

    if np.any(np.isnan(exp_tz)) or np.any(np.isinf(exp_tz)):
        return 0
    if np.absolute(np.trace(exp_tz)) > 1e9:
        return 0
    g = exp_tz[-n:, :n]
    f = exp_tz[-n:, -n:]

    rot = count_rotations(u, t, v, beta, q, m, r, rot_locs, cached_u, cached_quadrant)
    log_det_f = cmath.log(np.linalg.det(f)) + 2j * cmath.pi * rot

    a = np.linalg.solve(f, g)
    c = (
        -iu * t * beta * np.sum(r * q)
        - (beta / 2) * (np.trace(m) * t + log_det_f)
    )
    return cmath.exp(np.sum(a * v) + c)


#@njit(UniTuple(complex128, 2)(complex128, float64, float64[:, :], float64, float64[:, :], float64[:, :], float64[:, :]))
def det_f_with_derivative(u, t, v, beta, q, m, r):
    n = 2
    iu = 1j * u

    z = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    z[:n, :n] = m
    z[:n, -n:] = -2 * q.T @ q
    z[-n:, :n] = (iu * (iu  - 1) / 2) * np.eye(n)
    z[-n:, -n:] = -(m.T + 2 * iu * (r.T @ q))

    dz = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    dz[-n:, :n] = -(u + 0.5j) * np.eye(n)
    dz[-n:, -n:] = -2j * (r.T @ q)

    with objmode(exp_tz=COMPLEX_MATRIX_TYPE):
        exp_tz = expm(t * z)
    f = exp_tz[-n:, -n:]
    df = 0.5 * t * (exp_tz @ dz + dz @ exp_tz)[-n:, -n:]

    det_f = np.linalg.det(f)
    d_det_f = det_f * np.trace(np.linalg.solve(f, df))

    return det_f, d_det_f


#@njit(UniTuple(float64, 3)(float64, float64))
def sine_angle_with_derivative(x, y):
    l_sq = x ** 2 + y ** 2
    l = np.sqrt(l_sq)
    sin = y / l
    d_dx = -x * y / (l * l_sq)
    d_dy = x ** 2 / (l * l_sq)
    return sin, d_dx, d_dy


#@njit(UniTuple(float64, 2)(complex128, float64, float64[:, :], float64, float64[:, :], float64[:, :], float64[:, :]))
def sine_det_f_with_derivative(u, t, v, beta, q, m, r):
    det_f, d_det_f = det_f_with_derivative(u, t, v, beta, q, m, r)
    sin, d_dx, d_dy = sine_angle_with_derivative(det_f.real, det_f.imag)
    d_sin = d_det_f.real * d_dx + d_det_f.imag * d_dy
    return sin, d_sin


#@njit(UniTuple(float64, 2)(complex128, float64, float64[:, :], float64, float64[:, :], float64[:, :], float64[:, :]))
def cosine_det_f_with_derivative(u, t, v, beta, q, m, r):
    det_f, d_det_f = det_f_with_derivative(u, t, v, beta, q, m, r)
    cos, d_dy, d_dx = sine_angle_with_derivative(det_f.imag, det_f.real)
    d_cos = d_det_f.real * d_dx + d_det_f.imag * d_dy
    return cos, d_cos


#@njit(Tuple([complex128, int64, boolean])(complex128, int64, complex128, float64, float64, int64, int64, float64, float64[:, :], float64, float64[:, :], float64[:, :], float64[:, :]))
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


##@njit(int64(complex128, float64, float64[:, :], float64, float64[:, :], float64[:, :], float64[:, :]))
#@njit
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
            u, curr_n_iter, not_converged = find_next_quadrant(u, curr_quadrant, u_max, eps, delta, curr_n_iter, max_n_iter, t, v, beta, q, m, r)
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


def delta(k, t, v, beta, q, m, r):
    rot_locs = typed.List(lsttype=ListType(complex128))
    cached_u, cached_quadrant = np.array([-1j]), np.array([0])
    phi_inv = 1 / characteristic_function(-1j, t, v, beta, q, m, r, rot_locs, cached_u, cached_quadrant)
    def integrand(u):
        phi_num = characteristic_function(u - 1j, t, v, beta, q, m, r, rot_locs, cached_u, cached_quadrant)
        integrand = (cmath.exp(-1j * u * k) * phi_num * phi_inv / (1j * u)).real
        return integrand
    return 0.5 + quad(integrand, 0, np.inf)[0] / np.pi


def formula(k, t, v, beta, q, m, r):
    rot_locs = typed.List(lsttype=ListType(complex128))
    cached_u, cached_quadrant = np.array([-0.5j]), np.array([0])
    def integrand(u):
        psi = characteristic_function(u - 0.5j, t, v, beta, q, m, r, rot_locs, cached_u, cached_quadrant)
        integrand = (np.exp((0.5 - 1j * u) * k) * psi).real/(u ** 2 + 0.25)
        return integrand
    c = 1 - quad(integrand, 0, np.inf)[0] / np.pi

    return c


DEMO_PARAMS = dict(
    v=np.array([[0.0327, 0.0069],
                [0.0069, 0.0089]]),
    beta=0.6229,
    m=np.array([[-0.9858, -0.5224],
                [-0.1288, -0.9746]]),
    q=np.array([[0.3193, 0.2590],
                [0.2899, 0.2469]]),
    r=np.array([[-0.2116, -0.4428],
                [-0.2113, -0.5921]]),
)

#rot_locs = []
#cached_args = [0, 0]
#u_arr = np.linspace(0, 100, 1000)
#char_fun_arr = np.array([characteristic_function(u, 1/12, rot_locs=rot_locs, cached_args=cached_args, **DEMO_PARAMS) for u in u_arr])
expected_delta = 0.5592811030824658
expected_formula = 0.022944330308010463
assert np.abs(delta(0, 1/12, **DEMO_PARAMS) - expected_delta) < 1e-16
assert np.abs(formula(0, 1/12, **DEMO_PARAMS) - expected_formula) < 1e-16
