import numpy as np


def pack_vol(v, n):
    return v[np.triu_indices(n)]


def unpack_vol(v_packed, n):
    v = np.zeros((n, n))
    v[np.triu_indices(n)] = v_packed
    v[np.tril_indices(n, -1)] = v[np.triu_indices(n, 1)]
    return v


def pack_params(v, beta, q, m, r, n):
    v_end = n * (n + 1) // 2
    beta_end = v_end + 1
    q_end = beta_end + n ** 2
    m_end = q_end + n ** 2
    r_end = m_end + n ** 2
    params = np.zeros(r_end)
    params[:v_end] = pack_vol(v, n)
    params[v_end] = beta
    params[beta_end:q_end] = q.ravel()
    params[q_end:m_end] = m.ravel()
    params[m_end:r_end] = r.ravel()
    return params


def unpack_params(params, n):
    v_end = n * (n + 1) // 2
    beta_end = v_end + 1
    q_end = beta_end + n ** 2
    m_end = q_end + n ** 2
    r_end = m_end + n ** 2
    v = unpack_vol(params[:v_end], n)
    beta = params[v_end]
    q = params[beta_end:q_end].reshape((n, n))
    m = params[q_end:m_end].reshape((n, n))
    r = params[m_end:r_end].reshape((n, n))
    return v, beta, q, m, r
