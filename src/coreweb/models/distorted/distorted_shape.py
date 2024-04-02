# -*- coding: utf-8 -*-

from typing import Tuple

import numba
import numpy as np
from numba import guvectorize

from coreweb.rotqs import _numba_quaternion_rotate

s_h = 0.00002


@numba.njit(cache=True)
def distorted_bfield(
    mu_i: float,
    nu_i: float,
    s_i: float,
    rho_0: float,
    rho_1: float,
    tv: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    k1: float,
    k2: float,
    delta: float,
    twist: float,
) -> None:
    # ellipse circumference
    delta_ell = mu_i * delta + 1 - mu_i
    h = (delta_ell - 1) ** 2 / (1 + delta_ell) ** 2
    Efac = np.pi * (1 + delta_ell) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))

    # correct twist for ellipse circumference (as factor)
    t = twist / Efac * rho_1

    Df_ev = Df(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
    Dfdmu_ev = Dfdmu(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
    Dfdnu_ev = Dfdnu(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
    Dfds_ev = Dfds(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
    DOmdmu_ev = DOmdmu(mu_i, nu_i, rho_1, k1, k2)
    DOmdnu_ev = DOmdnu(mu_i, nu_i, rho_1, k1, k2)

    cosDOm = np.cos(DOm(mu_i, nu_i, rho_1, k1, k2))
    sinDOm = np.sin(DOm(mu_i, nu_i, rho_1, k1, k2))

    Df_ev_const_nu = Df(mu_i, 0, s_i, rho_1, k1, k2, delta)
    Df_ev_const_s = Df(mu_i, nu_i, 0.5, rho_1, k1, k2, delta)

    # corrections to make solution more classical
    corr_const_s = Dfdmu(mu_i, nu_i, 0.5, rho_1, k1, k2, delta) * DOmdnu(
        mu_i, nu_i, rho_1, k1, k2
    ) - Dfdnu(mu_i, nu_i, 0.5, rho_1, k1, k2, delta) * DOmdmu(mu_i, nu_i, rho_1, k1, k2)

    # curvature
    K = 1 - Df_ev * (k1 * cosDOm + k2 * sinDOm)

    # metric (without D)
    goDf = Df_ev * K * (Dfdmu_ev * DOmdnu_ev - Dfdnu_ev * DOmdmu_ev)

    denom = 1 + t**2 * mu_i**2

    # chi and xi (be careful with nu/s dependencies here)
    b_chi = Df_ev_const_nu * t / denom
    b_xi = Df_ev_const_s * corr_const_s / denom

    # apply metric
    b_nu = b_chi / goDf
    b_s = b_xi / goDf

    # basis vectors
    eps_nu = Dfdnu_ev * (n1 * cosDOm + n2 * sinDOm) + Df_ev * DOmdnu_ev * (
        -n1 * sinDOm + n2 * cosDOm
    )

    eps_s = K * tv + Dfds_ev * (n1 * cosDOm + n2 * sinDOm)

    return (b_nu, b_s, eps_nu, eps_s)


@numba.njit(cache=True)
def distorted_rhofield(
    mu_i: float,
    nu_i: float,
    s_i: float,
    rho_0: float,
    rho_1: float,
    tv: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    k1: float,
    k2: float,
    delta: float,
    twist: float,
) -> None:
    # ellipse circumference
    delta_ell = mu_i * delta + 1 - mu_i
    h = (delta_ell - 1) ** 2 / (1 + delta_ell) ** 2
    Efac = np.pi * (1 + delta_ell) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))

    # correct twist for ellipse circumference (as factor)
    t = twist / Efac * rho_1

    Df_ev = Df(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
    Dfdmu_ev = Dfdmu(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
    Dfdnu_ev = Dfdnu(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
    Dfds_ev = Dfds(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
    DOmdmu_ev = DOmdmu(mu_i, nu_i, rho_1, k1, k2)
    DOmdnu_ev = DOmdnu(mu_i, nu_i, rho_1, k1, k2)

    cosDOm = np.cos(DOm(mu_i, nu_i, rho_1, k1, k2))
    sinDOm = np.sin(DOm(mu_i, nu_i, rho_1, k1, k2))

    Df_ev_const_nu = Df(mu_i, 0, s_i, rho_1, k1, k2, delta)
    Df_ev_const_s = Df(mu_i, nu_i, 0.5, rho_1, k1, k2, delta)

    # corrections to make solution more classical
    corr_const_s = Dfdmu(mu_i, nu_i, 0.5, rho_1, k1, k2, delta) * DOmdnu(
        mu_i, nu_i, rho_1, k1, k2
    ) - Dfdnu(mu_i, nu_i, 0.5, rho_1, k1, k2, delta) * DOmdmu(mu_i, nu_i, rho_1, k1, k2)

    # curvature
    K = 1 - Df_ev * (k1 * cosDOm + k2 * sinDOm)

    # metric
    goDf = Df_ev * K * (Dfdmu_ev * DOmdnu_ev - Dfdnu_ev * DOmdmu_ev)

    denom = 1 + t**2 * mu_i**2

    # chi and xi (be careful with nu/s dependencies here)
    b_chi = Df_ev_const_nu * t / denom
    b_xi = Df_ev_const_s * corr_const_s / denom

    # apply metric
    b_nu = b_chi / goDf
    b_s = b_xi / goDf

    # basis vectors
    eps_nu = Dfdnu_ev * (n1 * cosDOm + n2 * sinDOm) + Df_ev * DOmdnu_ev * (
        -n1 * sinDOm + n2 * cosDOm
    )

    eps_s = K * tv + Dfds_ev * (n1 * cosDOm + n2 * sinDOm)

    bv = b_nu * eps_nu + b_s * eps_s

    return np.dot(bv, bv)


@numba.njit(cache=True)
def gamma(
    s: float, alpha: float, beta: float, lambda_v: float, epsilon: float, kappa: float
) -> np.ndarray:
    return np.array(
        [
            2
            * (
                (1 - np.cos(2 * np.pi * s)) ** (1 - alpha)
                + np.exp(-np.abs((s - 0.5) ** kappa))
                - np.exp(-(0.5**kappa))
            )
            / (2 ** (1 - alpha) + 0.11 * kappa),
            lambda_v
            * np.sin(2 * np.pi * s)
            * np.sin(np.pi * s) ** alpha
            / np.sin(np.pi / 2)
            / np.sin(np.pi / 4) ** alpha,
            beta * np.sin(np.pi * s) ** 2 * np.sin(epsilon * np.pi * s),
        ]
    )


@numba.njit(cache=True)
def dgds(
    s: float, alpha: float, beta: float, lambda_v: float, epsilon: float, kappa: float
) -> np.ndarray:
    if s < 2 * s_h:
        return (
            gamma(s + s_h, alpha, beta, lambda_v, epsilon, kappa)
            - gamma(s, alpha, beta, lambda_v, epsilon, kappa)
        ) / s_h
    elif s > 1 - 2 * s_h:
        return (
            gamma(s, alpha, beta, lambda_v, epsilon, kappa)
            - gamma(s - s_h, alpha, beta, lambda_v, epsilon, kappa)
        ) / s_h
    else:
        return (
            (
                gamma(s + s_h, alpha, beta, lambda_v, epsilon, kappa)
                - gamma(s - s_h, alpha, beta, lambda_v, epsilon, kappa)
            )
            / 2
            / s_h
        )


@numba.njit(cache=True)
def Df(
    mu: float, nu: float, s: float, rho_1: float, k1: float, k2: float, delta: float
) -> float:
    df = mu / delta + 1 - mu

    phi = DOm(mu, nu, rho_1, k1, k2)

    # return rho_1 * mu * np.abs(np.sin(np.pi * s)) ** 2

    return (
        rho_1
        * mu
        / np.sqrt(np.cos(phi) ** 2 + (df * np.sin(phi)) ** 2)
        * np.abs(np.sin(np.pi * s)) ** 2
    )

    # return (
    #     rho_1
    #     * mu
    #     / np.sqrt(np.cos(phi) ** 2 + (1 / delta * np.sin(phi)) ** 2)
    #     * np.abs(np.sin(np.pi * s)) ** 2
    # )


@numba.njit(cache=True)
def Dfdmu(
    mu: float, nu: float, s: float, rho_1: float, k1: float, k2: float, delta: float
) -> float:
    if mu > s_h:
        return (
            (
                Df(mu + s_h, nu, s, rho_1, k1, k2, delta)
                - Df(mu - s_h, nu, s, rho_1, k1, k2, delta)
            )
            / 2
            / s_h
        )
    else:
        return (
            Df(mu + s_h, nu, s, rho_1, k1, k2, delta)
            - Df(mu, nu, s, rho_1, k1, k2, delta)
        ) / s_h


@numba.njit(cache=True)
def Dfdnu(
    mu: float, nu: float, s: float, rho_1: float, k1: float, k2: float, delta: float
) -> float:
    return (
        (
            Df(mu, nu + s_h, s, rho_1, k1, k2, delta)
            - Df(mu, nu - s_h, s, rho_1, k1, k2, delta)
        )
        / 2
        / s_h
    )


@numba.njit(cache=True)
def Dfds(
    mu: float, nu: float, s: float, rho_1: float, k1: float, k2: float, delta: float
) -> float:
    return (
        (
            Df(mu, nu, s + s_h, rho_1, k1, k2, delta)
            - Df(mu, nu, s - s_h, rho_1, k1, k2, delta)
        )
        / 2
        / s_h
    )


@numba.njit(cache=True)
def get_n_vectors(
    gv: np.ndarray,
    s: float,
    rho_0: float,
    alpha: float,
    beta: float,
    lambda_v: float,
    epsilon: float,
    kappa: float,
) -> Tuple[np.ndarray, np.ndarray]:
    # compute t vector numerically
    dgdvs = (
        (
            rho_0 * gamma(s + 0.005, alpha, beta, lambda_v, epsilon, kappa)
            - rho_0 * gamma(s - 0.005, alpha, beta, lambda_v, epsilon, kappa)
        )
        / 2
        / s_h
    )

    # second order derivative
    dgdvs2 = (
        rho_0 * gamma(s + 0.005, alpha, beta, lambda_v, epsilon, kappa)
        - 2 * gv
        + rho_0 * gamma(s - 0.005, alpha, beta, lambda_v, epsilon, kappa)
    ) / s_h**2

    # get curve velocity vel
    vel = np.linalg.norm(dgdvs)
    tv = dgdvs / vel

    # derivative of tv, this is a sum of vk1n1 and vk2n2, written in a compact form
    dstv = -((np.dot(dgdvs, dgdvs2)) * dgdvs + vel**2 * dgdvs2) / vel**3

    # n1 candidate pointing towards (rho_0, 0, 0)
    n1c = np.array([rho_0, 0, 0]) - gv

    # orthogonalize vector
    n1cu = n1c - tv * np.dot(tv, n1c)

    # normalize
    n1 = n1cu / np.linalg.norm(n1cu)

    # n2 by cross product
    n2 = np.cross(tv, n1)

    # normalization correction
    tv = tv / np.linalg.norm(tv)
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    # get k1/k2
    k1 = -np.dot(n1, dstv) / vel
    k2 = -np.dot(n2, dstv) / vel

    return tv, n1, n2, k1, k2


# gold-hoyle style DOm
@numba.njit(cache=True)
def DOm(mu: float, nu: float, rho_1: float, k1: float, k2: float) -> float:
    # shift result by 2 * pi to keep function continous, this is important for derivatives
    shift = np.floor(np.abs(nu))
    shift_sgn = np.sign(nu)

    return 2 * np.pi * shift * shift_sgn + 2 * np.arctan(
        rho_1 * mu * k2
        + np.tan(np.pi * (nu + 0.5)) * np.sqrt(1 - (rho_1 * mu) ** 2 * (k1**2 + k2**2))
    )


# # normal DOm
# @numba.njit(cache=True)
# def DOm(mu: float, nu: float, rho_1: float, k1: float, k2: float) -> float:
#     return np.pi * (2 * nu - 1)


@numba.njit(cache=True)
def DOmdmu(mu: float, nu: float, rho_1: float, k1: float, k2: float) -> float:
    if mu > s_h:
        return (
            (DOm(mu + s_h, nu, rho_1, k1, k2) - DOm(mu - s_h, nu, rho_1, k1, k2))
            / 2
            / s_h
        )
    else:
        return (DOm(mu + s_h, nu, rho_1, k1, k2) - DOm(mu, nu, rho_1, k1, k2)) / s_h


@numba.njit(cache=True)
def DOmdnu(mu: float, nu: float, rho_1: float, k1: float, k2: float) -> float:
    return (
        (DOm(mu, nu + s_h, rho_1, k1, k2) - DOm(mu, nu - s_h, rho_1, k1, k2)) / 2 / s_h
    )


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(i), (j), (k), (l) -> (i)",
    target="parallel",
)
def distorted_qs(
    q: np.ndarray,
    iparams: np.ndarray,
    sparams: np.ndarray,
    q_xs: np.ndarray,
    s: np.ndarray,
) -> None:
    (q0, q1, q2) = q

    (
        _,
        _,
        _,
        _,
        _,
        delta,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        alpha,
        beta,
        lambda_v,
        epsilon,
        kappa,
    ) = iparams
    (_, rho_0, rho_1, b_t, _) = sparams

    gv = rho_0 * gamma(q2, alpha, beta, lambda_v, epsilon, kappa)

    _, n1, n2, k1, k2 = get_n_vectors(
        gv, q2, rho_0, alpha, beta, lambda_v, epsilon, kappa
    )

    r = gv + Df(q0, q1, q2, rho_1, k1, k2, delta) * (
        n1 * np.cos(DOm(q0, q1, rho_1, k1, k2))
        + n2 * np.sin(DOm(q0, q1, rho_1, k1, k2))
    )

    x = np.array([0, r[0], r[1], r[2]])

    s[:] = _numba_quaternion_rotate(x, q_xs)


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(i), (j), (k), (l), (l) -> (i)",
    fastmath=True,
)
def distorted_sq_gh(
    s: np.ndarray,
    iparams: np.ndarray,
    sparams: np.ndarray,
    q_sx: np.ndarray,
    q_xs: np.ndarray,
    q: np.ndarray,
) -> None:
    (
        _,
        _,
        _,
        _,
        d1au,
        delta,
        _,
        _,
        Tfac,
        _,
        _,
        _,
        _,
        _,
        alpha,
        beta,
        lambda_v,
        epsilon,
        kappa,
    ) = iparams
    (_, rho_0, rho_1, b_t, gamma_l) = sparams

    _s = np.array([0, s[0], s[1], s[2]]).astype(s.dtype)
    xs = _numba_quaternion_rotate(_s, q_sx)

    # evaluate 20 gammas and find closest s coordinate
    s_list = np.linspace(0.1, 0.9, 50)

    s_g_list = np.array(
        [
            np.linalg.norm(rho_0 * gamma(_, alpha, beta, lambda_v, epsilon, kappa) - xs)
            for _ in s_list
        ]
    )

    # check for any NaNs if present
    s_g_list[np.isnan(s_g_list)] = 1e9

    min_arg_s = np.argmin(s_g_list)
    s_i = s_list[min_arg_s]

    s_i_max = s_list[min_arg_s + 1]
    s_i_min = s_list[min_arg_s - 1]

    # basic newton method to find s coordinate
    # typically < 5 steps are required, max 10
    # last_corr must be limited to number of initial guesses for s so that it doesnt overcorrect
    # PS: this algo must not fail or be inaccurate otherewise next step will NOT work
    last_corr = 1 / (len(s_list) + 1) / 2
    sfac = 1

    s_h2 = 5 * s_h

    for i in range(20):
        df_si = np.linalg.norm(
            rho_0 * gamma(s_i, alpha, beta, lambda_v, epsilon, kappa) - xs
        )
        df_si_plus = np.linalg.norm(
            rho_0 * gamma(s_i + s_h2, alpha, beta, lambda_v, epsilon, kappa) - xs
        )
        df_si_minus = np.linalg.norm(
            rho_0 * gamma(s_i - s_h2, alpha, beta, lambda_v, epsilon, kappa) - xs
        )

        d1f = (df_si_plus - df_si_minus) / 2 / s_h2
        d2f = (df_si_plus - 2 * df_si + df_si_minus) / (s_h2**2)

        # break if derivatives become too small
        if np.isnan(d1f / d2f):
            break
        elif np.abs(d1f / d2f) > last_corr:
            # decrease sfac until correction is smaller or hits limit
            while sfac * np.abs(d1f / d2f) > last_corr:
                sfac /= 2

                if sfac * np.abs(d1f / d2f) < 1e-8:
                    break
        elif np.abs(d1f / d2f) < 1e-8:
            break

        s_i = s_i - sfac * d1f / d2f

        # safeguard, reset to opposite side of valid interval to avoid local minima
        if s_i > s_i_max:
            s_i_p = s_i
            s_i = (s_i_max + 3 * s_i_min) / 4

            # narrow boundaries
            s_i_max = s_i_p + sfac * d1f / d2f

            sfac /= 2

        elif s_i < s_i_min:
            s_i_p = s_i
            s_i = (3 * s_i_max + s_i_min) / 4

            # narrow boundaries
            s_i_min = s_i_p + sfac * d1f / d2f

            sfac /= 2
        else:
            last_corr = sfac * np.abs(d1f / d2f)

    # cut off s_i
    if s_i < 0.15 or s_i > 0.85:
        q[:] = np.array([0, 0, 0])
        return

    # compute n1, n2 for specific s_i
    gv = rho_0 * gamma(s_i, alpha, beta, lambda_v, epsilon, kappa)
    tv, n1, n2, k1, k2 = get_n_vectors(
        gv, s_i, rho_0, alpha, beta, lambda_v, epsilon, kappa
    )

    dxs = xs - gv

    n1p = np.dot(dxs, n1)
    n2p = np.dot(dxs, n2)

    # solve dxs = Df (n1 cos DOm + n2 sin DOm)
    # this can be rewritten as a 2x2 system of equations
    # Df * cos DOm = n1p, Df * sin DOm = n2p
    # we can thus use newton to find mu/nu
    # first we need good initial estimates, assuming DOm is linear
    # note that this all works very badly if mu_i << 1
    # TODO: improve algorithm for small mu
    Np = 8
    mnu_i_0 = np.empty((Np, 2))

    for i in range(Np):
        # offset initial angle values, this fixes things for some reason (dislikes 0.0/1.0 as initial)
        mnu_i_0[i, 1] = i / Np + 1 / Np / 3
        mnu_i_0[i, 0] = np.linalg.norm(dxs) / Df(
            1, i / Np + 1 / Np / 3, s_i, rho_1, k1, k2, delta
        )

    mnu_results = np.ones((Np, 4))

    if np.all(mnu_i_0[:, 0] > 2):
        q[:] = np.array([0, 0, 0])
        return
    else:
        pass

    for k in range(Np):
        mnu_i = mnu_i_0[k]

        # dont overcorrect at start
        last_corr = 1
        sfac = 1

        for i in range(8):
            mu_i = mnu_i[0]
            nu_i = mnu_i[1]

            cosDOm = np.cos(DOm(mu_i, nu_i, rho_1, k1, k2))
            sinDOm = np.sin(DOm(mu_i, nu_i, rho_1, k1, k2))

            Fv = np.array(
                [
                    Df(mu_i, nu_i, s_i, rho_1, k1, k2, delta) * cosDOm - n1p,
                    Df(mu_i, nu_i, s_i, rho_1, k1, k2, delta) * sinDOm - n2p,
                ]
            )

            DFv = np.array(
                [
                    [
                        Dfdmu(mu_i, nu_i, s_i, rho_1, k1, k2, delta) * cosDOm
                        - Df(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
                        * sinDOm
                        * DOmdmu(mu_i, nu_i, rho_1, k1, k2),
                        Dfdnu(mu_i, nu_i, s_i, rho_1, k1, k2, delta) * cosDOm
                        - Df(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
                        * sinDOm
                        * DOmdnu(mu_i, nu_i, rho_1, k1, k2),
                    ],
                    [
                        Dfdmu(mu_i, nu_i, s_i, rho_1, k1, k2, delta) * sinDOm
                        + Df(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
                        * cosDOm
                        * DOmdmu(mu_i, nu_i, rho_1, k1, k2),
                        Dfdnu(mu_i, nu_i, s_i, rho_1, k1, k2, delta) * sinDOm
                        + Df(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
                        * cosDOm
                        * DOmdnu(mu_i, nu_i, rho_1, k1, k2),
                    ],
                ]
            )

            if np.any(np.isnan(DFv)) or np.linalg.det(DFv) < 1e-10:
                break

            corr = np.dot(np.linalg.inv(DFv), Fv)
            corr_n = np.linalg.norm(corr)

            # break if derivatives become too small
            # allow for larger re-steps
            if np.isnan(np.linalg.det(DFv)) or np.isnan(corr_n):
                break
            elif mnu_i[0] - sfac * corr[0] < 0:
                while mnu_i[0] - sfac * corr[0] < 0:
                    sfac /= 2
            elif sfac * corr_n > last_corr:
                while sfac * corr_n > last_corr:
                    sfac /= 2

                    if sfac * corr_n < 1e-16:
                        break

            elif corr_n < 1e-16:
                break

            if sfac < 0.025:
                break

            mnu_i = mnu_i - sfac * corr

            last_corr = sfac * np.linalg.norm(corr)

        mnu_results[k, 0] = mnu_i[0]
        mnu_results[k, 1] = mnu_i[1]
        mnu_results[k, 2] = np.linalg.norm(Fv)
        mnu_results[k, 3] = i

    k_argmin = np.argmin(mnu_results[:, 2])

    (mu_i, nu_i, err, iter_n) = mnu_results[k_argmin]

    # shift nu_i
    if nu_i > 1:
        nu_i -= 1
    elif nu_i < 0:
        nu_i += 1

    # compute b within s coordinates
    if mu_i < 1 and err < 1e-7:

        # correct twist by flux rope length
        twist = Tfac / gamma_l

        (b_nu, b_s, eps_nu, eps_s) = distorted_bfield(
            mu_i, nu_i, s_i, rho_0, rho_1, tv, n1, n2, k1, k2, delta, twist
        )

        q[:] = b_t * (eps_nu * b_nu + eps_s * b_s)

        # rotate back into s-frame
        _qb = np.array([0, q[0], q[1], q[2]]).astype(s.dtype)
        q[:] = _numba_quaternion_rotate(_qb, q_xs)

    else:
        q[:] = np.array([0, 0, 0])


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(i), (j), (k), (l), (l) -> (i)",
    fastmath=True,
    target="cpu",
)
def distorted_sq_rho(
    s: np.ndarray,
    iparams: np.ndarray,
    sparams: np.ndarray,
    q_sx: np.ndarray,
    q_xs: np.ndarray,
    rho: np.float64,
) -> None:
    (
        _,
        _,
        _,
        _,
        d1au,
        delta,
        _,
        _,
        Tfac,
        _,
        _,
        _,
        _,
        _,
        alpha,
        beta,
        lambda_v,
        epsilon,
        kappa,
    ) = iparams
    (_, rho_0, rho_1, b_t, gamma_l) = sparams

    _s = np.array([0, s[0], s[1], s[2]]).astype(s.dtype)
    xs = _numba_quaternion_rotate(_s, q_sx)

    # evaluate 20 gammas and find closest s coordinate
    s_list = np.linspace(0.1, 0.9, 50)

    s_g_list = np.array(
        [
            np.linalg.norm(rho_0 * gamma(_, alpha, beta, lambda_v, epsilon, kappa) - xs)
            for _ in s_list
        ]
    )

    # check for any NaNs if present
    s_g_list[np.isnan(s_g_list)] = 1e9

    min_arg_s = np.argmin(s_g_list)
    s_i = s_list[min_arg_s]

    s_i_max = s_list[min_arg_s + 1]
    s_i_min = s_list[min_arg_s - 1]

    # basic newton method to find s coordinate
    # typically < 5 steps are required, max 10
    # last_corr must be limited to number of initial guesses for s so that it doesnt overcorrect
    # PS: this algo must not fail or be inaccurate otherewise next step will NOT work
    last_corr = 1 / (len(s_list) + 1) / 2
    sfac = 1

    s_h2 = 5 * s_h

    for i in range(50):
        df_si = np.linalg.norm(
            rho_0 * gamma(s_i, alpha, beta, lambda_v, epsilon, kappa) - xs
        )
        df_si_plus = np.linalg.norm(
            rho_0 * gamma(s_i + s_h2, alpha, beta, lambda_v, epsilon, kappa) - xs
        )
        df_si_minus = np.linalg.norm(
            rho_0 * gamma(s_i - s_h2, alpha, beta, lambda_v, epsilon, kappa) - xs
        )

        d1f = (df_si_plus - df_si_minus) / 2 / s_h2
        d2f = (df_si_plus - 2 * df_si + df_si_minus) / (s_h2**2)

        # break if derivatives become too small
        if np.isnan(d1f / d2f):
            break
        elif np.abs(d1f / d2f) > last_corr:
            # decrease sfac until correction is smaller or hits limit
            while sfac * np.abs(d1f / d2f) > last_corr:
                sfac /= 2

                if sfac * np.abs(d1f / d2f) < 1e-8:
                    break
        elif np.abs(d1f / d2f) < 1e-8:
            break

        s_i = s_i - sfac * d1f / d2f

        # safeguard, reset to opposite side of valid interval to avoid local minima
        if s_i > s_i_max:
            s_i_p = s_i
            s_i = (s_i_max + 3 * s_i_min) / 4

            # narrow boundaries
            s_i_max = s_i_p + sfac * d1f / d2f

            sfac /= 2

        elif s_i < s_i_min:
            s_i_p = s_i
            s_i = (3 * s_i_max + s_i_min) / 4

            # narrow boundaries
            s_i_min = s_i_p + sfac * d1f / d2f

            sfac /= 2
        else:
            last_corr = sfac * np.abs(d1f / d2f)

    # cut off s_i
    if s_i < 0.15 or s_i > 0.85:
        rho = 0.0
        return

    # compute n1, n2 for specific s_i
    gv = rho_0 * gamma(s_i, alpha, beta, lambda_v, epsilon, kappa)
    tv, n1, n2, k1, k2 = get_n_vectors(
        gv, s_i, rho_0, alpha, beta, lambda_v, epsilon, kappa
    )

    dxs = xs - gv

    n1p = np.dot(dxs, n1)
    n2p = np.dot(dxs, n2)

    # solve dxs = Df (n1 cos DOm + n2 sin DOm)
    # this can be rewritten as a 2x2 system of equations
    # Df * cos DOm = n1p, Df * sin DOm = n2p
    # we can thus use newton to find mu/nu
    # first we need good initial estimates, assuming DOm is linear
    # note that this all works very badly if mu_i << 1
    # TODO: improve algorithm for small mu
    Np = 9
    mnu_i_0 = np.empty((Np, 2))

    for i in range(Np):
        # offset initial angle values, this fixes things for some reason (dislikes 0.0/1.0 as initial)
        mnu_i_0[i, 1] = i / Np + 1 / Np / 3
        mnu_i_0[i, 0] = np.linalg.norm(dxs) / Df(
            1, i / Np + 1 / Np / 3, s_i, rho_1, k1, k2, delta
        )

    mnu_results = np.ones((Np, 4))

    if np.all(mnu_i_0[:, 0] > 2):
        rho[:] = np.array([0, 0, 0])
        return
    else:
        pass

    for k in range(Np):
        mnu_i = mnu_i_0[k]

        # dont overcorrect at start
        last_corr = 1
        sfac = 1

        for i in range(20):
            mu_i = mnu_i[0]
            nu_i = mnu_i[1]

            cosDOm = np.cos(DOm(mu_i, nu_i, rho_1, k1, k2))
            sinDOm = np.sin(DOm(mu_i, nu_i, rho_1, k1, k2))

            Fv = np.array(
                [
                    Df(mu_i, nu_i, s_i, rho_1, k1, k2, delta) * cosDOm - n1p,
                    Df(mu_i, nu_i, s_i, rho_1, k1, k2, delta) * sinDOm - n2p,
                ]
            )

            DFv = np.array(
                [
                    [
                        Dfdmu(mu_i, nu_i, s_i, rho_1, k1, k2, delta) * cosDOm
                        - Df(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
                        * sinDOm
                        * DOmdmu(mu_i, nu_i, rho_1, k1, k2),
                        Dfdnu(mu_i, nu_i, s_i, rho_1, k1, k2, delta) * cosDOm
                        - Df(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
                        * sinDOm
                        * DOmdnu(mu_i, nu_i, rho_1, k1, k2),
                    ],
                    [
                        Dfdmu(mu_i, nu_i, s_i, rho_1, k1, k2, delta) * sinDOm
                        + Df(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
                        * cosDOm
                        * DOmdmu(mu_i, nu_i, rho_1, k1, k2),
                        Dfdnu(mu_i, nu_i, s_i, rho_1, k1, k2, delta) * sinDOm
                        + Df(mu_i, nu_i, s_i, rho_1, k1, k2, delta)
                        * cosDOm
                        * DOmdnu(mu_i, nu_i, rho_1, k1, k2),
                    ],
                ]
            )

            if np.any(np.isnan(DFv)) or np.linalg.det(DFv) < 1e-10:
                break

            corr = np.dot(np.linalg.inv(DFv), Fv)
            corr_n = np.linalg.norm(corr)

            # break if derivatives become too small
            # allow for larger re-steps
            if np.isnan(np.linalg.det(DFv)) or np.isnan(corr_n):
                break
            elif mnu_i[0] - sfac * corr[0] < 0:
                while mnu_i[0] - sfac * corr[0] < 0:
                    sfac /= 2
            elif sfac * corr_n > last_corr:
                while sfac * corr_n > last_corr:
                    sfac /= 2

                    if sfac * corr_n < 1e-16:
                        break

            elif corr_n < 1e-16:
                break

            if sfac < 0.025:
                break

            mnu_i = mnu_i - sfac * corr

            last_corr = sfac * np.linalg.norm(corr)

        mnu_results[k, 0] = mnu_i[0]
        mnu_results[k, 1] = mnu_i[1]
        mnu_results[k, 2] = np.linalg.norm(Fv)
        mnu_results[k, 3] = i

    k_argmin = np.argmin(mnu_results[:, 2])

    (mu_i, nu_i, err, iter_n) = mnu_results[k_argmin]

    # shift nu_i
    if nu_i > 1:
        nu_i -= 1
    elif nu_i < 0:
        nu_i += 1

    # compute b within s coordinates
    if mu_i < 1 and err < 1e-7:

        # correct twist by flux rope length
        twist = Tfac / gamma_l

        rho[0] = (1 - np.sqrt(np.sqrt((np.abs(s_i - 0.5))))) * distorted_rhofield(
            mu_i, nu_i, s_i, rho_0, rho_1, tv, n1, n2, k1, k2, delta, twist
        )

        if mu_i > 1:
            rho[0] = (2 * (1.5 - mu_i)) ** 2 * distorted_rhofield(
                1, nu_i, s_i, rho_0, rho_1, tv, n1, n2, k1, k2, delta, twist
            )
    else:
        rho[:] = 0.0