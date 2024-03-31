# -*- coding: utf-8 -*-

from typing import Tuple

import numba
import numpy as np
from numba import guvectorize

from coreweb.rotqs import _numba_quaternion_rotate


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(i), (j), (k), (l) -> (i)",
    target="parallel",
)
def thin_torus_qs(
    q: np.ndarray,
    iparams: np.ndarray,
    sparams: np.ndarray,
    q_xs: np.ndarray,
    s: np.ndarray,
) -> None:
    (q0, q1, q2) = q # extract the three coordinates

    delta = iparams[5]

    (_, rho_0, rho_1, _) = sparams

    x = np.array(
        [
            0,
            -(rho_0 + q0 * rho_1 * np.sin(q1 / 2) ** 2 * np.cos(q2)) * np.cos(q1)
            + rho_0,
            (rho_0 + q0 * rho_1 * np.sin(q1 / 2) ** 2 * np.cos(q2)) * np.sin(q1),
            q0 * rho_1 * np.sin(q1 / 2) ** 2 * np.sin(q2) * delta,
        ]
    )

    s[:] = _numba_quaternion_rotate(x, q_xs)  # rotate from x to s


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(i), (j), (k), (l) -> (i)",
    target="parallel",
)
def thin_torus_sq(
    s: np.ndarray,
    iparams: np.ndarray,
    sparams: np.ndarray,
    q_sx: np.ndarray,
    q: np.ndarray,
) -> None:
    """
    Simulates the thin torus shape 
        
    Arguments:
        s         trajectory of observer
        iparams   initial parameters array
        sparams   state parameters array
        q_sx      quaternions to rotate from s to x 
        q         array with zeros of shape (len(iparams_arr),3)
        
    Returns:
        None      
    """
        
    delta = iparams[5]

    (_, rho_0, rho_1, _) = sparams

    _s = np.array([0, s[0], s[1], s[2]]).astype(s.dtype) #create 4vector

    xs = _numba_quaternion_rotate(_s, q_sx) #rotate from s to x

    (x0, x1, x2) = xs # extract three coordinates

    if x0 == rho_0:
        if x1 >= 0:
            psi = np.pi / 2
        else:
            psi = 3 * np.pi / 2
    else:
        psi = np.arctan2(-x1, x0 - rho_0) + np.pi

    g1 = np.cos(psi) ** 2 + np.sin(psi) ** 2
    rd = np.sqrt(((rho_0 - x0) ** 2 + x1**2) / g1) - rho_0

    if rd == 0:
        if x2 >= 0:
            phi = np.pi / 2
        else:
            phi = 3 * np.pi / 2
    else:
        if rd > 0:
            phi = np.arctan(x2 / rd / delta)
        else:
            phi = -np.pi + np.arctan(x2 / rd / delta)

        if phi < 0:
            phi += 2 * np.pi

    if phi == np.pi / 2 or phi == 3 * np.pi / 2:
        r = x2 / delta / rho_1 / np.sin(phi) / np.sin(psi / 2) ** 2
    else:
        r = np.abs(rd / np.cos(phi) / np.sin(psi / 2) ** 2 / rho_1)
        
    #store new coordinates in q

    q[0] = r
    q[1] = psi
    q[2] = phi


@numba.njit
def thin_torus_jacobian(
    q0: float, q1: float, q2: float, rho_0: float, rho_1: float, delta: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = 1

    dr = np.array(
        [
            -rho_1 * np.sin(q1 / 2) ** 2 * np.cos(q2) * np.cos(q1),
            w * rho_1 * np.sin(q1 / 2) ** 2 * np.cos(q2) * np.sin(q1),
            rho_1 * np.sin(q1 / 2) ** 2 * np.sin(q2) * delta,
        ]
    )

    dpsi = np.array(
        [
            rho_0 * np.sin(q1)
            + q0 * rho_1 * np.sin(q1 / 2) ** 2 * np.cos(q2) * np.sin(q1)
            - q0 * rho_1 * np.cos(q1 / 2) * np.sin(q1 / 2) * np.cos(q2) * np.cos(q1),
            w
            * (
                rho_0 * np.cos(q1)
                + q0 * rho_1 * np.sin(q1 / 2) ** 2 * np.cos(q2) * np.cos(q1)
                + q0 * rho_1 * np.cos(q1 / 2) * np.sin(q1 / 2) * np.cos(q2) * np.sin(q1)
            ),
            q0 * rho_1 * delta * np.cos(q1 / 2) * np.sin(q1 / 2) * np.sin(q2),
        ]
    )

    dphi = np.array(
        [
            q0 * rho_1 * np.sin(q1 / 2) ** 2 * np.sin(q2) * np.cos(q1),
            -w * q0 * rho_1 * np.sin(q1 / 2) ** 2 * np.sin(q2) * np.sin(q1),
            q0 * rho_1 * np.sin(q1 / 2) ** 2 * np.cos(q2) * delta,
        ]
    )

    return dr, dpsi, dphi


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(i), (j), (k), (l) -> (i)",
    target="parallel",
)
def thin_torus_gh(
    q: np.ndarray,
    iparams: np.ndarray,
    sparams: np.ndarray,
    q_xs: np.ndarray,
    b: np.ndarray,
) -> None:
    
    """
    Implements the gold-hoyle model.
    
    Arguments:
        q              array of shape (len(self.iparams_arr),3) containing zeros 
        iparams        initial parameter array
        sparams        state parameter array
        q_xs           quaternion to rotate from x to s 
        b              out mag field
    
    Returns:
        None
    """
    
    bsnp = np.empty((3,))

    (q0, q1, q2) = (q[0], q[1], q[2])

    if q0 <= 1:
        (t_i, _, _, _, _, delta, _, _, Tfac, _, _, _, _, _) = iparams
        (_, rho_0, rho_1, b_t) = sparams

        # get normal vectors
        (dr, dpsi, dphi) = thin_torus_jacobian(q0, q1, q2, rho_0, rho_1, delta)

        # unit normal vectors
        dr = dr / np.linalg.norm(dr)
        dpsi_norm = np.linalg.norm(dpsi)
        dpsi = dpsi / dpsi_norm
        dphi_norm = np.linalg.norm(dphi)
        dphi = dphi / dphi_norm

        br = 0

        fluxfactor = 1 / np.sin(q1 / 2) ** 2

        # ellipse circumference
        h = (delta - 1) ** 2 / (1 + delta) ** 2
        #Efac = np.pi * (1 + delta) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))

        t = Tfac * rho_1 / rho_0 / 2 / np.pi * np.sin(q1 / 2) ** 2 #/ Efac

        denom = 1 + t**2 * q0**2
        bpsi = b_t / denom * fluxfactor
        bphi = b_t * t * q0 / denom / (1 + q0 * rho_1 / rho_0 * np.cos(q2)) * fluxfactor

        # magnetic field in (x)
        bsnp[0] = dr[0] * br + dpsi[0] * bpsi + dphi[0] * bphi
        bsnp[1] = dr[1] * br + dpsi[1] * bpsi + dphi[1] * bphi
        bsnp[2] = dr[2] * br + dpsi[2] * bpsi + dphi[2] * bphi

        # magnetic field in (s)
        bss = _numba_quaternion_rotate(np.array([0, bsnp[0], bsnp[1], bsnp[2]]), q_xs)

        b[0] = bss[0]
        b[1] = bss[1]
        b[2] = bss[2]
    else:
        b[0] = 0
        b[1] = 0
        b[2] = 0
