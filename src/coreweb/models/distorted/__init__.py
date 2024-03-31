# -*- coding: utf-8 -*-

import datetime
import json
import os
from itertools import product
from typing import Union

import numpy as np
from heliosat.util import sanitize_dt
from numba import guvectorize

import py3dcore

from ...model import SimulationBlackBox
from .distorted_shape import dgds, distorted_qs, distorted_sq_gh

class DistortedModel(SimulationBlackBox):
    """Implements the distorted model.
    Model Parameters
    ================
        For this specific model there are a total of 14 initial parameters which are as follows:
        0: t_i          time offset
        1: lon          longitude
        2: lat          latitude
        3: inc          inclination
        4: dia          cross section diameter at 1 AU
        5: delta        cross section aspect ratio
        6: r0           initial cme radius
        7: v0           initial cme velocity
        8: T            T factor (related to the twist)
        9: n_a          expansion rate
        10: n_b         magnetic field decay rate
        11: b           magnetic field strength at center at 1AU
        12: bg_d        solar wind background drag coefficient
        13: bg_v        solar wind background speed

        14: alpha
        15: beta
        16: lambda
        17: epsilon
        18: kappa

        There are 5 state parameters which are as follows:
        0: v_t          current velocity
        1: rho_0        torus major radius
        2: rho_1        torus minor radius
        3: b_t          magnetic field strength at center
        4: gamma_l      axis length
    """

    mag_model: str
    shape_model: str

    def __init__(
        self,
        dt_0: Union[str, datetime.datetime],
        ensemble_size: int,
        iparams: dict = {},
        shape_model: str = "distorted",
        mag_model: str = "gh",
        dtype: type = np.float32,
    ) -> None:
        with open(
            os.path.join(
                os.path.dirname(py3dcore.__file__), "models/distorted/parameters.json"
            )
        ) as fh:
            iparams_dict = json.load(fh)

        for k, v in iparams.items():
            if k in iparams_dict:
                iparams_dict[k].update(v)
            else:
                raise KeyError('key "%s" not defined in parameters.json', k)

        super(DistortedModel, self).__init__(
            dt_0,
            iparams=iparams_dict,
            sparams=5,
            ensemble_size=ensemble_size,
            dtype=dtype,
        )

        self.mag_model = mag_model
        self.shape_model = shape_model

    def propagator(self, dt_to: Union[str, datetime.datetime]) -> None:
        _numba_propagator(
            self.dtype(sanitize_dt(dt_to).timestamp() - self.dt_0.timestamp()),
            self.iparams_arr,
            self.sparams_arr,
            self.sparams_arr,
        )

        self.dt_t = dt_to

    def simulator_mag(self, pos: np.ndarray, out: np.ndarray) -> None:
        if self.shape_model == "distorted":
            distorted_sq_gh(
                pos, self.iparams_arr, self.sparams_arr, self.qs_sx, self.qs_xs, out
            )
        else:
            raise NotImplementedError

    def visualize_shape(
        self, iparam_index: int = 0, resolution: int = 10
    ) -> np.ndarray:
        r = np.array([1.0], dtype=self.dtype)
        u = np.linspace(0, 1, resolution)
        v = np.linspace(0.1, 0.9, resolution)
        c = len(u)
        # combination of wireframe points in (q)
        arr = np.array(list(product(r, u, v)), dtype=self.dtype).reshape(c**2, 3)
        arr2 = np.zeros_like(arr)

        for i in range(0, len(arr)):
            distorted_qs(
                arr[i],
                self.iparams_arr[iparam_index],
                self.sparams_arr[iparam_index],
                self.qs_xs[iparam_index],
                arr2[i],
            )

        return arr2.reshape((c, c, 3))


@guvectorize(
    [
        "void(float32, float32[:], float32[:], float32[:])",
        "void(float64, float64[:], float64[:], float64[:])",
    ],
    "(), (j), (k) -> (k)",
    target="parallel",
)
def _numba_propagator(
    t_offset: float, iparams: np.ndarray, _: np.ndarray, sparams: np.ndarray
) -> None:
    (
        t_i,
        _,
        _,
        _,
        d,
        _,
        r,
        v,
        _,
        n_a,
        n_b,
        b_i,
        bg_d,
        bg_v,
        alpha,
        beta,
        lambda_v,
        epsilon,
        kappa,
    ) = iparams

    # rescale parameters
    bg_d = bg_d * 1e-7
    r = r * 695510

    dt = t_offset - t_i
    dv = v - bg_v

    bg_sgn = int(-1 + 2 * int(v > bg_v))

    rt = (bg_sgn / bg_d * np.log1p(bg_sgn * bg_d * dv * dt) + bg_v * dt + r) / 1.496e8

    vt = dv / (1 + bg_sgn * bg_d * dv * dt) + bg_v

    rho_1 = d * (rt**n_a) / 2
    rho_0 = (rt - rho_1) / 2
    b_t = b_i * (2 * rho_0) ** (-n_b)

    # compute flux rope length, rough estimate based on 20 points
    Nsr = 20
    s_r = np.linspace(0.05, 0.95, Nsr)

    gamma_l = 0

    for s_v in s_r:
        dv = rho_0 * np.linalg.norm((dgds(s_v, alpha, beta, lambda_v, epsilon, kappa)))

        if not np.isnan(dv):
            gamma_l += dv

    gamma_l /= Nsr

    sparams[0] = vt
    sparams[1] = rho_0
    sparams[2] = rho_1
    sparams[3] = b_t
    sparams[4] = gamma_l