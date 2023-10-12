# -*- coding: utf-8 -*-

"""abc_smc.py

Implements an ABC-SMC algorithm.
"""

import datetime
import faulthandler
import logging
import multiprocessing
import os
import pickle
import time
from typing import Any, Optional, Sequence, Tuple, Union

import heliosat
import numpy as np
from heliosat.util import sanitize_dt

logger = logging.getLogger(__name__)

import streamlit as st

from ..model import SimulationBlackBox, set_random_seed
from .data import FittingData
from .method import BaseMethod
from ..app.plotting import plot_fitting_process

def starmap(func, args):
    return [func(*_) for _ in args]

class streamlit_ABC_SMC(BaseMethod):
    
    """
    Fits a model to observations using "Approximate Bayesian Computation Monte Carlo"
    
    Arguments:
        *args:      Any
        **kwargs:   Any
    
    Returns:
        None
    
    Functions:
        initialize
        run
        abc_smc_worker      
    """
    
    
    iter_i: int

    hist_eps: list
    hist_eps_dim: int
    hist_time: list

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(streamlit_ABC_SMC, self).__init__(*args, **kwargs)

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        super(streamlit_ABC_SMC, self).initialize(*args, **kwargs)
        
        """
        Initializes the iterations, epsilon values and time.
        Sets the following properties for self:
        iter_i       keeps track of iterations
        hist_eps     keeps track of epsilon values
        hist_time    keeps track of time
    
        Arguments:
            *args:      Any
            **kwargs:   Any
    
        Returns:
            None
        """

        self.iter_i = 0 # keeps track of iterations
        self.hist_eps = [] # keeps track of epsilon values
        self.hist_time = [] # keeps track of time

    def run(
        self, st, epsgoal=0.25, iter_min=10, iter_max=15, ensemble_size=512, reference_frame="HEEQ", **kwargs: Any
    ) -> None:
        
        """
        Runs the fitting process.
        Sets the following properties for self:
            hist_eps_dim             Dimension of each eps (Number of observers)
    
        Arguments:
            epsgoal          0.25     Epsilon to be reached during optimization
            iter_min         10      Minimum iterations before epsgoal is checked
            iter_max         15      Maximum iterations until fitting is interrupted
            ensemble_size    512     Number of particles to be accepted 
            reference_frame  "HEEQ"  reference frame of coordinate system
            *args            Any
            **kwargs         Any
    
        Returns:
            None
        """
        
        # read kwargs
        balanced_iterations = kwargs.pop("balanced_iterations", 3) # number of iterations before eps gets set to eps_max to balance for multiple observersx
        data_kwargs = kwargs.pop("data_kwargs", {}) # kwargs to be used for the FittingData, usually not given
        eps_quantile = kwargs.pop("eps_quantile", 0.25) # which quantile to use for the new eps
        kernel_mode = kwargs.pop("kernel_mode", "cm") # kernel mode for perturbing the iparams - covariance matrix
        output = kwargs.get("output", None) # If output is set, results are saved to a file
        
        if output is not None:
            output = "output/" + output
        
        random_seed = kwargs.pop("random_seed", 42) # set random seed to ensure reproducible results
        summary_type = kwargs.pop("summary_statistic", "norm_rmse") # summary statistic used to measure the error of a fit
        time_offsets = kwargs.pop("time_offsets", [0]) # value used to correct arrival times at observers

        workers = kwargs.pop("workers", multiprocessing.cpu_count()-1) # number of workers 
        jobs = kwargs.pop("jobs", workers) # number of jobs
        use_multiprocessing = kwargs.pop("use_multiprocessing", False) # Whether to use multiprocessing 

        mpool = multiprocessing.Pool(processes=workers) # initialize Pool for multiprocessing
        
        # Fitting data comes from the module fitter.base.py 
        
        data_obj = FittingData(self.observers, reference_frame) 
            
        data_obj.generate_noise(
            kwargs.get("noise_model", "psd"),
            kwargs.get("sampling_freq", 300),
            **data_kwargs
        ) # noise is generated for the Fitting Data Object, function also comes from the module fitter.base.py 


        kill_flag = False
        pcount = 0
        timer_iter = None
        
        # Here the actual fitting loop starts

        try:
            for iter_i in range(self.iter_i, iter_max):
                
                # We first check if the minimum number of 
                # iterations is reached.If yes, we check if
                # the target value for epsilon "epsgoal" is reached.
                reached = False
                
                if iter_i >= iter_min:
                    reached = True
                    if self.hist_eps[-1] < epsgoal:
                        plot_fitting_process(st, reached)
                        kill_flag = True
                        break
                        
                st.session_state.current_iter = iter_i

                timer_iter = time.time()
                
                # correct observer arrival times

                if iter_i >= len(time_offsets):
                    _time_offset = time_offsets[-1]
                else:
                    _time_offset = time_offsets[iter_i]
                    
                # next, the data_kwargs, which are {} by default,
                # are used to set self.data_dt, self.data_b, self.data_o, 
                # self.data_m and self.data_l 

                data_obj.generate_data(_time_offset, **data_kwargs)
                
                # only gets executed during first iteration

                if len(self.hist_eps) == 0:
                    eps_init = data_obj.sumstat(
                        [np.zeros((1, 3))] * len(data_obj.data_b), use_mask=False
                    )[0]
                    # returns summary statistic for a vector of zeroes for each observer
                    self.hist_eps = [eps_init, eps_init * 0.98]
                    #hist_eps gets set to the eps_init and 98% of it
                    
                    self.hist_eps_dim = len(eps_init) # number of observers

                    
                    st.session_state.currenteps = self.hist_eps[-1]
                    st.session_state.epsdiff = 0
                    
                    # model kwargs are stored in a dictionary

                    model_obj_kwargs = dict(self.model_kwargs)  #
                    model_obj_kwargs["ensemble_size"] = ensemble_size
                    model_obj = self.model(self.dt_0, **model_obj_kwargs) # model gets initialized

                sub_iter_i = 0 # keeps track of subprocesses 


                _random_seed = random_seed + 100000 * iter_i # set random seed to ensure reproducible results
                # worker_args get stored
            
                worker_args = (
                    iter_i,
                    self.dt_0,
                    self.model,
                    self.model_kwargs,
                    model_obj.iparams_arr,
                    model_obj.iparams_weight,
                    model_obj.iparams_kernel_decomp,
                    data_obj,
                    summary_type,
                    self.hist_eps[-1],
                    kernel_mode,
                )
                
                plot_fitting_process(st)
                
                if use_multiprocessing == True:
                    _results = mpool.starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(jobs)]) # starmap returns a function for all given arguments
                else:
                    _results = starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(jobs)]) # starmap returns a function for all given arguments
                
                
                
                # the total number of runs depends on the ensemble size set in the model kwargs and the number of jobs
                total_runs = jobs * int(self.model_kwargs["ensemble_size"])  #

                # repeat until enough samples are collected
                while True:
                    pcounts = [len(r[1]) for r in _results] # number of particles collected per job 
                    _pcount = sum(pcounts) # number of particles collected in total
                    dt_pcount = _pcount - pcount # number of particles collected in current iteration
                    pcount = _pcount # particle count gets updated
                    
                    # iparams and according errors get stored in array
                    particles_temp = np.zeros(
                        (pcount, model_obj.iparams_arr.shape[1]), model_obj.dtype
                    )
                    epses_temp = np.zeros((pcount, self.hist_eps_dim), model_obj.dtype)

                    for i in range(0, len(_results)):
                        particles_temp[
                            sum(pcounts[:i]) : sum(pcounts[: i + 1])
                        ] = _results[i][0] # results of current iteration are stored
                        epses_temp[sum(pcounts[:i]) : sum(pcounts[: i + 1])] = _results[
                            i
                        ][1] # errors of current iteration are stored

                    progress_text = str(pcount) + "/" + str(ensemble_size)
                        
                        
                    
                    
                    if pcount > ensemble_size:
                        percentage = 100 
                    else:
                        percentage = (pcount / ensemble_size) * 100
                    
                    st.session_state.progressbar.progress(int(percentage), text=progress_text)

                    if pcount > ensemble_size:
                        break
                        
                    
                    # if ensemble size isn't reached, continue
                    # random seed gets updated

                    _random_seed = (
                        random_seed + 100000 * iter_i + 1000 * (sub_iter_i + 1)
                    )

                    if use_multiprocessing == True:
                        _results_ext = mpool.starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(jobs)]) # runs the abc_smc
                    else:
                        _results_ext = starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(jobs)]) # runs the abc_smc
                        
                    _results.extend(_results_ext) #results get appended to _results

                    sub_iter_i += 1
                    
                    # keep track of total number of runs
                    
                    total_runs += jobs * int(self.model_kwargs["ensemble_size"])  #

                    if pcount == 0:
                        logger.warning("no hits, aborting")
                        kill_flag = True
                        break

                if kill_flag:
                    break

                if pcount > ensemble_size: # no additional particles are kept
                    particles_temp = particles_temp[:ensemble_size]
                    
                 # if we're in the first iteration, the weights and kernels have to be initialized. Otherwise, they're updated. 

                if iter_i == 0:
                    model_obj.update_iparams(
                        particles_temp,
                        update_weights_kernels=False,
                        kernel_mode=kernel_mode,
                    ) # replace self.iparams_arr by particles_temp
                    model_obj.iparams_weight = (
                        np.ones((ensemble_size,), dtype=model_obj.dtype) / ensemble_size
                    )
                    model_obj.update_kernels(kernel_mode=kernel_mode)
                else:
                    model_obj.update_iparams(
                        particles_temp,
                        update_weights_kernels=True,
                        kernel_mode=kernel_mode,
                    )

                if isinstance(eps_quantile, float):
                    new_eps = np.quantile(epses_temp, eps_quantile, axis=0)
                    
                    # after the first couple of iterations, the new eps gets simply set to the its maximum value instead of choosing a different eps for each observer

                    if balanced_iterations > iter_i:
                        new_eps[:] = np.max(new_eps)

                    self.hist_eps.append(new_eps)
                elif isinstance(eps_quantile, list) or isinstance(
                    eps_quantile, np.ndarray
                ):
                    eps_quantile_eff = eps_quantile ** (1 / self.hist_eps_dim)  #
                    _k = len(eps_quantile_eff)  #

                    new_eps = np.array(
                        [
                            np.quantile(epses_temp, eps_quantile_eff[i], axis=0)[i]
                            for i in range(_k)
                        ]
                    )

                    self.hist_eps.append(new_eps)  #

                
                st.session_state.currenteps = self.hist_eps[-1]
                st.session_state.epsdiff = self.hist_eps[-1] - self.hist_eps[-2]
                    
                

                self.hist_time.append(time.time() - timer_iter)

                self.iter_i = iter_i + 1  # iter_i gets updated
                
                # save output to file 
                if output:
                    output_file = os.path.join(
                        output, "{0:02d}.pickle".format(self.iter_i - 1)
                    )

                    extra_args = {
                        "model_obj": model_obj,
                        "data_obj": data_obj,
                        "epses": epses_temp,
                    }

                    self.save(output_file, **extra_args)
        finally:
            pass
            # mpool.close()
            # mpool.join() # close the multiprocessing


def abc_smc_worker(*args: Any) -> Tuple[np.ndarray, np.ndarray]:
    (
        iter_i,
        dt_0,
        model_class,
        model_kwargs,
        old_iparams,
        old_weights,
        old_kernels,
        data_obj,
        summary_type,
        eps_value,
        kernel_mode,
        random_seed,
    ) = args
    
    """
    Worker for ABC-SMC 
    
    Arguments:
        iter_i              subcounter
        dt_0                launch time
        model_class         which model to use
        model_kwargs        kwargs to use for the model
        old_iparams         parameters from previous iteration
        old_weights         weights from previous iteration
        old_kernels         kernels from previous iteration
        data_obj            Fitting Data
        summary_type        summary statistic type
        eps_value           epsilon value
        kernel_mode         kernel mode to be used for perturbation (usually covariance matrix)
        random seed         set random seed to ensure reproducible results
        
    Returns:
        result              accepted iparams
        error[accept_mask]  error of accepted iparams
    """
    
    logger = logging.getLogger(__name__)

    if iter_i == 0:
        model_obj = model_class(dt_0, **model_kwargs) # model gets initialized with model kwargs
        model_obj.generator(random_seed=random_seed) # generator is run to create the iparams array and the quaternions
    else:
        set_random_seed(random_seed) # random seed gets set
        model_obj = model_class(dt_0, **model_kwargs) # model gets initialized with model kwargs
        model_obj.perturb_iparams(
            old_iparams, old_weights, old_kernels, kernel_mode=kernel_mode
        ) #iparams are perturbed


    # TODO: sort data_dt by time

    # sort
    sort_index = np.argsort([_.timestamp() for _ in data_obj.data_dt]) # the index is sorted by time (sometimes this isn't the case by default)

    # generate synthetic profiles
    try:
        profiles = np.array(
            model_obj.simulator(
                np.array(data_obj.data_dt)[sort_index],
                np.array(data_obj.data_o)[sort_index],
            )[0],
            dtype=model_obj.dtype,
        )
    except IndexError:
        raise IndexError('Data not in file, try to set different times!')

    # resort profiles
    sort_index_rev = np.argsort(sort_index)
    profiles = profiles[sort_index_rev]

    # TODO: revert profiles to proper order after data_dt resorting

    # generate synthetic noise
    profiles = data_obj.add_noise(profiles)
    
    #calculate the error for each ensemble member and set the eps to inf if magnetic field at reference points is nonzero 
    error = data_obj.sumstat(profiles, stype=summary_type, use_mask=True)
    
    # accept all ensemble members for which the error is lower than the eps
    accept_mask = np.all(error < eps_value, axis=1)

    result = model_obj.iparams_arr[accept_mask]
    # print("WORKER DONE", result.shape, error[accept_mask].shape)
    return result, error[accept_mask]
