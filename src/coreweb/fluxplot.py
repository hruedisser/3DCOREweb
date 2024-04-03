import os

import numpy as np
import pickle as p
import pandas as pds
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('paper')

import datetime
from datetime import timedelta
import coreweb
from .methods.method import BaseMethod

from coreweb.dashcore.utils.utils import generate_ensemble

import heliosat

from coreweb.models.toroidal import thin_torus_gh, thin_torus_qs, thin_torus_sq

from .rotqs import generate_quaternions

import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LightSource

import logging

logger = logging.getLogger(__name__)

# settings for plots

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.it'] = 'DejaVu Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'DejaVu Sans:bold'
#matplotlib.rcParams['mathtext.fontset'] = 'stix' 
#matplotlib.rcParams['font.family'] = 'STIXGeneral'

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure title

##################################


def get_params(filepath, give_mineps=False):
    
    """ Gets params from file. """
    ######### get parameters (mean +/- std)
    
    fit_res = coreweb.ABC_SMC(filepath)
    fit_res_mean = np.mean(fit_res.model_obj.iparams_arr, axis=0)
    fit_res_std = np.std(fit_res.model_obj.iparams_arr, axis=0)

    # calculate the twist from t-factor and circumference of ellipse Efac - twist = t-factor/Efac
    delta = fit_res.model_obj.iparams_arr[:,5]
    #print(delta)
    # formula from 3DCORE
    h = (delta - 1) ** 2 / (1 + delta) ** 2
    Efac = np.pi * (1 + delta) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))

    # twist = fit_res.model_obj.iparams_arr[:,8]/Efac
    twist = fit_res.model_obj.iparams_arr[:,8]/Efac
    #print(twist)

    twist_mean = np.mean(twist, axis=0)
    twist_std = np.std(twist, axis=0)
    print(twist_mean, twist_std)
    print("\t{}: \t\t{:.02f} +/- {:.02f}".format('twist', twist_mean, twist_std))

    keys = list(fit_res.model_obj.iparams.keys()) # names of the parameters 
    fit_res_mean_t = np.append(fit_res_mean,twist_mean)
    fit_res_std_t = np.append(fit_res_std,twist_std)

    print("Results :\n")
    for i in range(1, len(keys)):
        print("\t{}: \t\t{:.02f} +/- {:.02f}".format(keys[i], fit_res_mean[i], fit_res_std[i]))
     
    #########
    
    ######### get parameters for run with min(eps)
    # read from pickle file
    file = open(filepath, "rb")
    data = p.load(file)
    file.close()
    
    model_objt = data["model_obj"]
    maxiter = model_objt.ensemble_size-1

    # get index ip for run with minimum eps    
    epses_t = data["epses"]
    ip = np.argmin(epses_t[0:maxiter])    
    
    # get parameters (stored in iparams_arr) for the run with minimum eps
    
    iparams_arrt = model_objt.iparams_arr
    
    resparams = iparams_arrt[ip] # parameters with run for minimum eps

    if give_mineps == True:
        logger.info("Retrieved the following parameters for the run with minimum epsilon:")
        logger.info(" --epsilon {}".format(np.array2string(epses_t[ip], precision=5)[1:-1])) # a bit of a hacky approach to get rid of the brackets around the numpy array epses_t
    
        for i in range(1, len(keys)):
            logger.info(" --{} {:.2f}".format(keys[i], mineps_params[i]))

    ## return mineps_params, fit_res_mean_t, fit_res_std_t, ip, keys, iparams_arrt_t, np.array2string(epses_t[ip], precision=5)[1:-1]
    return mineps_params, fit_res_mean_t, fit_res_std_t, ip, keys, iparams_arrt, np.array2string(epses_t[ip], precision=5)[1:-1]


def scatterparams(df):
    
    ''' returns scatterplots'''
    
    # drop first column
    df.drop(df.columns[[0]], axis=1, inplace=True)

    # rename columns
    df.columns = ['lon', 'lat', 'inc', 'D1AU', 'delta', 'launch radius', 'launch speed', 't factor', 'expansion rate', 'B decay rate', 'B1AU', 'gamma', 'vsw']

    g = sns.pairplot(df, 
                     corner=True,
                     plot_kws=dict(marker="+", linewidth=1)
                    )
    g.map_lower(sns.kdeplot, levels=[0.05, 0.32], color=".2") #  levels are 2-sigma and 1-sigma contours
    g.savefig(path[:-7] + 'scatter_plot_matrix.pdf')
    plt.show()
    

def scatterparams_small(df):
    
    ''' returns scatterplots'''
    
    # drop first column
    df.drop(df.columns[[0,8,9,10,12]], axis=1, inplace=True)

    # rename columns
   # df.columns = ['lon', 'lat', 'inc', 'D1AU', 'delta', 'launch radius', 'launch speed', 't factor', 'expansion rate', 'B decay rate', 'B1AU', 'gamma', 'vsw']
    # df.columns = ['lon', 'lat', 'inc', 'D1AU', 'delta', 'launch radius', 'launch speed', 'B1AU', 'vsw', 'twist']
    df.columns = ['lon', 'lat', 'inc', 'D1AU', 'delta', 'launch radius', 'launch speed', 'B1AU', 'vsw']

    g = sns.pairplot(df, 
                     corner=True,
                     plot_kws=dict(marker="+", linewidth=1)
                    )
    g.map_lower(sns.kdeplot, levels=[0.05, 0.32], color=".2") #  levels are 2-sigma and 1-sigma contours
    g.savefig(path[:-7] + 'scatter_plot_matrix_small.pdf')
    plt.show()


def fullinsitu(observer, t_fit=None, launchtime=None, start=None, end=None, t_s=None, t_e=None, filepath=None, ref_frame=None, save_fig=True, best=True, mean=False, ensemble=True, legend=True, max_index=512, title=True, 
               fit_points=True, prediction=False,  graph=None, row=None):
    
    """
    Plots the synthetic insitu data plus the measured insitu data and ensemble fit.
    """
    
    if start == None:
        start = t_fit[0]

    if end == None:
        end = t_fit[-1]
    
    print(start, end)


    t = graph['t_data']

    # in situ data

    if ref_frame == "HEEQ":
        b = graph['b_data_HEEQ']
    else:
        b = graph['b_data_RTN']
    
    #observer_obj = getattr(heliosat, observer)() # get observer obj
    #logger.info("Using HelioSat to retrieve observer data")
    #t, b = observer_obj.get([start, end], "mag", reference_frame=ref_frame, as_endpoints=True, return_datetimes=True)
    #print(len(t))
    #t = []
    #t = [datetime.datetime.fromtimestamp(dt[i]) for i in range(len(dt))] # .strftime('%Y-%m-%d %H:%M:%S.%f')
    #pos = observer_obj.trajectory(t, reference_frame=ref_frame, smoothing="gaussian")
    #print(pos)
    
    if best == True:
        iparams = get_iparams_live(*extract_row(row))
        print(iparams)
        model_obj = coreweb.ToroidalModel(launchtime, **iparams) # model gets initialized
        model_obj.generator()  

        outa = np.array(model_obj.simulator(graph['t_data'], graph['pos_data']), dtype=object)

        outa = np.squeeze(outa[0])
        if ref_frame != "HEEQ":
            x,y,z = hc.separate_components(graph['pos_data'])
            print(x,y,z)
            rtn_bx, rtn_by, rtn_bz = hc.convert_HEEQ_to_RTN_mag(x,y,z, outa[:, 0], outa[:, 1], outa[:, 2])
            outa[:, 0],outa[:, 1],outa[:, 2] = rtn_bx, rtn_by, rtn_bz
                        
            
        outa[outa==0] = np.nan

    if mean == True:
        model_obj = returnmodel(filepath)
        outam = np.squeeze(np.array(model_obj.simulator(t, pos))[0])       
        outam[outam==0] = np.nan
        print(len(outam)) 


             
    # get ensemble_data
    if ensemble == True:
        # read from pickle file
        file = open(filepath, "rb")
        data = p.load(file)
        file.close()
        
        ensemble_filepath = filepath.split('.')[0] + '_ensembles.pickle'
        with open(ensemble_filepath, 'rb') as ensemble_file:
            ensemble_data = p.load(ensemble_file)  
        if ref_frame == 'HEEQ':
            ed = ensemble_data['ensemble_HEEQ']
        else:
            ed = ensemble_data['ensemble_RTN']    
        #ed = generate_ensemble(filepath, t, reference_frame=ref_frame, reference_frame_to=ref_frame, max_index=max_index)
        #ed = generate_ensemble(filepath, t, pos, reference_frame=ref_frame, reference_frame_to=ref_frame, max_index=max_index)
    
    lw_insitu = 2  # linewidth for plotting the in situ data
    lw_best = 3  # linewidth for plotting the min(eps) run
    lw_mean = 3  # linewidth for plotting the mean run
    lw_fitp = 2  # linewidth for plotting the lines where fitting points
    
    if observer == 'SOLO':
        obs_title = 'Solar Orbiter'

    if observer == 'PSP':
        obs_title = 'Parker Solar Probe'
        
    if observer == 'WIND':
        obs_title = 'Wind'    

    # colours 
    c0 = "xkcd:black"
    c1 = "xkcd:magenta"
    c2 = "xkcd:orange"
    c3 = "xkcd:azure"    

    plt.figure(figsize=(20, 10))
    
    if title == True:
        plt.title("3DCORE fitting result - "+obs_title)
    
    if ensemble == True:
        # plotting the ensemble = 2 sigma spread of ensemble
        plt.fill_between(t, ed[0][3][0], ed[0][3][1], alpha=0.25, color=c0)
        plt.fill_between(t, ed[0][2][0][:, 0], ed[0][2][1][:, 0], alpha=0.25, color=c1)
        plt.fill_between(t, ed[0][2][0][:, 1], ed[0][2][1][:, 1], alpha=0.25, color=c2)
        plt.fill_between(t, ed[0][2][0][:, 2], ed[0][2][1][:, 2], alpha=0.25, color=c3)
        
    if best == True:
        # plotting the run with the parameters with min(eps)
        plt.plot(t, np.sqrt(np.sum(outa**2, axis=1)), alpha=0.5, linestyle='dashed', lw=lw_best, color=c0)#, label='parameters with min(eps)')
        plt.plot(t, outa[:, 0], alpha=0.5, linestyle='dashed', lw=lw_best, color=c1)
        plt.plot(t, outa[:, 1], alpha=0.5, linestyle='dashed', lw=lw_best, color=c2)
        plt.plot(t, outa[:, 2], alpha=0.5, linestyle='dashed', lw=lw_best, color=c3)

    if mean == True:
        # plotting the run with mean of the parameters
        plt.plot(t, np.sqrt(np.sum(outam**2, axis=1)), c0, alpha=0.5, linestyle='dashed', lw=lw_mean)#, label='mean parameters') 
        plt.plot(t, outam[:, 0], c1, alpha=0.5, linestyle='dashed', lw=lw_mean)
        plt.plot(t, outam[:, 1], c2, alpha=0.5, linestyle='dashed', lw=lw_mean)
        plt.plot(t, outam[:, 2], c3, alpha=0.5, linestyle='dashed', lw=lw_mean)    
        
    
    # finding index of last fitting point (important for plotting the prediction)
    tempt = []
    
    for i in range(len(t)):
        temptt = t[i].strftime('%Y-%m-%d-%H-%M')
        tempt.append(temptt)   
    
    temp_fit = t_fit[-1].strftime('%Y-%m-%d-%H-%M')
    #print(temp_fit)
    
    tind = tempt.index(temp_fit)
    #print(tind, t[tind])
    
    # plotting magnetic field data
    plt.plot(t[0:tind], np.sqrt(np.sum(b[0:tind]**2, axis=1)), c0, alpha=0.5, lw=3, label='|$\mathbf{B}$|')
    plt.plot(t[0:tind], b[0:tind, 0], c1, alpha=1, lw=lw_insitu, label='B$_r$')
    plt.plot(t[0:tind], b[0:tind, 1], c2, alpha=1, lw=lw_insitu, label='B$_t$')
    plt.plot(t[0:tind], b[0:tind, 2], c3, alpha=1, lw=lw_insitu, label='B$_n$')

    
    if prediction == True:
        # plotting the magnetic field data as dots from the last fitting point onwards
        plt.plot(t[tind+1:-1], np.sqrt(np.sum(b[tind+1:-1]**2, axis=1)), c0, ls=':', alpha=0.5, lw=3)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 0], c1, ls=':', alpha=1, lw=lw_insitu)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 1], c2, ls=':', alpha=1, lw=lw_insitu)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 2], c3, ls=':', alpha=1, lw=lw_insitu)    
        
    else:
        # plotting the magnetic field data as usual from the last fitting point onwards
        plt.plot(t[tind+1:-1], np.sqrt(np.sum(b[tind+1:-1]**2, axis=1)), c0, alpha=0.5, lw=3)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 0], c1, alpha=1, lw=lw_insitu)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 1], c2, alpha=1, lw=lw_insitu)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 2], c3, alpha=1, lw=lw_insitu)    

            
    date_form = mdates.DateFormatter("%h %d %H")
    plt.gca().xaxis.set_major_formatter(date_form)
           
    plt.ylabel("B [nT]")
    plt.xticks(rotation=25, ha='right')
    plt.xlim(start,end)
    
    # plotting lines at t_s and t_e of fit interval
    plt.axvline(x=t_s, lw=lw_fitp, alpha=0.75, color="k", ls="-.")
    plt.axvline(x=t_e, lw=lw_fitp, alpha=0.75, color="k", ls="-.")
    

    if legend == True:
        plt.legend(loc='lower right', ncol=2)
        
    if fit_points == True:    
        for _ in t_fit:
            plt.axvline(x=_, lw=lw_fitp, alpha=0.25, color="k", ls="--")
            
    if save_fig == True:
        plt.savefig(filepath[:-7] + 'fullinsitu.png', dpi=300)  
        plt.savefig(filepath[:-7] + 'fullinsitu.pdf', dpi=300) 
        
    plt.show()  

