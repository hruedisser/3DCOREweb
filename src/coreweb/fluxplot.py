import os

import numpy as np
import pickle as p
import pandas as pds
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('paper')

import datetime as datetime
from datetime import timedelta
import py3dcore as py3dcore
from py3dcore.methods.method import BaseMethod

import heliosat

from py3dcore.models.toroidal import thin_torus_gh, thin_torus_qs, thin_torus_sq

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
def generate_ensemble(path: str, dt: datetime.datetime, reference_frame: str="HCI", reference_frame_to: str="HCI", perc: float=0.95, max_index=None) -> np.ndarray:
    
    """
    Generates an ensemble from a Fitter object.
    
    Arguments:
        path                where to load from
        dt                  time axis used for fitting
        reference_frame     reference frame used for fitter object
        reference_frame_to  reference frame for output data
        perc                percentage of quantile to be used
        max_index           how much of ensemble is kept
    Returns:
        ensemble_data 
    """
    
    observers = BaseMethod(path).observers
    ensemble_data = []
    

    for (observer, _, _, _, _, _) in observers:
        ftobj = BaseMethod(path) # load Fitter from path
        
        observer_obj = getattr(heliosat, observer)() # get observer object
            
        # simulate flux ropes using iparams from loaded fitter
        ensemble = np.squeeze(np.array(ftobj.model_obj.simulator(dt, observer_obj.trajectory(dt, reference_frame=reference_frame))[0]))
        
        # how much to keep of the generated ensemble
        if max_index is None:
            max_index = ensemble.shape[1]

        ensemble = ensemble[:, :max_index, :]

        # transform frame
        if reference_frame != reference_frame_to:
            for k in range(0, ensemble.shape[1]):
                ensemble[:, k, :] = transform_reference_frame(dt, ensemble[:, k, :], reference_frame, reference_frame_to)

        ensemble[np.where(ensemble == 0)] = np.nan

        # generate quantiles
        b_m = np.nanmean(ensemble, axis=1)

        b_s2p = np.nanquantile(ensemble, 0.5 + perc / 2, axis=1)
        b_s2n = np.nanquantile(ensemble, 0.5 - perc / 2, axis=1)

        b_t = np.sqrt(np.sum(ensemble**2, axis=2))
        b_tm = np.nanmean(b_t, axis=1)

        b_ts2p = np.nanquantile(b_t, 0.5 + perc / 2, axis=1)
        b_ts2n = np.nanquantile(b_t, 0.5 - perc / 2, axis=1)

        ensemble_data.append([None, None, (b_s2p, b_s2n), (b_ts2p, b_ts2n)])
        
        return ensemble_data
    

def get_params(filepath, give_mineps=False):
    
    """ Gets params from file. """
    ######### get parameters (mean +/- std)
    
    fit_res = py3dcore.ABC_SMC(filepath)
    fit_res_mean = np.mean(fit_res.model_obj.iparams_arr, axis=0)
    fit_res_std = np.std(fit_res.model_obj.iparams_arr, axis=0)

    keys = list(fit_res.model_obj.iparams.keys()) # names of the parameters 

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
    
        for i in range(1, len(keys)):
            logger.info(" --{} {:.2f}".format(keys[i], resparams[i]))

    return resparams, fit_res_mean, fit_res_std, ip, keys, iparams_arrt


def get_overwrite(out):
    
    """ creates iparams from run with specific output"""

    
    overwrite = {
        "cme_longitude": {
                "maximum": out[1] + 1,
                "minimum": out[1] - 1
            },
        "cme_latitude": {
                "maximum": out[2] + 1,
                "minimum": out[2] - 1
            },
        "cme_inclination" :{
                "maximum": out[3] + 1,
                "minimum": out[3] - 1
            },
        "cme_diameter_1au" :{
                "maximum": out[4] + 0.01,
                "minimum": out[4] - 0.01
            },
        "cme_aspect_ratio": {
                "maximum": out[5] + 0.1,
                "minimum": out[5] - 0.1
            },
        "cme_launch_radius": {
                "distribution": "uniform",
                "maximum": out[6] + 0.1,
                "minimum": out[6] - 0.1
            },
        "cme_launch_velocity": {
                "maximum": out[7] + 100 + 1,  # faster speed
                "minimum": out[7] + 100 - 1
            },
        "t_factor": {
                "maximum": out[8] + 1,
                "minimum": out[8] - 1
            },
        "cme_expansion_rate": {
                "default_value": out[9]
            },
        "magnetic_decay_rate": {
                "default_value": out[10]
            },
        "magnetic_field_strength_1au": {
                "maximum": out[11] + 0.1,
                "minimum": out[11] - 0.1
            },
        "background_drag": {
                "maximum": out[12] + 0.01,
                "minimum": out[12] - 0.01
            },
        "background_velocity": {
                "maximum": out[13] + 1,
                "minimum": out[13] - 1
            }
    }
    
    return overwrite


def get_ensemble_stats(filepath):
    
    ftobj = BaseMethod(filepath) # load Fitter from path
    model_obj = ftobj.model_obj
    
    df = pds.DataFrame(model_obj.iparams_arr)
    cols = df.columns.values.tolist()

    # drop first column
    df.drop(df.columns[[0]], axis=1, inplace=True)

    # rename columns
    df.columns = ['lon', 'lat', 'inc', 'D1AU', 'delta', 'launch radius', 'launch speed', 't factor', 'expansion rate', 'B decay rate', 'B1AU', 'gamma', 'vsw']
    
    df.describe()
    
    return df
    

def scatterparams(path):
    
    ''' returns scatterplots from a results file'''
    
    res, res_mean, res_std, ind, keys, iparams_arrt = get_params(path)
    
    df = pds.DataFrame(iparams_arrt)
    cols = df.columns.values.tolist()

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
    

def loadpickle(path=None, number=-1):

    """ Loads the filepath of a pickle file. """

    # Get the list of all files in path
    dir_list = sorted(os.listdir(path))

    resfile = []
    respath = []
    # we only want the pickle-files
    for file in dir_list:
        if file.endswith(".pickle"):
            resfile.append(file) 
            respath.append(os.path.join(path,file))
            
    filepath = path + resfile[number]

    return filepath


def fullinsitu(observer, t_fit=None, start=None, end=None, filepath=None, ref_frame=None, save_fig=True, best=True, ensemble=True, legend=True, max_index=128, title=True, fit_points=True, prediction=False):
    
    """
    Plots the synthetic insitu data plus the measured insitu data and ensemble fit.

    Arguments:
        observer          name of the observer
        t_fit             datetime points used for fitting
        start             starting point of the plot
        end               ending point of the plot
        filepath          where to find the fitting results (the pickle files)
        ref_frame         the reference frame of the spacecraft data
        save_fig          whether to save the created figure
        best              whether to plot the run with min(eps) ('best')
        ensemble          whether to plot the ensemble spread of 2 sigma
        legend            whether to plot legend 
        max_index         how much to keep of the generated ensemble
        title             whether to plot title 
        fit_points        whether to indicate fitting points in plot
        prediction        when 3DCORE is used for flux rope prediction

    Returns:
        None
    """
    
    if start == None:
        start = t_fit[0]

    if end == None:
        end = t_fit[-1]
    
    print(start, end)
    
    observer_obj = getattr(heliosat, observer)() # get observer obj
    logger.info("Using HelioSat to retrieve observer data")
    t, b = observer_obj.get([start, end], "mag", reference_frame=ref_frame, as_endpoints=True, return_datetimes=True)
    #print(len(t))
    #t = []
    #t = [datetime.datetime.fromtimestamp(dt[i]) for i in range(len(dt))] # .strftime('%Y-%m-%d %H:%M:%S.%f')
    pos = observer_obj.trajectory(t, reference_frame=ref_frame)
    #print(pos)
    
    if best == True:
        model_obj = returnfixedmodel(filepath)
        outa = np.squeeze(np.array(model_obj.simulator(t, pos))[0])       
        outa[outa==0] = np.nan
        print(len(outa))
        print(outa)
             
    # get ensemble_data
    if ensemble == True:
        ed = generate_ensemble(filepath, t, reference_frame=ref_frame, reference_frame_to=ref_frame, max_index=max_index)
    
    lw_insitu = 2  # linewidth for plotting the in situ data
    lw_best = 3  # linewidth for plotting the min(eps) run
    lw_mean = 3  # linewidth for plotting the mean run
    lw_fitp = 2  # linewidth for plotting the lines where fitting points
    
    if observer == 'SOLO':
        obs_title = 'Solar Orbiter'

    if observer == 'PSP':
        obs_title = 'Parker Solar Probe'
        
    if observer == 'Wind':
        obs_title = 'Wind'    

    plt.figure(figsize=(20, 10))
    
    if title == True:
        plt.title("3DCORE fitting result - "+obs_title)
    
    if ensemble == True:
        # plotting the ensemble = 2 sigma spread of ensemble
        plt.fill_between(t, ed[0][3][0], ed[0][3][1], alpha=0.25, color="k")
        plt.fill_between(t, ed[0][2][0][:, 0], ed[0][2][1][:, 0], alpha=0.25, color="r")
        plt.fill_between(t, ed[0][2][0][:, 1], ed[0][2][1][:, 1], alpha=0.25, color="g")
        plt.fill_between(t, ed[0][2][0][:, 2], ed[0][2][1][:, 2], alpha=0.25, color="b")
        
    if best == True:
        # plotting the run with the parameters with min(eps)
        plt.plot(t, np.sqrt(np.sum(outa**2, axis=1)), "k", alpha=0.5, linestyle='dashed', lw=lw_best, label='parameters with min(eps)')
        
            
        plt.plot(t, outa[:, 0], "r", alpha=0.5, linestyle='dashed', lw=lw_best)
        plt.plot(t, outa[:, 1], "g", alpha=0.5, linestyle='dashed', lw=lw_best)
        plt.plot(t, outa[:, 2], "b", alpha=0.5, linestyle='dashed', lw=lw_best)
        
    
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
    plt.plot(t[0:tind], np.sqrt(np.sum(b[0:tind]**2, axis=1)), "k", alpha=0.5, lw=3, label='|$\mathbf{B}$|')
    plt.plot(t[0:tind], b[0:tind, 0], "r", alpha=1, lw=lw_insitu, label='B$_r$')
    plt.plot(t[0:tind], b[0:tind, 1], "g", alpha=1, lw=lw_insitu, label='B$_t$')
    plt.plot(t[0:tind], b[0:tind, 2], "b", alpha=1, lw=lw_insitu, label='B$_n$')
    
    if prediction == True:
        # plotting the magnetic field data as dots from the last fitting point onwards
        plt.plot(t[tind+1:-1], np.sqrt(np.sum(b[tind+1:-1]**2, axis=1)), "k", ls=':', alpha=0.5, lw=3)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 0], "r", ls=':', alpha=1, lw=lw_insitu)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 1], "g", ls=':', alpha=1, lw=lw_insitu)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 2], "b", ls=':', alpha=1, lw=lw_insitu)    
        
    else:
        # plotting the magnetic field data as usual from the last fitting point onwards
        plt.plot(t[tind+1:-1], np.sqrt(np.sum(b[tind+1:-1]**2, axis=1)), "k", alpha=0.5, lw=3)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 0], "r", alpha=1, lw=lw_insitu)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 1], "g", alpha=1, lw=lw_insitu)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 2], "b", alpha=1, lw=lw_insitu)    
            
    date_form = mdates.DateFormatter("%h %d %H")
    plt.gca().xaxis.set_major_formatter(date_form)
           
    plt.ylabel("B [nT]")
    plt.xticks(rotation=25, ha='right')
    
    if legend == True:
        plt.legend(loc='lower right', ncol=2)
        
    if fit_points == True:    
        for _ in t_fit:
            plt.axvline(x=_, lw=lw_fitp, alpha=0.25, color="k", ls="--")
            
    if save_fig == True:
        plt.savefig(filepath[:-7] + 'fullinsitu.pdf', dpi=300)  
        
    plt.show()
    
    
def returnfixedmodel(filepath):
    
    '''returns a fixed model not generating random iparams, but using the parameters of the run with min(eps)'''
    
    ftobj = BaseMethod(filepath) # load Fitter from path
    model_obj = ftobj.model_obj
    
    model_obj.ensemble_size = 1
    
    logger.info("Using parameters for run with minimum eps.")
    res_mineps, res_mean, res_std, ind, keys, allres = get_params(filepath)
    model_obj.iparams_arr = np.expand_dims(res_mineps, axis=0)
    
    model_obj.sparams_arr = np.empty((model_obj.ensemble_size, model_obj.sparams), dtype=model_obj.dtype)
    model_obj.qs_sx = np.empty((model_obj.ensemble_size, 4), dtype=model_obj.dtype)
    model_obj.qs_xs = np.empty((model_obj.ensemble_size, 4), dtype=model_obj.dtype)
    
    model_obj.iparams_meta = np.empty((len(model_obj.iparams), 7), dtype=model_obj.dtype)
    
    #iparams_meta is updated
    generate_quaternions(model_obj.iparams_arr, model_obj.qs_sx, model_obj.qs_xs)
    
    return model_obj  


def returnmodel(filepath):
    
    '''returns a model using the parameters of the run with min(eps) from a previous result'''
    
    t_launch = BaseMethod(filepath).dt_0
    
    res_mineps, res_mean, res_std, ind, keys, allres = get_params(filepath)
    overwrite = get_overwrite(res_mineps)
    #print(overwrite)
    
    model_obj = py3dcore.ToroidalModel(t_launch, 1, iparams=overwrite)
    
    model_obj.generator()
    
    return model_obj


def full3d(spacecraftlist=['solo', 'psp'], planetlist=['Earth'], t=None, traj=50, filepath=None, save_fig=True, legend=True, 
           title=True, view_azim=0, view_elev=45, view_radius=0.2, index=0, **kwargs):
    
    """
    Plots 3d.
    
    Parameters:
        index        the index of the run with parameters with min(eps)
    """
    
    #colors for 3dplots

    c0 = 'mediumseagreen'
    c1 = "xkcd:red"
    c2 = "xkcd:blue"
    
    #Color settings    
    C_A = "xkcd:red"
    C_B = "xkcd:blue"

    C0 = "xkcd:black"
    C1 = "xkcd:magenta"
    C2 = "xkcd:orange"
    C3 = "xkcd:azure"

    earth_color='blue'
    solo_color='mediumseagreen'
    venus_color='orange'
    mercury_color='grey'
    psp_color='mediumorchid'
    sta_color='red'
    bepi_color='coral' 
    
    sns.set_context("talk")     

    sns.set_style("ticks",{'grid.linestyle': '--'})
    fsize=15

    fig = plt.figure(figsize=(13,9),dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    plot_configure(ax, view_azim=view_azim, view_elev=view_elev, view_radius=view_radius)
    
    model_obj = returnmodel(filepath)
    
    plot_3dcore(ax, model_obj, t, index=index, color=c2)
    #plot_3dcore_field(ax, model_obj, color=c2, step_size=0.005, lw=1.1, ls="-")
    
    if 'solo' in spacecraftlist:

        plot_traj(ax, sat='Solar Orbiter', t_snap=t, frame="HEEQ", traj_pos=True, traj_major=traj, traj_minor=None, color=solo_color, **kwargs)
        
        
    if 'psp' in spacecraftlist:
        plot_traj(ax, sat='Parker Solar Probe', t_snap=t, frame="HEEQ", traj_pos=True, traj_major=traj, traj_minor=None, color=psp_color, **kwargs)
        
    
    if 'Earth' in planetlist:
        earthpos = np.asarray([1, 0, 0])
        plot_planet(ax, earthpos, color=earth_color, alpha=0.9, label='Earth')
        plot_circle(ax, earthpos[0])
        
    #if 'Venus' in planetlist:
    #    t_ven, pos_ven, traj_ven=getpos('Venus Barycenter', t.strftime('%Y-%m-%d-%H'), start, end)
    #    plot_planet(ax, pos_ven, color=venus_color, alpha=0.9, label='Venus')
    #    plot_circle(ax, pos_ven[0])
        
    #if 'Mercury' in planetlist:
    #    t_mer, pos_mer, traj_mer=getpos('Mercury Barycenter', t.strftime('%Y-%m-%d-%H'), start, end)
    #    plot_planet(ax, pos_mer, color=mercury_color, alpha=0.9, label='Mercury')
    #    plot_circle(ax, pos_mer[0])    
    
    if legend == True:
        ax.legend(loc='lower left')
    if title == True:
        plt.title('3DCORE fitting result - ' + t.strftime('%Y-%m-%d-%H-%M'))
    if save_fig == True:
        plt.savefig(filepath[:-7] + 'full3d.pdf', dpi=300)  
    
    return fig


def plot_traj(ax, sat, t_snap, frame="HEEQ", traj_pos=True, traj_minor=None, **kwargs):
    kwargs["alpha"] = kwargs.pop("alpha", 1)
    color = kwargs.pop("color", "k")
    kwargs["lw"] = kwargs.pop("lw", 1)
    kwargs["s"] = kwargs.pop("s", 25)
    traj_major = kwargs.pop("traj_major", 80)
    
    if sat == "Solar Orbiter":
        print('observer:', sat)
        observer = "SOLO"
        
    elif sat == "Parker Solar Probe":
        print('observer:', sat)
        observer = "PSP"
        
    else:
        print('no observer specified')
        
    #observer_obj.trajectory(dt, reference_frame=reference_frame    
        
    inst = getattr(heliosat, observer)() # get observer obj
    #print('inst', inst)
    logger.info("Using HelioSat to retrieve observer data")
    
    _s = kwargs.pop("s")

    if traj_pos:
        print('t_snap:', t_snap)
        pos = inst.trajectory(t_snap, reference_frame="HEEQ")
        #print('pos [HEEQ - xyz]:', pos)

        ax.scatter(*pos.T, s=_s, **kwargs)
        
    if traj_major and traj_major > 0:
        traj = inst.trajectory([t_snap + datetime.timedelta(hours=i) for i in range(-traj_major, traj_major)], reference_frame="HEEQ")
        ax.plot(*traj.T, **kwargs)
        
    if traj_minor and traj_minor > 0:
        traj = inst.trajectory([t_snap + datetime.timedelta(hours=i) for i in range(-traj_minor, traj_minor)], reference_frame="HEEQ")
        
    if "ls" in kwargs:
        kwargs.pop("ls")

    _ls = "--"
    _lw = kwargs.pop("lw") / 2

    # ax.plot(*traj.T, ls=_ls, lw=_lw, **kwargs)


def full3d_multiview(tsnap, filepath):
    
    """
    Plots 3d from multiple views.
    
    Parameters
        tsnaps       times for spacecrafts
    """
    
    TP_A =  tsnap
    TP_B =  tsnap + datetime.timedelta(hours=41)

    C_A = "xkcd:red"
    C_B = "xkcd:blue"

    C0 = "xkcd:black"
    C1 = "xkcd:magenta"
    C2 = "xkcd:orange"
    C3 = "xkcd:azure"

    sns.set_style('whitegrid')

    fig = plt.figure(figsize=(15, 11), dpi=100)

    #define subplot grid
    ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2, projection='3d')  
    ax2 = plt.subplot2grid((2, 3), (0, 2), projection='3d')  
    ax3 = plt.subplot2grid((2, 3), (1, 2), projection='3d')  
    
    model_obj = returnmodel(filepath)
    
    ######### tilted view
    plot_configure(ax1, view_azim=150, view_elev=25, view_radius=.2, light_source=True) #view_radius=.08

    plot_3dcore(ax1, model_obj, TP_A, color=C_A, light_source=True)
    #plot_3dcore_field(ax1, model_obj, color=C_A, step_size=0.0005, lw=1.0, ls="-")
    plot_traj(ax1, "Parker Solar Probe", t_snap=TP_A, frame="HEEQ", color=C_A)
    
    #plot_3dcore(ax1, model_obj, TP_B, color=C_B, light_source = True)
    #plot_3dcore_field(ax1, model_obj, color=C_B, step_size=0.001, lw=1.0, ls="-")
    #plot_traj(ax1, "Solar Orbiter", t_snap = TP_B, frame="HEEQ", custom_data = 'sunpy', color=C_B)
    
    #dotted trajectory
    #plot_traj(ax1, "PSP", TP_B, frame="ECLIPJ2000", color="k", traj_pos=False, traj_major=None, traj_minor=144,lw=1.5)

    #shift center
    plot_shift(ax1, 0.31, -0.25, 0.0, -0.2)
    
    
    ########### top view panel
    plot_configure(ax2, view_azim=165-90, view_elev=90, view_radius=.08, light_source=True)
    
    plot_3dcore(ax2, model_obj, TP_A, color=C_A, light_source=True)
    #plot_3dcore_field(ax2, model_obj, color=C_A, step_size=0.0005, lw=1.0, ls="-")
    plot_traj(ax2, "Parker Solar Probe", t_snap=TP_A, frame="HEEQ", color=C_A)
    
    #plot_3dcore(ax2, model_obj, TP_B, color=C_B, light_source = True)
    #plot_3dcore_field(ax2, model_obj, color=C_B, step_size=0.001, lw=1.0, ls="-")
    #plot_traj(ax2, "Solar Orbiter", t_snap = TP_B, frame="HEEQ", custom_data = 'sunpy', color=C_B)
    plot_shift(ax2, 0.26, -0.41, 0.08, 0.0) 
    
    
    ############## edge on view panel
    plot_configure(ax3, view_azim=65, view_elev=-5, view_radius=.01, light_source=True)
    plot_traj(ax3, "Parker Solar Probe", t_snap=TP_A, frame="HEEQ", color=C_A)

    #plot_3dcore(ax3, model_obj, TP_B, color=C_B, light_source = True)
    ##plot_3dcore_field(ax3, model_obj, color=C_B, step_size=0.001, lw=1.0, ls="-")
    #plot_traj(ax3, "Solar Orbiter", t_snap = TP_B, frame="HEEQ", custom_data = 'sunpy', color=C_B)

    plot_shift(ax3, 0.26, -0.41, 0.08, 0.0)


    plt.savefig(filepath[:-7] + 'multiview.pdf', bbox_inches='tight')
    
    
def plot_3dcore(ax, obj, t_snap, index=0, light_source=False, **kwargs):
    kwargs["alpha"] = kwargs.pop("alpha", .05)
    kwargs["color"] = kwargs.pop("color", "k")
    kwargs["lw"] = kwargs.pop("lw", 1)

    if light_source == False:
        ax.scatter(0, 0, 0, color="y", s=50) # 5 solar radii
        
    #ax.plot_surface(x, y, z, rstride=1, cstride=1, color='yellow', linewidth=0, antialiased=False)

    obj.propagator(t_snap)
    wf_model = obj.visualize_shape(iparam_index=index)  
    #wf_model = obj.visualize_shape(iparam_index=2)  
    #ax.plot_wireframe(*wf_model.T, **kwargs)
    ax.plot_wireframe(*wf_model.T, **kwargs)

    ## from reader notebook
    #fit_wind.model_obj.propagator(datetime.datetime(2022, 2, 3, tzinfo=datetime.timezone.utc))
    #fig = plt.figure(figsize=(12, 12))
    #ax = fig.add_subplot(111, projection='3d')
    #wf=fit_wind.model_obj.visualize_shape(iparam_index=2)
    #ax.plot_wireframe(*wf.T)
    
    
def plot_circle(ax,dist,**kwargs):        

    thetac = np.linspace(0, 2 * np.pi, 100)
    xc = dist * np.sin(thetac)
    yc = dist * np.cos(thetac)
    zc = 0
    ax.plot(xc, yc, zc, ls='--', color='black', lw=0.3, **kwargs)
    
    
def plot_planet(ax, satpos1, **kwargs):

    xc = satpos1[0] * np.cos(np.radians(satpos1[1]))
    yc = satpos1[0] * np.sin(np.radians(satpos1[1]))
    zc = 0
    #print(xc,yc,zc)
    ax.scatter3D(xc, yc, zc, s=10, **kwargs)
    
    
def plot_configure(ax, light_source=False, **kwargs):
    view_azim = kwargs.pop("view_azim", -25)
    view_elev = kwargs.pop("view_elev", 25)
    view_radius = kwargs.pop("view_radius", .5)
    
    ax.view_init(azim=view_azim, elev=view_elev)

    ax.set_xlim([-view_radius, view_radius])
    ax.set_ylim([-view_radius, view_radius])
    ax.set_zlim([-view_radius, view_radius])
    
    if light_source == True:
        #draw sun        
        ls = LightSource(azdeg=320, altdeg=40)  
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='yellow', lightsource=ls, linewidth=0, antialiased=False, zorder=5)
    
    ax.set_axis_off()
    
    
def plot_shift(axis, extent, cx, cy, cz):
    #shift center of plot
    axis.set_xbound(cx - extent, cx + extent)
    axis.set_ybound(cy - extent, cy + extent)
    axis.set_zbound(cz - extent * 0.75, cz + extent * 0.75)
    
#define sun here so it does not need to be recalculated every time
scale = 695510 / 149597870.700 #Rs in km, AU in km
# sphere with radius Rs in AU
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
x = np.cos(u) * np.sin(v) * scale
y = np.sin(u) * np.sin(v) * scale
z = np.cos(v) * scale    