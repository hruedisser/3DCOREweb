import os

import numpy as np
import pickle as p
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('paper')

import datetime
from datetime import timedelta
import coreweb
from .methods.method import BaseMethod

from coreweb.dashcore.utils.utils import generate_ensemble, get_iparams_live, round_to_hour_or_half
from coreweb.methods.offwebutils import extract_row
import coreweb.dashcore.utils.heliocats as hc

import coreweb.methods.data_frame_transforms as dft

import heliosat

from coreweb.models.toroidal import thin_torus_gh, thin_torus_qs, thin_torus_sq

from .rotqs import generate_quaternions

import matplotlib as matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LightSource
import matplotlib.lines as mlines

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
    
    #print(start, end)


    t = graph['t_data']

    # in situ data

    # Find indices for the closest values to start and end
    start_index = np.argmin(np.abs(t - start))
    end_index = np.argmin(np.abs(t - end))

    # Filter data to contain only values from start to end
    t = t[start_index:end_index+1]

    #print(t)

    # in situ data
    if ref_frame == "HEEQ":
        b = graph['b_data_HEEQ'][start_index:end_index+1]
        names = ['B$_X$','B$_Y$','B$_Z$']
    elif ref_frame == "GSM":
        b = graph['b_data_GSM'][start_index:end_index+1]
        names = ['B$_X$','B$_Y$','B$_Z$']
    else:
        b = graph['b_data_RTN'][start_index:end_index+1]
        names = ['B$_R$','B$_T$','B$_N$']
    
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
        #print(iparams)
        model_obj = coreweb.ToroidalModel(launchtime, **iparams) # model gets initialized
        model_obj.generator()  

        #print('arra', model_obj.iparams_arr)

        outa = np.array(model_obj.simulator(graph['t_data'][start_index:end_index+1], graph['pos_data'][start_index:end_index+1]), dtype=object)

        outa = np.squeeze(outa[0])

        #print(graph['t_data'])
        #print(outa)

        if ref_frame != "HEEQ":
            x,y,z = hc.separate_components(graph['pos_data'][start_index:end_index+1])
            #print(x,y,z)
            bx,by,bz = hc.separate_components(outa)
            rtn_bx, rtn_by, rtn_bz = hc.convert_HEEQ_to_RTN_mag(x,y,z, bx,by,bz)
            outa[:, 0],outa[:, 1],outa[:, 2] = rtn_bx, rtn_by, rtn_bz    

            if ref_frame == "GSM":
                gsm_bx, gsm_by, gsm_bz = dft.RTN_to_GSM(x, y, z, rtn_bx,rtn_by,rtn_bz, t)
                outa[:, 0],outa[:, 1],outa[:, 2] = gsm_bx, gsm_by, gsm_bz   

            if np.any(rtn_bx > 1500) or np.any(rtn_by > 1500) or np.any(rtn_bz > 1500):
                print(iparams)
                #return
            #else:
            #    return
        outa[outa==0] = np.nan
        #print(outa)
        #print(graph['t_data'][start_index:end_index+1])
        
    # get ensemble_data
    if ensemble == True:
        # read from pickle file
        file = open(filepath, "rb")
        data = p.load(file)
        file.close()
        
        if ref_frame == 'GSM':
            ensemble_filepath = filepath.split('.')[0] + '_ensembles_GSM.pickle'
            with open(ensemble_filepath, 'rb') as ensemble_file:
                ensemble_data = p.load(ensemble_file)  
            ed = ensemble_data['ensemble_GSM']
        else:
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

    #print( len(ed[0][3][0][start_index:end_index+1]))
    #print( len(outa))
    
    if ensemble == True:
        # plotting the ensemble = 2 sigma spread of ensemble
        plt.fill_between(t, ed[0][3][0][start_index:end_index+1], ed[0][3][1][start_index:end_index+1], alpha=0.25, color=c0)
        plt.fill_between(t, ed[0][2][0][:, 0][start_index:end_index+1], ed[0][2][1][:, 0][start_index:end_index+1], alpha=0.25, color=c1)
        plt.fill_between(t, ed[0][2][0][:, 1][start_index:end_index+1], ed[0][2][1][:, 1][start_index:end_index+1], alpha=0.25, color=c2)
        plt.fill_between(t, ed[0][2][0][:, 2][start_index:end_index+1], ed[0][2][1][:, 2][start_index:end_index+1], alpha=0.25, color=c3)
        
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
        
    

    
    if prediction == True:

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
        plt.plot(t[0:tind], np.sqrt(np.sum(b[0:tind]**2, axis=1)), c0, alpha=0.5, lw=3, label='B$_{TOT}$')
        plt.plot(t[0:tind], b[0:tind, 0], c1, alpha=1, lw=lw_insitu, label=names[0])
        plt.plot(t[0:tind], b[0:tind, 1], c2, alpha=1, lw=lw_insitu, label=names[1])
        plt.plot(t[0:tind], b[0:tind, 2], c3, alpha=1, lw=lw_insitu, label=names[2])


        # plotting the magnetic field data as dots from the last fitting point onwards
        plt.plot(t[tind+1:-1], np.sqrt(np.sum(b[tind+1:-1]**2, axis=1)), c0, ls=':', alpha=0.5, lw=3)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 0], c1, ls=':', alpha=1, lw=lw_insitu)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 1], c2, ls=':', alpha=1, lw=lw_insitu)
        plt.plot(t[tind+1:-1], b[tind+1:-1, 2], c3, ls=':', alpha=1, lw=lw_insitu)    
        
    else:
        # plotting the magnetic field data as usual from the last fitting point onwards
        plt.plot(t, np.sqrt(np.sum(b**2, axis=1)), c0, alpha=0.5, lw=3, label='B$_{TOT}$')
        plt.plot(t, b[:, 0], c1, alpha=1, lw=lw_insitu, label=names[0])
        plt.plot(t, b[:, 1], c2, alpha=1, lw=lw_insitu, label=names[1])
        plt.plot(t, b[:, 2], c3, alpha=1, lw=lw_insitu, label=names[2])    

    #print(iparams)
            
    date_form = mdates.DateFormatter("%h %d %H")
    plt.gca().xaxis.set_major_formatter(date_form)
           
    plt.ylabel("B [nT]")
    plt.xticks(rotation=25, ha='right')
    plt.xlim(start,end)
    
    if t_s is not None:# plotting lines at t_s and t_e of fit interval
        plt.axvline(x=t_s, lw=lw_fitp, alpha=0.75, color="k", ls="-.")
    if t_e is not None:# plotting lines at t_s and t_e of fit interval
        plt.axvline(x=t_e, lw=lw_fitp, alpha=0.75, color="k", ls="-.")
    
    #plt.grid(color = 'lightgrey')

    if legend == True:
        plt.legend(loc='lower right', ncol=2)
        
    if fit_points == True:    
        for _ in t_fit:
            plt.axvline(x=_, lw=lw_fitp, alpha=0.25, color="k", ls="--")
            
    if save_fig == True:
        plt.savefig(filepath[:-7] + 'fullinsitu' + ref_frame +'.png', dpi=300)  
        plt.savefig(filepath[:-7] + 'fullinsitu' + ref_frame +'.pdf', dpi=300) 
        
    plt.show()  
    plt.savefig('pulsarplots/fittingresults.png')


def full3d(graph, timesnap, plotoptions, spacecraftoptions=['solo', 'psp'], bodyoptions=['Earth'], *modelstatevars, viewlegend = False, posstore = None, addfield = False, launchtime=None, 
           title=False, view_azim=0, view_elev=45, view_radius=0.2, black = False, sc = 'SOLO', fontsize = 8, showtext = True):
    
    """
    Plots 3d.
    
    Parameters:
        index        the index of the run with specific parameters
    """
    
    c2 = 'blue'

    #colors for 3dplots

    earth_color='mediumseagreen'
    #venus_color='orange'
    #mercury_color='dimgrey'
    #mars_color='orangered'

    solo_color='coral'
    wind_color='mediumseagreen'
    psp_color='black'
    syn_color="red"
    #sta_color='red'
    #bepi_color='blue' 

    #earth_color='blue'
    venus_color='orange'
    mercury_color='grey'
    sta_color='darkred'
    bepi_color='blue' 

    sns.set_context("talk")     

    sns.set_style("ticks",{'grid.linestyle': '--'})
    fsize=15

    fig = plt.figure(figsize=(13,9),dpi=300)
    if black==True:
        ax = fig.add_subplot(111, projection='3d', facecolor='black')    
    else:    
        ax = fig.add_subplot(111, projection='3d')
    
    if "Sun" in bodyoptions:
        plot_configure(ax, light_source= True, view_azim=view_azim, view_elev=view_elev, view_radius=view_radius)
    else:
        plot_configure(ax, view_azim=view_azim, view_elev=view_elev, view_radius=view_radius)

    # model_obj = returnmodel(filepath)
    iparams = get_iparams_live(*modelstatevars)
    model_obj = coreweb.ToroidalModel(launchtime, **iparams) # model gets initialized
    model_obj.generator()  

    if sc == "SOLO":
        cmecolor = solo_color
    elif sc == 'PSP':
        cmecolor = psp_color
    elif sc == 'STEREO-A':
        cmecolor = sta_color
    elif sc == 'DSCOVR':
        cmecolor = earth_color
    elif sc == 'BEPI':
        cmecolor = bepi_color
    elif sc == 'Wind':
        cmecolor = wind_color
    elif sc == 'SYN':
        cmecolor = syn_color

    plot_3dcore(ax, model_obj, timesnap, color=cmecolor)

    

    if addfield == True:
        plot_3dcore_field(ax, model_obj, timesnap, step_size=0.005, q0=[0.8, 0.1, np.pi/2],color=cmecolor, alpha = .95, lw = .8)

    if "Earth" in bodyoptions:
        try:
            plot_planet(ax, graph['bodydata']['Earth']['data'], timesnap, color=earth_color, alpha=0.9, label='Earth')
            #print(graph['bodydata']['Earth']['data'])
        except Exception as e:
            print('Data for Earth not found: ', e)

    #return fig, ax

    ### insert mercury
    if "Mercury" in bodyoptions:
        try:
            plot_planet(ax, graph['bodydata']['Mercury']['data'], timesnap, color=mercury_color, alpha=0.9, label='Mercury')
        except Exception as e:
            print('Data for Mercury not found: ', e)

    ### insert venus
    ### insert mars
            
    if spacecraftoptions is not None:
        for scopt in spacecraftoptions:
            #try:
            if scopt == "SYN":
                pass
            #insert SYN 
            else:
                plot_traj(ax, posstore[scopt]['data']['data'], launchtime, timesnap, posstore[scopt]['data']['color'], scopt)

    if viewlegend == True:
        ax.legend(loc='best', fontsize = fontsize)
    if title == True:
        plt.title(timesnap.strftime('%Y-%m-%d-%H-%M'))
    
    if "Longitudinal Grid" in plotoptions:
        plot_longgrid(ax, fontsize=fontsize, text=showtext, view_radius = view_radius)

    # #### still do   
    
    # if save_fig == True:
    #     plt.savefig(save_file + '.pdf', dpi=300)  
        
    
    return fig, ax

def add_cme(ax, graph, timesnap, *modelstatevars, addfield = False, launchtime=None, sc = 'SOLO'):
    
    solo_color='coral'
    psp_color='black'
    sta_color='darkred'
    bepi_color='blue' 
    earth_color='mediumseagreen'

    # model_obj = returnmodel(filepath)
    iparams = get_iparams_live(*modelstatevars)
    model_obj = coreweb.ToroidalModel(launchtime, **iparams) # model gets initialized
    model_obj.generator()  

    if sc == "SOLO":
        cmecolor = solo_color
    elif sc == 'PSP':
        cmecolor = psp_color
    elif sc == 'STEREO-A':
        cmecolor = sta_color
    elif sc == 'DSCOVR':
        cmecolor = earth_color
    elif sc == 'BEPI':
        cmecolor = bepi_color
    elif sc == "SYN":
        cmecolor = psp_color

    plot_3dcore(ax, model_obj, timesnap, color=cmecolor)

    if addfield == True:
        plot_3dcore_field(ax, model_obj, timesnap, step_size=0.005, q0=[0.8, 0.1, np.pi/2],color=cmecolor, alpha = .95, lw = .8)

    return ax

def plot_traj(ax, data_list, date, nowdate, color, sc, **kwargs):


    nowdate = round_to_hour_or_half(nowdate.replace(tzinfo=None))
    date = date.replace(tzinfo=None)

    kwargs["alpha"] = kwargs.pop("alpha", 1)
    #color = kwargs.pop("color", "k")
    kwargs["lw"] = kwargs.pop("lw", .9)
    kwargs["s"] = kwargs.pop("s", 20)
    
    try:
        data = np.array(data_list, dtype=[('time', 'O'), ('r', '<f8'), ('lon', '<f8'), ('lat', '<f8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
    except:
        # Convert string dates to datetime objects
        try:
            # Define the dtype for the structured array
            dtype = [('time', 'O'), ('r', '<f8'), ('lon', '<f8'), ('lat', '<f8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8')]

            # Create an empty structured array
            data = np.empty(len(data_list), dtype=dtype)

            # Populate the structured array
            for i, row in enumerate(data_list):
                data[i] = tuple(row)
        except Exception as e:
            print(e)

    df_columns = ['time', 'r', 'lon', 'lat', 'x', 'y', 'z']
    df = pd.DataFrame(data, columns=df_columns)
        
    df['time'] = pd.to_datetime(df['time'])  # Convert time column to datetime objects
    
    # Filter data based on date and nowdate
    filtered_data = df[(df['time'] >= date) & (df['time'] <= date + datetime.timedelta(days=7))]

    # Split data into past, future, and now coordinates
    past_data = filtered_data[filtered_data['time'] < nowdate]
    future_data = filtered_data[filtered_data['time'] > nowdate]
    now_data = filtered_data[filtered_data['time'] == nowdate]

    # Extract coordinates for each category
    x_past, y_past, z_past, times_past = past_data['x'], past_data['y'], past_data['z'], past_data['time']
    x_future, y_future, z_future, times_future = future_data['x'], future_data['y'], future_data['z'], future_data['time']
    x_now, y_now, z_now, now_time = now_data['x'], now_data['y'], now_data['z'], now_data['time']
    
    r_past, lon_past, lat_past = past_data['r'], past_data['lon'], past_data['lat']
    r_future, lon_future, lat_future = future_data['r'], future_data['lon'], future_data['lat']
    r_now, lon_now, lat_now = now_data['r'], now_data['lon'], now_data['lat']
    
    # Convert Timestamp Series to a list of Timestamps
    times_past_list = times_past.tolist()
    times_future_list = times_future.tolist()
    times_now_list = now_time.tolist()
        
    # Convert Timestamps to formatted strings
    times_past_str = [time.strftime("%Y-%m-%d %H:%M:%S") for time in times_past_list]
    times_future_str = [time.strftime("%Y-%m-%d %H:%M:%S") for time in times_future_list]
    now_time_str = [time.strftime("%Y-%m-%d %H:%M:%S") for time in times_now_list]
    
    _s = kwargs.pop("s")

    ax.scatter(xs=x_now, ys=y_now, zs=z_now, color=color, label=sc, s=_s, marker='s', **kwargs)

    ax.plot(xs=x_past, ys=y_past, zs=z_past, color=color, **kwargs)

    _ls = "--"
    _lw = kwargs.pop("lw") * 0.7

    ax.plot(xs=x_future, ys=y_future, zs=z_future, ls=_ls, lw=_lw, color=color) #, **kwargs)



def plot_3dcore(ax, obj, t_snap, light_source=False, **kwargs):
    kwargs["alpha"] = kwargs.pop("alpha", .15)
    kwargs["color"] = kwargs.pop("color", "k")
    kwargs["lw"] = kwargs.pop("lw", 1)

    if light_source == False:
        # draw Sun
        ax.scatter(0, 0, 0, color="y", s=30, label="Sun") 
        
    obj.propagator(t_snap)
    wf_model = obj.visualize_shape(iparam_index=0)  

    wf_array = np.array(wf_model)

    # Extract x, y, and z data from wf_array
    x = wf_array[:,:,0].flatten()
    y = wf_array[:,:,1].flatten()
    z = wf_array[:,:,2].flatten()
    ax.plot_wireframe(*wf_model.T, **kwargs) 


def plot_3dcore_field(ax, obj, t_snap, step_size=0.005, q0=[0.8, 0.1, np.pi/2],**kwargs):
    print('Tracing Fieldlines')
    #q0=[0.9, .1, .5]
    q0i =np.array(q0, dtype=np.float32)
    obj.propagator(t_snap)
    fl, qfl = obj.visualize_fieldline(q0, index=0,  steps=10000, step_size=2e-3, return_phi=True)
    ax.plot(*fl.T, **kwargs)

    diff = qfl[1:-10] - qfl[:-11]
    print("total turns estimates: ", np.sum(diff[diff > 0]) / np.pi / 2)#, np.sum(diff2[diff2 > 0]) / np.pi / 2)



def visualize_fieldline(obj, q0, index=0, steps=1000, step_size=0.01):
        print('visualize_fieldline')
        """Integrates along the magnetic field lines starting at a point q0 in (q) coordinates and
        returns the field lines in (s) coordinates.

        Parameters
        ----------
        q0 : np.ndarray
            Starting point in (q) coordinates.
        index : int, optional
            Model run index, by default 0.
        steps : int, optional
            Number of integration steps, by default 1000.
        step_size : float, optional
            Integration step size, by default 0.01.

        Returns
        -------
        np.ndarray
            Integrated magnetic field lines in (s) coordinates.
        """

        _tva = np.empty((3,), dtype=obj.dtype)
        _tvb = np.empty((3,), dtype=obj.dtype)

        thin_torus_qs(q0, obj.iparams_arr[index], obj.sparams_arr[index], obj.qs_xs[index], _tva)

        fl = [np.array(_tva, dtype=obj.dtype)]
        def iterate(s):
            thin_torus_sq(s, obj.iparams_arr[index], obj.sparams_arr[index], obj.qs_sx[index],_tva)
            thin_torus_gh(_tva, obj.iparams_arr[index], obj.sparams_arr[index], obj.qs_xs[index], _tvb)
            return _tvb / np.linalg.norm(_tvb)

        while len(fl) < steps:
            # use implicit method and least squares for calculating the next step
            try:
                sol = getattr(least_squares(
                    lambda x: x - fl[-1] - step_size *
                    iterate((x.astype(obj.dtype) + fl[-1]) / 2),
                    fl[-1]), "x")

                fl.append(np.array(sol.astype(obj.dtype)))
            except Exception as e:
                break

        fl = np.array(fl, dtype=obj.dtype)

        return fl
        
    
def plot_circle(ax, dist, color=None, **kwargs):        

    thetac = np.linspace(0, 2 * np.pi, 100)
    xc = dist * np.sin(thetac)
    yc = dist * np.cos(thetac)
    zc = 0
    ax.plot(xc, yc, zc, color=color, lw=0.3, **kwargs)

def plot_longgrid(ax, fontsize=6, color = 'k', text = True, view_radius = 1):

    radii = [
        #0.3, 
        0.5, 0.8]

    if view_radius < .3:
        multip1 = .225
        multip2 = .25
        radii = [0.2]
    elif view_radius > .9:
        multip1 = 1.2
        multip2 = 1.3
        radii = radii = [0.5, 0.8, 1.]
    else:
        multip1 = .85
        multip2 = .9
    
    for r in radii:
        plot_circle(ax, r, color=color)
        if text == True:
            ax.text(x = -0.085, y = r - 0.25, z = 0, s = f'{r} AU', fontsize = fontsize) 

    # Create data for the AU lines and their labels
    num_lines = 8
    for i in range(num_lines):
        angle_degrees = -180 + (i * 45)  # Adjusted angle in degrees (-180 to 180)
        angle_radians = np.deg2rad(angle_degrees)
        x = [0, np.cos(angle_radians)* multip1] 
        y = [0, np.sin(angle_radians)* multip1]
        z = [0, 0]

        ax.plot(x, y, z, color=color, lw=0.3)

        label_x = multip2 * np.cos(angle_radians)
        label_y = multip2 * np.sin(angle_radians)
        

        if text == True:
            ax.text(x = label_x, y = label_y, z = 0, s = f'+/{angle_degrees}°' if angle_degrees == -180 else f'{angle_degrees}°', fontsize = fontsize, horizontalalignment='center',
     verticalalignment='center') 

    

    
    
def plot_planet(ax, data_list, nowdate, **kwargs):
    
    nowdate = round_to_hour_or_half(nowdate.replace(tzinfo=None))

    try:
        data = np.array(data_list, dtype=[('time', 'O'), ('r', '<f8'), ('lon', '<f8'), ('lat', '<f8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
    except:
        # Convert string dates to datetime objects
        try:
            # Define the dtype for the structured array
            dtype = [('time', 'O'), ('r', '<f8'), ('lon', '<f8'), ('lat', '<f8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8')]

            # Create an empty structured array
            data = np.empty(len(data_list), dtype=dtype)

            # Populate the structured array
            for i, row in enumerate(data_list):
                data[i] = tuple(row)
        except Exception as e:
            print(e)

    df_columns = ['time', 'r', 'lon', 'lat', 'x', 'y', 'z']
    df = pd.DataFrame(data, columns=df_columns)
    
    df['time'] = pd.to_datetime(df['time'])  # Convert time column to datetime objects
    # Filter data based on date and nowdate
    now_data = df[df['time']== nowdate]
    #print(now_data)
    x_now, y_now, z_now, now_time = now_data['x'], now_data['y'], now_data['z'], now_data['time']
    r_now, lon_now, lat_now = now_data['r'], now_data['lon'], now_data['lat']
    
    times_now_list = now_time.tolist()

    now_time_str = [time.strftime("%Y-%m-%d %H:%M:%S") for time in times_now_list]
    ax.scatter3D(x_now, y_now, z_now, s=20, **kwargs)
    #ax.scatter3D(0.3, 0, 0, s=10, **kwargs)
    
    
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
    
        # Create a proxy artist for the Sun to include in the legend
        #ax.legend(handles=[sun_proxy], loc='upper right')  # Add legend entry for the Sun

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