import cdflib
import base64
import io
import sys
import tempfile
import re
from sunpy.time import parse_time
from scipy.io import readsav
import numpy as np
import datetime
import pickle as p
import pandas as pd

import multiprocess as mp # ing as mp

from heliosat.util import sanitize_dt

import time

from coreweb.dashcore.utils.utils import get_rt_data, cart2sphere, sphere2cart, getbodytraces, get_posdata, load_pos_data, round_to_hour_or_half, get_iparams_live, plot_body3d,process_coordinates, load_body_data, load_cat_id, get_archivedata, get_insitudata
import coreweb.dashcore.utils.heliocats as hc
import coreweb.dashcore.utils.twist_plot as tp
from coreweb.dashcore.utils.plotting import *

import coreweb

from coreweb.methods.method import BaseMethod
from coreweb.methods.data import FittingData
from coreweb.methods.abc_smc import abc_smc_worker

import copy

manager = mp.Manager()
processes = []
        
        
import os  
from PIL import Image

import plotly.io as pio

import pickle

def signaturecheckfull(many_lats, many_lons, model_obj, rinput, savedir, checkanimany, graphstore, t_launch):
    fontsize = 22  # Define the fontsize variable
    gridline_color = 'rgba(0, 0, 0, 0.3)'  # Define the color of gridlines
    trace_width = 4  # Define the width of traces
    
    onward = 3000 #2800 
    toward = -1000 if len(many_lons) > 4 else -4000

    names = ['Br', 'Bt', 'Bn', 'Btot']
    line_colors =['#c20078','#f97306', '#069af3', '#000000']
    eventshade = "LightSalmon"
    framecolor = 'rgba(100, 100, 100, 0.8)'

    total_height = len(many_lats) * 200  
    total_width = len(many_lons) * 400

    subplot_data = {}

    # Create a new figure for each longitude with increased height
    fig = make_subplots(rows=len(many_lats), cols=len(many_lons), shared_xaxes=True, subplot_titles=[f'Longitude: {lon}° <br> ' for lon in many_lons],
                        vertical_spacing=0.04,
                        horizontal_spacing=0.04, row_heights=[total_height // len(many_lats)] * len(many_lats))
    
    fig.update_annotations(font_size=fontsize)  # Update annotation fontsize
    # Update layout for all subplots to plotly_white theme
    fig.update_layout(
        template='plotly_white',
        xaxis=dict(
            tickfont=dict(size=fontsize),
            gridcolor=gridline_color,
            zerolinecolor=gridline_color 
        ),
        yaxis=dict(
            tickfont=dict(size=fontsize),
            gridcolor=gridline_color,
            zerolinecolor=gridline_color 
        )
    )

    for k, lat in enumerate(many_lats):
        fig.update_yaxes(title_text=f"Latitude: {lat}° <br> <br> B [nT]", row=k + 1, col=1, title_font=dict(size=fontsize))  # Update yaxis title fontsize

    for j, lon in enumerate(many_lons):
        for k, lat in enumerate(many_lats):
            showlegend = (k == 0 and j == 0)

            #print(lon)

            if lon == 60:
                delta1 = 105
                delta2 = (graphstore['t_data'][toward] - t_launch).total_seconds() / 3600

            else:
                delta1 = (graphstore['t_data'][onward] - t_launch).total_seconds() / 3600
                delta2 = 130
            
            
            x,y,z = sphere2cart(float(rinput), np.deg2rad(-float(lat)+90), np.deg2rad(float(lon)))

            # Update subplot_data dictionary with data for each subplot
            subplot_key = f'Longitude: {lon}°, Latitude: {lat}°'
            subplot_data[subplot_key] = {
                'x': [],
                'y': [[] for _ in range(4)]  # 4 lists for 4 traces
            }

            # Add traces for each latitude as subplots
            pos_array = get_longmove_array(0, rinput, lon, lat, graphstore)
            outa = np.array(model_obj.simulator(graphstore['t_data'], pos_array), dtype=object)
            outa = np.squeeze(outa[0])
            rtn_bx, rtn_by, rtn_bz = hc.convert_HEEQ_to_RTN_mag(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], outa[:, 0], outa[:, 1], outa[:, 2])
            outa[:, 0], outa[:, 1], outa[:, 2] = rtn_bx, rtn_by, rtn_bz
            outa[outa==0] = np.nan

            # Update subplot_data with x and y data for each trace
            subplot_data[subplot_key]['x'] = graphstore['t_data'][onward:toward]
            subplot_data[subplot_key]['y'][0] = outa[:, 0][onward:toward]
            subplot_data[subplot_key]['y'][1] = outa[:, 1][onward:toward]
            subplot_data[subplot_key]['y'][2] = outa[:, 2][onward:toward]
            subplot_data[subplot_key]['y'][3] = np.sqrt(np.sum(outa**2, axis=1))[onward:toward]

            # Add traces
            for i in range(4):
                fig.add_trace(go.Scatter(x=subplot_data[subplot_key]['x'], y=subplot_data[subplot_key]['y'][i],
                                         line=dict(color=line_colors[i], width=trace_width), name=names[i], showlegend=showlegend), row=k+1, col=j+1)

            # Update layout
            fig.update_yaxes(row=k+1, col=j+1, range=[-60,60], tickvals=np.arange(-50, 60, 25), tickfont=dict(size=fontsize),gridcolor=gridline_color,zerolinecolor=gridline_color )
            fig.update_xaxes(row=k+1, col=j+1, range=[t_launch+datetime.timedelta(hours=delta1), t_launch+datetime.timedelta(hours=delta2)])

        fig.update_layout(showlegend=True, height=total_height, width=total_width,
            legend=dict(font=dict(size=fontsize)),
            xaxis=dict(tickfont=dict(size=fontsize)),  # Set x-axis grid color and hide zeroline
            yaxis=dict(tickfont=dict(size=fontsize))  # Set y-axis grid color
        )
        sampled_ticks = [t_launch + datetime.timedelta(hours=i * 24) for i in [3,5,7,9]]
        tick_labels = [("+" + str(int((i - t_launch).total_seconds()/3600)) + "h") for i in sampled_ticks]
        fig.update_xaxes(tickvals=sampled_ticks, ticktext=tick_labels, tickfont=dict(size=fontsize),gridcolor=gridline_color,zerolinecolor=gridline_color )
        #fig.show()    
                
    fig.show()    
    pio.write_image(fig, f"signatureplots/{savedir}.pdf")

    with open(f"signatureplots/{savedir}_data.pkl", "wb") as f:
        pickle.dump(subplot_data, f)

def signaturecheck(many_lats,many_lons,model_obj, rinput, savedir, checkanimany):
    
    onward = 1100 
    toward = -900
    names = ['Br', 'Bt', 'Bn']
    template = "none"  
    bg_color = 'rgba(0,0,0,0)'
    line_color = 'black'
    line_colors =['#c20078','#f97306', '#069af3', '#000000']
    eventshade = "LightSalmon"
    framecolor = 'rgba(100, 100, 100, 0.8)'
            
    for j, lon in enumerate(many_lons):
        # Create a new figure for each longitude
        total_height = len(many_lats) * 200  # Adjust this value as needed
        
        # Create a new figure for each longitude with increased height
        fig = make_subplots(rows=len(many_lats), cols=1, shared_xaxes=True, subplot_titles=[f'Latitude: {lat}°' for lat in many_lats],
                            vertical_spacing=0.02, row_heights=[total_height // len(many_lats)] * len(many_lats))
        
        for k, lat in enumerate(many_lats):

            showlegend = k == 0
            
            x,y,z = sphere2cart(float(rinput), np.deg2rad(-float(lat)+90), np.deg2rad(float(lon)))

            checkanimany.add_trace(
                go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode='markers', 
                    marker=dict(size=4, 
                                symbol='square',
                                color='red'),
                    showlegend=False,
                ), row=1, col=1)

            # Add traces for each latitude as subplots
            pos_array = get_longmove_array(0, rinput, lon, lat, graphstore)
            insitufig = make_subplots(rows=1, cols=1, specs=[[{"type": "xy"}]], subplot_titles=['Synthetic Spacecraft at ' + str(lon) + '° lon, ' + str(lat) + '° lat, '])
            outa = np.array(model_obj.simulator(graphstore['t_data'], pos_array), dtype=object)
            outa = np.squeeze(outa[0])
            rtn_bx, rtn_by, rtn_bz = hc.convert_HEEQ_to_RTN_mag(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], outa[:, 0], outa[:, 1], outa[:, 2])
            outa[:, 0], outa[:, 1], outa[:, 2] = rtn_bx, rtn_by, rtn_bz
            outa[outa==0] = np.nan
            insitufig.add_trace(
                go.Scatter(
                    x=graphstore['t_data'][onward:toward],
                    y=outa[:, 0][onward:toward],
                    line=dict(color=line_colors[0], width=3, dash='dot'),
                    name=names[0],
                    showlegend=showlegend,
                ),
                row=1, col=1
            )

            insitufig.add_trace(
                go.Scatter(
                    x=graphstore['t_data'][onward:toward],
                    y=outa[:, 1][onward:],
                    line=dict(color=line_colors[1], width=3, dash='dot'),
                    name=names[1],
                    showlegend=showlegend,
                ),
                row=1, col=1
            )

            insitufig.add_trace(
                go.Scatter(
                    x=graphstore['t_data'][onward:toward],
                    y=outa[:, 2][onward:],
                    line=dict(color=line_colors[2], width=3, dash='dot'),
                    name=names[2],
                    showlegend=showlegend,
                ),
                row=1, col=1
            )

            insitufig.add_trace(
                go.Scatter(
                    x=graphstore['t_data'][onward:toward],
                    y=np.sqrt(np.sum(outa**2, axis=1))[onward:toward],
                    line=dict(color=line_colors[3], width=3, dash='dot'),
                    name='Btot',
                    showlegend=showlegend,
                ),
                row=1, col=1
            )

            insitufig.update_yaxes(title_text='B [nT]', row=1, col=1, range = [-60,60])
            #insitufig.update_yaxes(showgrid=True, zeroline=False, showticklabels=True,
            #                showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
            #                spikethickness=1, row=1, col=1)

            # Halve the font size of subplot titles
            #insitufig.update_layout(font=dict(size=6))
            
            # Add each latitude plot as a subplot to the main figure
            for trace in insitufig.data:
                fig.add_trace(trace, row=k+1, col=1)

            fig.update_yaxes(title_text='B [nT]', row=k+1, col=1 ) #, range = [-60,60], tickvals=np.arange(-60, 80, 20))
            fig.update_annotations(font_size=6, row=k+1, col=1)
        # Update layout for the main figure
        fig.update_layout(title=f'Longitude: {lon}°' , showlegend=True, height= total_height)

        # Sample a subset of time steps for tick labels
        sampled_ticks = [t_launch + datetime.timedelta(hours = i * 36) for i in range (5)]

        # Update x-axis tick labels
        tick_labels = [("+ " + str(int((i - t_launch).total_seconds()/3600)) + " h") for i in sampled_ticks]

        fig.update_xaxes(tickvals=sampled_ticks, ticktext=tick_labels)
                
        # Show the figure for each longitude
        # Save the figure for each longitude
        pio.write_image(fig, f"signatureplots/{savedir}_longitude_{lon}.png")
                
                

    #checkanimany.show()

def generate_figures_and_data(t_data, filtered_points, model_obj, lineofsightvector, lineofsightnorm, lineofsightunit):
    lineofsightfig = make_subplots(rows=1, cols=1, specs=[[{"type": "xy"}]], subplot_titles=['Btot as measured by synthetic Spacecraft along the Line of Sight in HEEQ'])
    lineofsightfig.update_yaxes(title_text='B [nT]', row=1, col=1)

    lineofsightfigx = make_subplots(rows=1, cols=1, specs=[[{"type": "xy"}]], subplot_titles=['Bx as measured by synthetic Spacecraft along the Line of Sight in HEEQ'])
    lineofsightfigx.update_yaxes(title_text='B [nT]', row=1, col=1)

    lineofsightfig2 = make_subplots(rows=1, cols=1, specs=[[{"type": "xy"}]], subplot_titles=['|B_parallel| as measured by synthetic Spacecraft along the Line of Sight'])
    lineofsightfig2.update_yaxes(title_text='|B| [nT]', row=1, col=1)

    lineofsightfig3 = make_subplots(rows=1, cols=1, specs=[[{"type": "xy", "secondary_y": True}]], subplot_titles=['Number of contributing Spacecraft and Summation over Bparallel'])
    lineofsightfig3.update_yaxes(title_text='|B| [nT]', row=1, col=1)

    total_projected_b = np.zeros(len(t_data))
    total_contribute = np.zeros(len(t_data))

    for i, synspace in enumerate(filtered_points):
        pos_array = np.empty((len(t_data), 3))
        pos_array[:, 0], pos_array[:, 1], pos_array[:, 2] = filtered_points[i][0], filtered_points[i][1], filtered_points[i][2]
        outa = np.array(model_obj.simulator(t_data, pos_array), dtype=object)
        outa = np.squeeze(outa[0])

        scalarproduct = np.dot(outa, lineofsightvector)
        projected_b = scalarproduct / lineofsightnorm
        b_parallel = projected_b.reshape(-1, 1) * lineofsightunit
        x_sign = np.sign(b_parallel[:, 0])
        b_parallel_tot = np.sqrt(np.sum(b_parallel ** 2, axis=1)) * x_sign

        contribute = np.zeros_like(b_parallel_tot)
        contribute[b_parallel_tot != 0] = 1

        total_projected_b += b_parallel_tot
        total_contribute += contribute

        outa[outa == 0] = np.nan

        lineofsightfig.add_trace(go.Scatter(x=np.array(t_data), y=np.sqrt(np.sum(outa ** 2, axis=1)), name='Btot_synth_' + str(i)), row=1, col=1)

        lineofsightfigx.add_trace(go.Scatter(x=np.array(t_data), y=outa[:, 0], name='Bx_synth_' + str(i)), row=1, col=1)

        b_parallel[b_parallel == 0] = np.nan

        lineofsightfig2.add_trace(go.Scatter(x=np.array(t_data), y=np.sqrt(np.sum(b_parallel ** 2, axis=1)), name='B_parallel_tot_' + str(i)), row=1, col=1)

    #lineofsightfig.show()
    #lineofsightfigx.show()
    #lineofsightfig2.show()

    lineofsightfig3.add_trace(go.Scatter(x=np.array(t_data), y=total_projected_b, name='Sum of B_parallel_synth'), row=1, col=1)
    lineofsightfig3.add_trace(go.Scatter(x=np.array(t_data), y=total_contribute, name='Number of measuring spacecraft'), row=1, col=1, secondary_y=True)
    lineofsightfig3.update_layout(yaxis=dict(title='Sum of B_parallel_synth [unit]', side='left'), yaxis2=dict(title='Number of measuring spacecraft [unit]', overlaying='y', side='right'))
    #lineofsightfig3.show()

    lineofsightfig4 = make_subplots(rows=1, cols=1, specs=[[{"type": "xy"}]], subplot_titles=['Average B_parallel per measuring spacecraft'])
    lineofsightfig4.update_yaxes(title_text='|B| [nT]', row=1, col=1)
    lineofsightfig4.add_trace(go.Scatter(x=np.array(t_data), y=total_projected_b / total_contribute, name='Sum of B_parallel_synth'), row=1, col=1)
    #lineofsightfig4.show()

    return lineofsightfig, lineofsightfigx, lineofsightfig2, lineofsightfig3, lineofsightfig4, total_projected_b, total_contribute

def ropechecker(iparams):
    paramlist = []

    highincflag = False

    if iparams['iparams']['t_factor']['default_value'] > 0:
        ropetype = "RH"
        rhparam = iparams['iparams']['t_factor']['default_value']
        lhparam = -rhparam
    else:
        ropetype = "LH"
        lhparam = iparams['iparams']['t_factor']['default_value']
        rhparam = -lhparam

    if iparams['iparams']['cme_inclination']['default_value'] > 90 and iparams['iparams']['cme_inclination']['default_value'] < 90 + 180:
        inc1 = iparams['iparams']['cme_inclination']['default_value']
        inc2 = inc1 + 180
        ropetype = ropetype + '-1'
    else:
        ropetype = ropetype + '-2'
        inc2 = iparams['iparams']['cme_inclination']['default_value']
        inc1 = inc2 + 180


   
    if inc1 > 360:
        inc1 = inc1 - 360
    if inc2 > 360:
        inc2 = inc2 - 360

    if inc1 == 45 or inc2 == 45:
        raise TypeError('Intermediate Type!')

    if inc1 > 45 and inc1 < 45 + 90:
        print('High Inclination!')
        highincflag = True

    if inc2 > 45 and inc2 < 45 + 90:
        print('High Inclination!')
        highincflag = True

    
    if ropetype == "RH-1":
        if highincflag == False:
            print('righthanded: SWN')
        else:
            print('righthanded: ESW')
    elif ropetype == "RH-2":
        if highincflag == False:
            print('righthanded: NES')
        else:
            print('righthanded: ENW')
        
    elif ropetype == "LH-1":
        if highincflag == False:
            print('lefthanded: NWS')
        else:
            print('lefthanded: WSE')
    else:
        if highincflag == False:
            print('lefthanded: SEN')
        print('lefthanded: WNE')


    ## RH - 1
    rh_1_params = copy.deepcopy(iparams)  # Perform a deep copy of iparams
    rh_1_params['iparams']['t_factor']['default_value'] = rhparam
    rh_1_params['iparams']['cme_inclination']['default_value'] = inc1
    paramlist.append(rh_1_params)

    ## RH - 2
    rh_2_params = copy.deepcopy(iparams)  # Perform a deep copy of iparams
    rh_2_params['iparams']['t_factor']['default_value'] = rhparam
    rh_2_params['iparams']['cme_inclination']['default_value'] = inc2
    paramlist.append(rh_2_params)

    ## LH - 1
    lh_1_params = copy.deepcopy(iparams)  # Perform a deep copy of iparams
    lh_1_params['iparams']['t_factor']['default_value'] = lhparam
    lh_1_params['iparams']['cme_inclination']['default_value'] = inc1
    paramlist.append(lh_1_params)

    ## LH - 2
    lh_2_params = copy.deepcopy(iparams)  # Perform a deep copy of iparams
    lh_2_params['iparams']['t_factor']['default_value'] = lhparam
    lh_2_params['iparams']['cme_inclination']['default_value'] = inc2
    paramlist.append(lh_2_params)

    return paramlist, highincflag

def ropeplotter(modelstatevars, timeslide):

    figs = []

    iparams = get_iparams_live(*modelstatevars)

    iparamlist, highincflag = ropechecker(iparams)
    if highincflag == False:
        fluxtypes = [
            ["SWN","",""], # RH-1
            ["NES","",""], # RH-2
            ["NWS","",""], # LH-1
            ["SEN","",""] # LH-2
            ]
    else:
        fluxtypes = [
            ["WNE","",""], # RH-1
            ["ESW","",""], # RH-2
            ["ENW","",""], # LH-1
            ["WSE","",""] # LH-2
            ]
    
    names = ['Br', 'Bt', 'Bn']
    
    modelvartypes = []

    specs = [
        [{"type": "scene", "rowspan": 2}, {"type": "scene", "rowspan": 2}],
        [None, None],
        [{"type": "xy", "colspan": 2}, None]
    ]

    template = "none"  
    bg_color = 'rgba(0,0,0,0)'
    line_color = 'black'
    line_colors =['#c20078','#f97306', '#069af3', '#000000']
    eventshade = "LightSalmon"
    framecolor = 'rgba(100, 100, 100, 0.8)'
    cmecolor = 'rgba(100, 100, 100, 0.8)'

    roundedlaunch = datetime.datetime(2012,12,21,6)
    insituend = roundedlaunch + datetime.timedelta(hours=100)
    resolution = datetime.timedelta(minutes=1)
    t_data = [roundedlaunch + i * resolution for i in range(int((insituend - roundedlaunch).total_seconds() / resolution.total_seconds()))]

    for i, fluxtype in enumerate(fluxtypes):

        print(iparamlist[i])
        ### general stuff

        fig = make_subplots(
            rows=3,
            cols=2,
            specs=specs,
            subplot_titles=fluxtype)
        
        

        model_obj = coreweb.ToroidalModel(roundedlaunch, **iparamlist[i]) # model gets initialized
        model_obj.generator()

        ### polar

        model_obj.propagator(roundedlaunch + datetime.timedelta(hours=timeslide))
            
        wf_model = model_obj.visualize_shape(iparam_index=0)  
        
        wf_array = np.array(wf_model)

        # Extract x, y, and z data from wf_array
        x = wf_array[:,:,0].flatten()
        y = wf_array[:,:,1].flatten()
        z = wf_array[:,:,2].flatten()

        # Create a 3D wireframe plot using plotly
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                        line=dict(width=1, color=cmecolor),
                        showlegend=False), row=1, col=1)

        # Transpose the wf_array to extract wireframe points along the other direction
        x_wire = wf_array[:,:,0].T.flatten()
        y_wire = wf_array[:,:,1].T.flatten()
        z_wire = wf_array[:,:,2].T.flatten()

        # Create another 3D wireframe plot using plotly
        fig.add_trace(go.Scatter3d(x=x_wire, y=y_wire, z=z_wire, mode='lines',
                        line=dict(width=1, color=cmecolor),
                        showlegend=False), row=1, col=1)
        
        # Create data for the Sun
        sun_trace = go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=8, color='yellow'),
            name='Sun',
            showlegend=False,
        )

        fig.add_trace(sun_trace, row=1, col=1)

        rinput = 0.7
        lonput = 0
        latput = 0

        longmove_array = get_longmove_array(0, rinput,lonput,latput, None)

        x,y,z = longmove_array[60*int(timeslide)] #sphere2cart(float(rinput), np.deg2rad(-float(latput)+90), np.deg2rad(float(lonput)))
        fig.add_trace(
            go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers', 
                marker=dict(size=3, 
                            symbol='square',
                            color='red'),
                name="SYN",
                customdata=np.vstack((rinput, latput, lonput)).T,
                showlegend=False,
                legendgroup = '1',
            ), row=1, col=1)
        

        # Create data for concentrical circles
        circle_traces = []
        radii = [0.3, 0.5, 0.8]  # Radii for the concentrical circles
        for r in radii:
            theta = np.linspace(0, 2 * np.pi, 100)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.zeros_like(theta)
            circle_trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='gray'),
                showlegend=False,
                hovertemplate = None, 
                hoverinfo = "skip", 
            )
            fig.add_trace(circle_trace, row=1, col=1)

            # Add labels for the circles next to the line connecting Sun and Earth
            label_x = r  # x-coordinate for label position
            label_y = 0  # y-coordinate for label position
            label_trace = go.Scatter3d(
                x=[label_x], y=[label_y], z=[0],
                mode='text',
                text=[f'{r} AU'],
                textposition='middle left',
                textfont=dict(size=8),
                showlegend=False,
                hovertemplate = None, 
                hoverinfo = "skip", 
            )
            fig.add_trace(label_trace, row=1, col=1)

        
        
        
        
        # Create data for the AU lines and their labels
        num_lines = 8
        for i in range(num_lines):
            angle_degrees = -180 + (i * 45)  # Adjusted angle in degrees (-180 to 180)
            angle_radians = np.deg2rad(angle_degrees)
            x = [0, np.cos(angle_radians)]
            y = [0, np.sin(angle_radians)]
            z = [0, 0]
            au_line = go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=framecolor),
                name=f'{angle_degrees}°',
                showlegend=False,
                hovertemplate = None, 
                hoverinfo = "skip", 
            )
            fig.add_trace(au_line, row=1, col=1)

            # Add labels for the AU lines
            label_x = 1.1 * np.cos(angle_radians)
            label_y = 1.1 * np.sin(angle_radians)
            label_trace = go.Scatter3d(
                x=[label_x], y=[label_y], z=[0],
                mode='text',
                text=[f'+/{angle_degrees}°' if angle_degrees == -180 else f'{angle_degrees}°'],
                textposition='middle center',
                textfont=dict(size=8),
                showlegend=False,
                hovertemplate = None, 
                hoverinfo = "skip", 
            )
            fig.add_trace(label_trace, row=1, col=1)

        ran = 1.2

        # Set the layout
        fig.update_layout(
            template=template, 
            plot_bgcolor=bg_color,  # Background color for the entire figure
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title='', showspikes=False, range=[-ran, ran]),  # Adjust the range as needed
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title='', showspikes=False, range=[-ran, ran]),  # Adjust the range as needed
                zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title='', showspikes=False, range=[-ran, ran]),  # Adjust the range as needed
                aspectmode='cube',
                camera=dict(eye=dict(x=0.7, y=0, z=0.5)),
                bgcolor=bg_color,
            ),
        )

        import base64

        strname = fluxtype[0] + '.png'

        if highincflag == False:
            # Load the PNG image into the top-right subplot
            with open(strname, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('ascii')
                img = go.layout.Image(
                    source='data:image/png;base64,' + image_data,
                    xref="paper", yref="paper",
                    x=0.9, y=0.55,
                    sizex=0.25, sizey=0.25,
                    xanchor="right", yanchor="bottom"
                )
                fig.add_layout_image(img)
        else:
            # Load the PNG image into the top-right subplot
            with open(strname, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('ascii')
                img = go.layout.Image(
                    source='data:image/png;base64,' + image_data,
                    xref="paper", yref="paper",
                    x=0.7, y=0.55,
                    sizex=0.5, sizey=0.5,
                    xanchor="right", yanchor="bottom"
                )
                fig.add_layout_image(img)

        
        
        outa = np.array(model_obj.simulator(t_data, longmove_array), dtype=object)

        outa = np.squeeze(outa[0])

        rtn_bx, rtn_by, rtn_bz = hc.convert_HEEQ_to_RTN_mag(longmove_array[:, 0], longmove_array[:, 1], longmove_array[:, 2], outa[:, 0],outa[:, 1],outa[:, 2])
        outa[:, 0],outa[:, 1],outa[:, 2] = rtn_bx, rtn_by, rtn_bz

        outa[outa==0] = np.nan

        fig.add_trace(
            go.Scatter(
                x=t_data,
                y=outa[:, 0],
                line=dict(color=line_colors[0], width=3, dash='dot'),
                name=names[0],
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=t_data,
                y=outa[:, 1],
                line=dict(color=line_colors[1], width=3, dash='dot'),
                name=names[1],
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=t_data,
                y=outa[:, 2],
                line=dict(color=line_colors[2], width=3, dash='dot'),
                name=names[2]+'',
            ),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=t_data,
                y=np.sqrt(np.sum(outa**2, axis=1)),
                line=dict(color=line_colors[3], width=3, dash='dot'),
                name='Btot',
            ),
            row=3, col=1
        )

        # Sample a subset of time steps for tick labels
        max_ticks = 10  # Adjust the number of ticks as needed
        sampled_ticks = t_data[::len(t_data) // max_ticks]

        # Update x-axis tick labels
        tick_labels = [("+ " + str(int((i - roundedlaunch).total_seconds()/3600)) + " h") for i in sampled_ticks]

        fig.update_xaxes(tickvals=sampled_ticks, ticktext=tick_labels, row=3, col=1)



        fig.update_yaxes(title_text='B [nT]', row=3, col=1) #, range=[min_b_data - y_range_padding, max_b_data + y_range_padding])
        fig.update_yaxes(showgrid=True, zeroline=False, showticklabels=True,
                         showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
                         spikethickness=1, row=3, col=1)
        fig.update_xaxes(showgrid=True, zeroline=False, showticklabels=True, rangeslider_visible=False,
                         showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
                         spikethickness=1, row=3, col=1)


        figs.append(fig)

    return figs

def get_eventinfo(cat_event, purelysyn = False, custom = False, loaded = False):


    if custom == True:

        # Extract the date using regular expression
        date_pattern = r'(\d{8})'

        match = re.search(date_pattern, cat_event[0])
        if match:
            extracted_date = match.group(1)
            extracted_datetime = datetime.datetime.strptime(extracted_date, '%Y%m%d')
        else:
            match = re.search(date_pattern, cat_event)
            extracted_date = match.group(1)
            extracted_datetime = datetime.datetime.strptime(extracted_date, '%Y%m%d')

        input_datetime_formatted = extracted_datetime.strftime('%Y-%m-%dT%H:%M:%S')
        endtime_formatted = extracted_datetime.strftime('%Y-%m-%dT%H:%M:%S')

        sc = cat_event.split('_')[1]

        if sc == 'NOAA':
            sc = cat_event.split('_')[1] + '_' + cat_event.split('_')[2] #'NOAA_RTSW'
            #print(sc)


        eventinfo = {"processday": [input_datetime_formatted],
            "begin": [input_datetime_formatted],
            "end": [endtime_formatted],
            "sc": [sc],
            "id": [cat_event],
            "loaded": loaded,
            "changed": True
           }

    elif purelysyn == True:

        alldate = datetime.datetime(2012,12,21,6)

        input_datetime_formatted = alldate.strftime('%Y-%m-%dT%H:%M:%S')
        endtime_formatted = alldate.strftime('%Y-%m-%dT%H:%M:%S')

        eventinfo = {"processday": [input_datetime_formatted],
            "begin": [input_datetime_formatted],
            "end": [endtime_formatted],
            "sc": ['SYN'],
            "id": [cat_event],
            "loaded": loaded,
            "changed": True
           }
        
    else:

        url='https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v22.csv'
        icmecat=pd.read_csv(url)
        starttime = icmecat.loc[:,'icme_start_time']
        idds = icmecat.loc[:,'icmecat_id']
        sc = icmecat.loc[:, 'sc_insitu']
        endtime = icmecat.loc[:,'mo_end_time']
        mobegintime = icmecat.loc[:, 'mo_start_time']
        
        dateFormat="%Y/%m/%d %H:%M"
        begin = pd.to_datetime(starttime, format=dateFormat)
        mobegin = pd.to_datetime(mobegintime, format=dateFormat)
        end = pd.to_datetime(endtime, format=dateFormat)
        
        i = np.where(idds == cat_event)[0]
        
        # Format the datetime object
        input_datetime_formatted = mobegin.iloc[i[0]].strftime('%Y-%m-%dT%H:%M:%S') #%z')
        endtime_formatted = end.iloc[i[0]].strftime('%Y-%m-%dT%H:%M:%S') #%z')
                
        eventinfo = {"processday": [input_datetime_formatted],
                "begin": [input_datetime_formatted],
                "end": [endtime_formatted],
                "sc": [sc.iloc[i[0]]],
                "id": [idds.iloc[i[0]]],
                "loaded": loaded,
                "changed": True
            }
    return eventinfo

def get_uploaddata(data, filename, plushours):
    
    '''
    used to generate the insitudata for the graphstore from upload (app.py)
    '''

    #print(data)

    if isinstance(data, dict) and not data:

        try:
            data = np.array(p.load(open(filename, "rb" ))[0], dtype=[('time', 'O'), ('bx', '<f8'), ('by', '<f8'), ('bz', '<f8'), ('bt', '<f8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('r', '<f8'), ('lat', '<f8'), ('lon', '<f8')])
        except:
            data = np.array(p.load(open(filename, "rb" ))[0], dtype=[('time', 'O'), ('bx', '<f8'), ('by', '<f8'), ('bz', '<f8'), ('bt', '<f8'), ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'), ('vt', '<f8'), ('np', '<f8'), ('tp', '<f8'), ('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('r', '<f8'), ('lat', '<f8'), ('lon', '<f8')])

        #print(data)

        time = data['time']
        bx = data['bx']
        by = data['by']
        bz = data['bz']

        desired_length = len(time)
        AUkm=149597870.7   

        # Create an array with NaN values
        posdata = np.empty((desired_length, 3))

        posdata[:, 0], posdata[:, 1], posdata[:, 2] = data['x']/AUkm, data['y']/AUkm, data['z']/AUkm

        heeq_bx, heeq_by, heeq_bz = hc.convert_RTN_to_HEEQ_mag(posdata[:, 0], posdata[:, 1],  posdata[:, 2], bx, by, bz)
        b_HEEQ = np.column_stack((heeq_bx, heeq_by, heeq_bz))
        
        dt = time
        b_RTN = np.column_stack((bx, by, bz))
        pos = np.column_stack((posdata[:, 0], posdata[:, 1],  posdata[:, 2]))



    else:

        # Extract relevant fields
        time = data['time']
        bx = data['br']
        by = data['bt']
        bz = data['bn']
        
        if filename.startswith('so'):
            sc = "SOLO"
        elif filename.startswith('psp'):
            sc = "PSP"
        elif filename.startswith('st'):
            sc = "STEREO-A"
        elif filename.startswith('bepi'):
            sc = "BEPI"
        elif filename.startswith('pa'):
            sc = "SYN"
            
        # Check for archive path
        archivepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashcore/data/archive"))
        file = '/positions_psp_solo_sta_bepi_wind_planets_HEEQ_10min_degrees.p'
        try:
            datafile=p.load(open(archivepath + file, "rb" ) ) 
        except:
            try:
                print("No Archive available, searching Helioforecast")
                url = 'https://helioforecast.space/static/sync/insitu_python/positions_now.p'
                file = urllib.request.urlopen(url)
                datafile = p.load(file)
            except:
                datafile=None
                
                
            
        if filename.startswith('pa'):
            # Calculate the desired length
            desired_length = len(time)
            
            parts = filename.split('_')
            distance = int(parts[3].split('.')[0])*0.00465047
            direction = int(parts[1])

            # Create an array with NaN values
            posdata = np.empty((desired_length, 3))

            posdata[:, 0], posdata[:, 1], posdata[:, 2] = sphere2cart(distance, np.deg2rad(-0+90), np.deg2rad(direction))
            
            heeq_bx, heeq_by, heeq_bz = hc.convert_RTN_to_HEEQ_mag( posdata[:, 0], posdata[:, 1],  posdata[:, 2], bx, by, bz)
            b_HEEQ = np.column_stack((heeq_bx, heeq_by, heeq_bz))
            
            dt = time
            b_RTN = np.column_stack((bx, by, bz))
            pos = np.column_stack((posdata[:, 0], posdata[:, 1],  posdata[:, 2]))
            
        else:
            posdata = getarchivecoords(sc, begin= time[0]-datetime.timedelta(minutes = 10), end = time[-1], arrays =  '10T', datafile = datafile)
        
            heeq_bx, heeq_by, heeq_bz = hc.convert_RTN_to_HEEQ_mag(posdata.x,posdata.y, posdata.z, bx, by, bz)
            b_HEEQ = np.column_stack((heeq_bx, heeq_by, heeq_bz))
            
            dt = time
            b_RTN = np.column_stack((bx, by, bz))
            pos = np.column_stack((posdata.x, posdata.y, posdata.z))

    if plushours == None:
        pass
    else:
        print('Extending timeframe, padding NaNs, using last known position')

        # Padding NaNs and using the last known position
        extended_length = len(time) + int(plushours * 60)  # Convert plushours to minutes
        nan_array = np.full((int(plushours * 60), 3), np.nan)
        
        b_HEEQ = np.concatenate((b_HEEQ, nan_array))
        b_RTN = np.concatenate((b_RTN, nan_array))
        dt = np.concatenate((dt, np.array([dt[-1] + datetime.timedelta(minutes=i) for i in range(1, int(plushours * 60) + 1)])))
        pos = np.concatenate((pos, np.tile(pos[-1], (int(plushours * 60), 1))))
     
    return b_HEEQ, b_RTN, dt, pos

def process_sav(path):
    '''
    used to process uploaded cdf files
    '''
    
    sav_data = readsav(path)
    
    name = path.split('/')[-1]
    parts = name.split('_')
    distance = float(parts[3].split('.')[0]) *0.00465047
    direction = float(parts[1])
    
    
    date_format = '%Y/%m/%d %H:%M:%S.%f'
    starttime = datetime.datetime.strptime(sav_data.time[0].decode('utf-8'), date_format)
    endtime = datetime.datetime.strptime(sav_data.time[-1].decode('utf-8'), date_format)
    
    eventbegin = datetime.datetime.strptime(sav_data.mctime[0].decode('utf-8'), date_format)
    eventend = datetime.datetime.strptime(sav_data.mctime[-1].decode('utf-8'), date_format)
    
    filename = name[:-4] + '.pickle'
    
    time_int = []
    
    while starttime <= endtime:
        time_int.append(starttime)
        starttime += datetime.timedelta(minutes=30)
        
    ll = np.zeros(np.size(sav_data.ibx),dtype=[('time',object),('br', float),('bt', float),('bn', float)] )
    
    ll = ll.view(np.recarray)  
    
    ll.time = time_int[:np.size(sav_data.ibx)]
    ll.br = sav_data.ibx * sav_data.b0*1e9
    ll.bt = sav_data.iby * sav_data.b0*1e9
    ll.bn = sav_data.ibz * sav_data.b0*1e9
    
    endtime_formatted = eventend.strftime("%Y-%m-%dT%H:%M:%S%z")
    input_datetime_formatted = eventbegin.strftime("%Y-%m-%dT%H:%M:%S%z")

    dateFormat = "%Y%m%d"
    firstdate = datetime.datetime.strftime(eventbegin, dateFormat)
            
    eventinfo = {"processday": [input_datetime_formatted],
            "begin": [input_datetime_formatted],
            "end": [endtime_formatted],
            "sc": ["SYN"],
            "id": [f"ICME_SYN_{filename[:-7]}_{firstdate}"],
            "loaded": filename,
            "changed": True
           }
    
    
    return ll, direction, distance, eventinfo



def generate_graphstore(infodata, reference_frame, rawdata = None, plushours = None):
    
    posstore = {}
    newhash = infodata['id']

    if reference_frame == "HEEQ" or reference_frame == "GSM":
        names = ['Bx', 'By', 'Bz']
    elif reference_frame == "RTN": 
        names = ['Br', 'Bt', 'Bn']
    sc = infodata['sc'][0]
    begin = infodata['begin'][0]
    end = infodata['end'][0]

    dateFormat = "%Y-%m-%dT%H:%M:%S%z"
    dateFormat2 = "%Y-%m-%d %H:%M:%S"
    dateFormat3 = "%Y-%m-%dT%H:%M:%S"

    try:
        begin = datetime.datetime.strptime(begin, dateFormat2)
    except ValueError:
        try:
            begin = datetime.datetime.strptime(begin, dateFormat)
        except:
            try:
                begin = datetime.datetime.strptime(begin, dateFormat3)
            except:
                pass

    try:
        end = datetime.datetime.strptime(end, dateFormat2)
    except ValueError:
        try:
            end = datetime.datetime.strptime(end, dateFormat)
        except:
            try:
                end = datetime.datetime.strptime(end, dateFormat3)
            except:
                pass

    if sc == "SYN":
        insitubegin = begin - datetime.timedelta(hours=24)
        insituend = end + datetime.timedelta(hours=24)
    else:
        
        insitubegin = begin - datetime.timedelta(hours=24)
        insituend = end + datetime.timedelta(hours=24)
        
    # Check if the data file exists in the "data" folder
    data_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashcore/data", f"{newhash[0]}.pkl"))
    
    if os.path.exists(data_file_path) and not (sc == "NOAA_RTSW" or sc == "STEREO-A_beacon"): 
        # Load data from the file
        with open(data_file_path, 'rb') as file:             
            saved_data = p.load(file)
            b_data_HEEQ = saved_data['b_data_HEEQ']
            b_data_RTN = saved_data['b_data_RTN']
            t_data = saved_data['t_data']
            pos_data = saved_data['pos_data']
            bodytraces = saved_data['bodytraces']
            bodydata = saved_data['bodydata']
            posstore = saved_data['posstore']
            
            
            if reference_frame == "HEEQ":
                b_data = b_data_HEEQ
            else:
                b_data = b_data_RTN
            
            
            print('Data loaded from ' + data_file_path)
            
            

    else:
        if infodata['loaded'] is not False:
            print('getting uploaded data')
            b_data_HEEQ, b_data_RTN, t_data, pos_data = get_uploaddata(rawdata, infodata['loaded'], plushours)
        else:
            try:
                if sc == "NOAA_ARCHIVE":
                     raise Exception("")
                b_data_HEEQ, b_data_RTN, t_data, pos_data = get_archivedata(sc, insitubegin, insituend)

                if len(b_data_HEEQ) == 0:
                    raise Exception("Data not contained in Archive")

                print("Data loaded from Data Archive")

            except Exception as e:

                try:
                    if sc == "SYN":
                        print('Skipping insitu data')
                        # Define the time resolution as 1 minute
                        resolution = datetime.timedelta(minutes=1)
                        t_data = [insitubegin + i * resolution for i in range(int((insituend - insitubegin).total_seconds() / resolution.total_seconds()))]

                        # Calculate the desired length
                        desired_length = len(t_data)

                        # Create an array with NaN values
                        nan_array = np.empty((desired_length, 3))
                        nan_array[:] = np.nan

                        b_data_HEEQ = nan_array
                        b_data_RTN = nan_array
                        b_data = nan_array
                        pos_data = np.empty((desired_length, 3))
                        pos_data[:] = 0.5

                    elif (sc == "NOAA_RTSW") or (sc=="NOAA_ARCHIVE") or (sc == "STEREO-A_beacon"):
                        print('Loading realtime data...')
                        b_data_HEEQ, b_data_RTN, b_data_GSM, t_data, pos_data = get_rt_data(sc, insitubegin, insituend, plushours)
                        if len(b_data_HEEQ) == 0:
                            raise Exception("Data not contained in Archive")
                        print('Realtime insitu data obtained successfully')

                    else:

                        print("Consider downloading Data Archive: ", e)

                        print("Starting automatic download via HelioSat...")

                        b_data_HEEQ, b_data_RTN, t_data, pos_data = get_insitudata(sc, insitubegin, insituend)

                        print('Insitu data obtained successfully')

                except Exception as e:
                    print("An error occurred:", e)
                    return {}, {}, {}

        if reference_frame == "HEEQ":
            b_data = b_data_HEEQ
        elif reference_frame == "GSM":
            b_data = b_data_GSM
        else:
            b_data = b_data_RTN
          
            
        # Check for archive path
        archivepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashcore/data/archive"))
        file = '/positions_psp_solo_sta_bepi_wind_planets_HEEQ_10min_degrees.p'

        try:
            datafile=p.load(open(archivepath + file, "rb" ) ) 
        except Exception as e:
            try:
                print("No Archive available, searching Helioforecast: ", e)
                url = 'https://helioforecast.space/static/sync/insitu_python/positions_now.p'
                file = urllib.request.urlopen(url)
                datafile = p.load(file)
            except:
                datafile=None
                
        try:
            bodytraces = getbodytraces("HEEQ", sc, infodata['processday'][0], datafile = datafile)
            print('Body traces obtained successfully')
        except Exception as e:
            bodytraces = None
            print('Failed to load body traces: ', e)


        # Extract the date using regular expression
        date_pattern = r'(\d{8})'

        match = re.search(date_pattern, newhash[0])
        if match:
            extracted_date = match.group(1)
            extracted_datetime = datetime.datetime.strptime(extracted_date, '%Y%m%d')
        else:
            match = re.search(date_pattern, newhash)
            extracted_date = match.group(1)
            extracted_datetime = datetime.datetime.strptime(extracted_date, '%Y%m%d')



        try:
            bodydata = load_body_data("HEEQ", extracted_datetime, datafile=datafile)
            print('Body data obtained successfully')
        except Exception as e:
            bodydata = None
            print('Failed to load body data: ', e)


        scs = ["SOLO", "PSP", "BEPI", "STEREO-A"]
        failcount = 0

        for scc in scs:
            if failcount < 5:
                
                try:

                    traces, types = get_posdata('HEEQ', scc, extracted_datetime, datafile=datafile)
                    
                    data, types = load_pos_data('HEEQ', scc, extracted_datetime, datafile=datafile)
                    # Update posstore using a dictionary
                    traj_data = {scc: {'traces': traces, 'data': data}}
                    print("Successfully loaded data for " + scc + " from " + types)
                    if posstore == None:
                        posstore = traj_data
                    else:
                        posstore.update(traj_data)

                except Exception as e:
                    failcount +=1
                    print("Failed to load data for ", scc, ":", e)
            else:
                print('Quit data retrieval to avoid overload')
                
        


        # Save obtained data to the file
        saved_data = {
            'b_data_HEEQ': b_data_HEEQ,
            'b_data_RTN': b_data_RTN,
            't_data': t_data,
            'bodytraces': bodytraces,
            'bodydata': bodydata,
            'pos_data': pos_data,
            'posstore': posstore,
        }
        
        
        
        if not (sc == "NOAA_RTSW" or sc == "STEREO-A_beacon"  or sc == "NOAA_ARCHIVE"):
            with open(data_file_path, 'wb') as file:
                p.dump(saved_data, file)
        else:
            # Save obtained data to the file
            saved_data = {
                'b_data_HEEQ': b_data_HEEQ,
                'b_data_RTN': b_data_RTN,
                'b_data_GSM': b_data_GSM,
                't_data': t_data,
                'bodytraces': bodytraces,
                'bodydata': bodydata,
                'pos_data': pos_data,
                'posstore': posstore,
            }
            # Get the current date and time as a string
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            # Append the current time to the file name
            file_name, file_extension = os.path.splitext(data_file_path)
            updated_file_path = f"{file_name}_{current_time}{file_extension}"

            with open(updated_file_path, 'wb') as file:
                p.dump(saved_data, file)
            
            
    try:
        view_legend_insitu = True
        fig = plot_insitu(names, t_data, b_data, view_legend_insitu) 
    except Exception as e:
        print("An error occurred:", e)
        return {}, {},{}
    
    
    if not (sc == "NOAA_RTSW" or sc == "STEREO-A_beacon"):
        return {'fig' : fig, 'b_data_HEEQ': b_data_HEEQ, 'b_data_RTN': b_data_RTN, 't_data': t_data, 'pos_data': pos_data, 'names': names, 'bodytraces': bodytraces, 'bodydata': bodydata}, posstore, {}
    else:
        return {'fig' : fig, 'b_data_HEEQ': b_data_HEEQ, 'b_data_RTN': b_data_RTN, 'b_data_GSM': b_data_GSM, 't_data': t_data, 'pos_data': pos_data, 'names': names, 'bodytraces': bodytraces, 'bodydata': bodydata}, posstore, {}




def update_posfig_offweb(posstore, rinput, lonput, latput, togglerange, timeslider, dim, graph, infodata, launchlabel,plotoptions, spacecraftoptions, bodyoptions, refframe, *modelstatevars):
    
    if launchlabel == "Launch Time:":
        raise PreventUpdate

    marks = {i: '+' + str(i)+'h' for i in range(0, 169, 12)}
    
    if "Catalog Event" in plotoptions:
        sc = infodata['sc'][0]
        begin = infodata['begin'][0]
        end = infodata['end'][0]

        if infodata['id'][0] == 'I':
            opac = 0
        else:
            opac = 0.5

        dateFormat = "%Y-%m-%dT%H:%M:%S%z"
        dateFormat2 = "%Y-%m-%d %H:%M:%S"
        dateFormat3 = "%Y-%m-%dT%H:%M:%S"

        try:
            begin = datetime.datetime.strptime(begin, dateFormat2)
        except ValueError:
            try:
                begin = datetime.datetime.strptime(begin, dateFormat)
            except:
                try:
                    begin = datetime.datetime.strptime(begin, dateFormat3)
                except:
                    pass

        try:
            end = datetime.datetime.strptime(end, dateFormat2)
        except ValueError:
            try:
                end = datetime.datetime.strptime(end, dateFormat)
            except:
                try:
                    end = datetime.datetime.strptime(end, dateFormat3)
                except:
                    pass
                
    
        
    
    datetime_format = "Launch Time: %Y-%m-%d %H:%M"
    t_launch = datetime.datetime.strptime(launchlabel, datetime_format)
    roundedlaunch = round_to_hour_or_half(t_launch) 
                
    if "Synthetic Event" in plotoptions:
        iparams = get_iparams_live(*modelstatevars)
        model_obj = coreweb.ToroidalModel(roundedlaunch, **iparams) # model gets initialized
        model_obj.generator()
        
    
    
    ################################################################
    ############################ INSITU ############################
    ################################################################
                
    
    if (graph is {}) or (graph is None): 
        insitufig = {}
    
    else:
        insitufig = go.Figure(graph['fig'])
              
        
        if "Title" in plotoptions:
            insitufig.update_layout(title=infodata['id'][0]+'_'+refframe)
        
        if "Catalog Event" in plotoptions:
            insitufig.add_vrect(
                    x0=begin,
                    x1=end,
                    fillcolor="LightSalmon", 
                    opacity=opac,
                    layer="below",
                    line_width=0
            )
            
        if "Synthetic Event" in plotoptions:
            # Create ndarray with dtype=object to handle ragged nested sequences
            if sc == "SYN":
                try:
                    # Calculate the desired length
                    desired_length = len(graph['t_data'])

                    # Create an array with NaN values
                    pos_array = np.empty((desired_length, 3))

                    pos_array[:, 0], pos_array[:, 1], pos_array[:, 2] = sphere2cart(rinput, np.deg2rad(-latput+90), np.deg2rad(lonput))  
                    
                    #print(pos_array)

                    outa = np.array(model_obj.simulator(graph['t_data'], pos_array), dtype=object)
                    
                    
                
                except:
                    outa = np.array(model_obj.simulator(graph['t_data'], graph['pos_data']), dtype=object)
            else:
                outa = np.array(model_obj.simulator(graph['t_data'], graph['pos_data']), dtype=object)
                
                
                #print(graph['pos_data'])
            
            outa = np.squeeze(outa[0])
            
            if sc == "SYN":
                if refframe == "RTN":
                    rtn_bx, rtn_by, rtn_bz = hc.convert_HEEQ_to_RTN_mag(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], outa[:, 0],outa[:, 1],outa[:, 2])
                    outa[:, 0],outa[:, 1],outa[:, 2] = rtn_bx, rtn_by, rtn_bz
            else:
                if refframe == "RTN":
                    x,y,z = hc.separate_components(graph['pos_data'])
                    rtn_bx, rtn_by, rtn_bz = hc.convert_HEEQ_to_RTN_mag(x,y,z, outa[:, 0],outa[:, 1],outa[:, 2])
                    outa[:, 0],outa[:, 1],outa[:, 2] = rtn_bx, rtn_by, rtn_bz
                        
            
            outa[outa==0] = np.nan

            names = graph['names']

            insitufig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=outa[:, 0],
                    line=dict(color='red', width=3, dash='dot'),
                    name=names[0]+'_synth',
                )
            )

            insitufig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=outa[:, 1],
                    line=dict(color='green', width=3, dash='dot'),
                    name=names[1]+'_synth',
                )
            )

            insitufig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=outa[:, 2],
                    line=dict(color='blue', width=3, dash='dot'),
                    name=names[2]+'_synth',
                )
            )

            insitufig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=np.sqrt(np.sum(outa**2, axis=1)),
                    line=dict(color='black', width=3, dash='dot'),
                    name='Btot_synth',
                )
            )
        if dim == "3D":
            insitufig.add_vrect(
                x0=roundedlaunch + datetime.timedelta(hours=timeslider),
                x1=roundedlaunch + datetime.timedelta(hours=timeslider),
                line=dict(color="Red", width=.5),
                name="Current Time",  # Add label "Ref_A" for t_s
            )
        
    
    
    ################################################################
    ############################## 3D ##############################
    ################################################################
    
    fig = go.Figure()
    
    if dim == "3D":
        
        if "Synthetic Event" in plotoptions:
            model_obj.propagator(roundedlaunch + datetime.timedelta(hours=timeslider))
            
            wf_model = model_obj.visualize_shape(iparam_index=0)  
            
            wf_array = np.array(wf_model)

            # Extract x, y, and z data from wf_array
            x = wf_array[:,:,0].flatten()
            y = wf_array[:,:,1].flatten()
            z = wf_array[:,:,2].flatten()

            # Create a 3D wireframe plot using plotly
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                           line=dict(width=1, color='rgba(100, 100, 100, 0.8)'),
                           showlegend=False))

            # Transpose the wf_array to extract wireframe points along the other direction
            x_wire = wf_array[:,:,0].T.flatten()
            y_wire = wf_array[:,:,1].T.flatten()
            z_wire = wf_array[:,:,2].T.flatten()

            # Create another 3D wireframe plot using plotly
            fig.add_trace(go.Scatter3d(x=x_wire, y=y_wire, z=z_wire, mode='lines',
                           line=dict(width=1, color='rgba(100, 100, 100, 0.8)'),
                           showlegend=False))
            
            
        if "Catalog Event" in plotoptions:
            roundedbegin = round_to_hour_or_half(begin) 
            roundedend = round_to_hour_or_half(end)
            
            roundedbegin = roundedbegin.replace(tzinfo=None)
            roundedend = roundedend.replace(tzinfo=None)
            roundedlaunch = roundedlaunch.replace(tzinfo=None)
            
            minevent = (roundedbegin - roundedlaunch).total_seconds() / 3600
            maxevent = (roundedend - roundedlaunch).total_seconds() / 3600

            
            if (maxevent - minevent) < 12:
                maxevent = maxevent//12*13
                minevent = minevent//12*12
            
            marks = {
                i: {'label': '+' + str(i) + 'h', 'style': {'color': 'red'}}
                if minevent <= i <= maxevent
                else '+' + str(i) + 'h'
                for i in range(0, 169, 12)
            }

        if "Sun" in bodyoptions:

            # Create data for the Sun
            sun_trace = go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=8, color='yellow'),
                name='Sun'
            )

            fig.add_trace(sun_trace)

        if "Earth" in bodyoptions:

            # Create data for the Earth
            try:
                fig.add_trace(plot_body3d(graph['bodydata']['Earth']['data'], roundedlaunch + datetime.timedelta(hours=timeslider), 'mediumseagreen', 'Earth')[0])
            except Exception as e:
                print('Data for Earth not found: ', e)


            #earth_trace = go.Scatter3d(
            #    x=[1], y=[0], z=[0],
            #    mode='markers',
            #    marker=dict(size=4, color='mediumseagreen'),
            #    name='Earth'
            #)

            #fig.add_trace(earth_trace)
                        
        if "Mercury" in bodyoptions:
            try:
                fig.add_trace(plot_body3d(graph['bodydata']['Mercury']['data'], roundedlaunch + datetime.timedelta(hours=timeslider), 'slategrey', 'Mercury')[0])
            except Exception as e:
                print('Data for Mercury not found: ', e)
            
            
        if "Venus" in bodyoptions:
            try:
                fig.add_trace(plot_body3d(graph['bodydata']['Venus']['data'], roundedlaunch + datetime.timedelta(hours=timeslider), 'darkgoldenrod', 'Venus')[0])
            except Exception as e:
                print('Data for Venus not found: ', e)
            
        if "Mars" in bodyoptions:
            try:
                fig.add_trace(plot_body3d(graph['bodydata']['Mars']['data'], roundedlaunch + datetime.timedelta(hours=timeslider), 'red', 'Mars')[0])
            except Exception as e:
                print('Data for Mars not found: ', e)
            
        if spacecraftoptions is not None:
            for scopt in spacecraftoptions:
                #try:
                if scopt == "SYN":
                    #try:

                    x,y,z = sphere2cart(float(rinput), np.deg2rad(-float(latput)+90), np.deg2rad(float(lonput)))
                    fig.add_trace(
                        go.Scatter3d(
                            x=[x], y=[y], z=[z],
                            mode='markers', 
                            marker=dict(size=3, 
                                        symbol='square',
                                        color='red'),
                            name="SYN",
                            customdata=np.vstack((rinput, latput, lonput)).T,
                            showlegend=True,
                            hovertemplate="<b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                         + sc + "</extra>"
                        ))
                    #except:
                     #   pass

                else:
                    traces = process_coordinates(posstore[scopt]['data']['data'], roundedlaunch, roundedlaunch + datetime.timedelta(hours=timeslider), posstore[scopt]['data']['color'], scopt)                    


                    if "Trajectories" in plotoptions:
                        fig.add_trace(traces[0])
                        fig.add_trace(traces[1])

                    fig.add_trace(traces[2])
                #except Exception as e:
                #    print('Data for ' + scopt + ' not found: ', e)


        if "Longitudinal Grid" in plotoptions:
            # Create data for concentrical circles
            circle_traces = []
            radii = [0.3, 0.5, 0.8]  # Radii for the concentrical circles
            for r in radii:
                theta = np.linspace(0, 2 * np.pi, 100)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = np.zeros_like(theta)
                circle_trace = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='gray'),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(circle_trace)

                # Add labels for the circles next to the line connecting Sun and Earth
                label_x = r  # x-coordinate for label position
                label_y = 0  # y-coordinate for label position
                label_trace = go.Scatter3d(
                    x=[label_x], y=[label_y], z=[0],
                    mode='text',
                    text=[f'{r} AU'],
                    textposition='middle left',
                    textfont=dict(size=8),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(label_trace)

            
            
            
            
            # Create data for the AU lines and their labels
            num_lines = 8
            for i in range(num_lines):
                angle_degrees = -180 + (i * 45)  # Adjusted angle in degrees (-180 to 180)
                angle_radians = np.deg2rad(angle_degrees)
                x = [0, np.cos(angle_radians)]
                y = [0, np.sin(angle_radians)]
                z = [0, 0]
                au_line = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='gray'),
                    name=f'{angle_degrees}°',
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(au_line)

                # Add labels for the AU lines
                label_x = 1.1 * np.cos(angle_radians)
                label_y = 1.1 * np.sin(angle_radians)
                label_trace = go.Scatter3d(
                    x=[label_x], y=[label_y], z=[0],
                    mode='text',
                    text=[f'+/{angle_degrees}°' if angle_degrees == -180 else f'{angle_degrees}°'],
                    textposition='middle center',
                    textfont=dict(size=8),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(label_trace)
                
        if "Latitudinal Grid" in plotoptions:
            # Create data for concentrical circles
            circle_traces = []
            radii = [0.3, 0.5, 0.8]  # Radii for the concentrical circles
            for r in radii:
                theta = np.linspace(0, 1/2 * np.pi, 100)
                x = r * np.cos(theta)
                y = np.zeros_like(theta)
                z = r * np.sin(theta)
                circle_trace = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='gray'),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(circle_trace)

                # Add labels for the circles next to the line connecting Sun and Earth
                label_x = r  # x-coordinate for label position
                label_y = 0  # y-coordinate for label position
                label_trace = go.Scatter3d(
                    x=[0], y=[0], z=[r],
                    mode='text',
                    text=[f'{r} AU'],
                    textposition='middle left',
                    textfont=dict(size=8),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(label_trace)

            # Create data for the AU lines and their labels
            num_lines = 10
            for i in range(num_lines):
                angle_degrees = (i * 10)  # Adjusted angle in degrees (0 to 90)
                angle_radians = np.deg2rad(angle_degrees)
                x = [0, np.cos(angle_radians)]
                y = [0, 0]
                z = [0, np.sin(angle_radians)]
                au_line = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='gray'),
                    name=f'{angle_degrees}°',
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(au_line)

                # Add labels for the AU lines
                label_x = 1.1 * np.cos(angle_radians)
                label_y = 1.1 * np.sin(angle_radians)
                label_trace = go.Scatter3d(
                    x=[label_x], y=[0], z=[label_y],
                    mode='text',
                    text=[f'{angle_degrees}°'],
                    textposition='middle center',
                    textfont=dict(size=8),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(label_trace)

        if "Title" in plotoptions:
            fig.update_layout(title=str(roundedlaunch + datetime.timedelta(hours=timeslider)))
            fig.add_annotation(text="HEEQ", xref="paper", yref="paper", x=1.1, y=0.1, showarrow=False)
            
        if "Timer" in plotoptions:
            fig.add_annotation(text=f"t_launch + {timeslider} h", xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False)
            

        # Set the layout
        fig.update_layout(
            template="none", 
            plot_bgcolor='rgba(0,0,0,0)',  # Background color for the entire figure
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title = ''),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title = ''),
                zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title = '', range=[-1, 1]),  # Adjust the range as needed
                aspectmode='cube',
                
            bgcolor='rgba(0,0,0,0)',
            ),
        )
        
        
        
    ################################################################
    ############################## 2D ##############################
    ################################################################
    
    elif dim == "2D":
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])
        
        if "AU axis" in plotoptions:
            showticks = True
        else:
            showticks = False
            
            
        if "Parker Spiral" in plotoptions:
                
            res_in_days=1 #/48.
            AUkm=149597870.7   
            sun_rot=26.24
            theta=np.arange(0,180,0.01)
            omega=2*np.pi/(sun_rot*60*60*24) #solar rotation in seconds

            v=modelstatevars[9]/AUkm #km/s
            r0=695000/AUkm
            r=v/omega*theta+r0*7
            
            # Create Parker spiral traces
            for q in np.arange(0, 12):
                omega = 2 * np.pi / (sun_rot * 60 * 60)  # Solar rotation in radians per second
                r = v / omega * theta + r0 * 7
                trace = go.Scatterpolar(
                    r=r,
                    theta=-theta + (0 + (360 / sun_rot) * res_in_days + 360 / 12 * q),
                    mode='lines',
                    line=dict(width=1, color='rgba(128, 128, 128, 0.3)'),
                    showlegend=False,
                    hovertemplate="Parker Spiral" +
                    "<extra></extra>",
                )
                fig.add_trace(trace)
                
                
        if togglerange == 0:
            ticktext = [ '0°', '45°', '90°', '135°', '+/-180°', '-135°', '-90°', '-45°',]
        else:
            ticktext = [ '0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°',]
        
        
        fig.update_layout(
            template="seaborn",
            polar=dict(
                angularaxis=dict(
                    tickmode='array',  # Set tick mode to 'array'
                    tickvals=[0, 45, 90, 135, 180, 225, 270, 315],  # Specify tick values for angles
                    ticktext=ticktext,  # Specify tick labels
                    showticklabels=True,  # Show tick labels
                    #rotation=90  # Rotate tick labels
                ),
                radialaxis=dict(
                    tickmode='array',  # Set tick mode to 'array'
                    tickvals=[0.2,0.4, 0.6,0.8,1, 1.2],  # Provide an empty list to remove tick labels
                    ticktext=['0.2 AU', '0.4 AU', '0.6 AU', '0.8 AU', '1 AU', '1.2 AU'],  # Specify tick labels
                    tickfont=dict(size=10),
                    showticklabels=showticks,  # Hide tick labels
                    range=[0, 1.2]  # Adjust the range of the radial axis,
                )
            )
        )
        
        if spacecraftoptions is not None:
            for scopt in spacecraftoptions:
                try:
                    if scopt == "SYN":
                        fig.add_trace(
                            go.Scatterpolar(
                                r=[float(rinput)], 
                                theta=[lonput], 
                                mode='markers', 
                                marker=dict(size=8, symbol='square', color='red'), 
                                name="SYN",
                                showlegend=False, 
                                hovertemplate="%{r:.1f} AU<br>%{theta:.1f}°<extra>" + "SYN" + "</extra>"
                            )
                        )
                    else:            
                        if "Trajectories" in plotoptions:
                            fig.add_trace(posstore[scopt]['traces'][0])
                            fig.add_trace(posstore[scopt]['traces'][1])

                        fig.add_trace(posstore[scopt]['traces'][2])
                except Exception as e:
                    print('Data for ' + scopt + ' not found: ', e)

        if "Sun" in bodyoptions:
            # Add the sun at the center
            fig.add_trace(go.Scatterpolar(r=[0], theta=[0], mode='markers', marker=dict(color='yellow', size=10, line=dict(color='black', width=1)), showlegend=False, hovertemplate="%{r:.1f} AU<br>%{theta:.1f}°"+
                    "<extra></extra>"))
            # Add label "Sun" next to the sun marker
            fig.add_trace(go.Scatterpolar(r=[0.03], theta=[15], mode='text', text=['Sun'],textposition='top right', showlegend=False, hovertemplate = None, hoverinfo = "skip", textfont=dict(color='black', size=14)))


        if "Earth" in bodyoptions:# Add Earth at radius 1
            fig.add_trace(go.Scatterpolar(r=[1], theta=[0], mode='markers', marker=dict(color='mediumseagreen', size=10), showlegend=False, hovertemplate="%{r:.1f} AU<br>%{theta:.1f}°"+
                    "<extra></extra>"))
            fig.add_trace(go.Scatterpolar(r=[1.03], theta=[1], mode='text', text=['Earth'],textposition='top right', name = 'Earth', showlegend=False, hovertemplate = None, hoverinfo = "skip",  textfont=dict(color='mediumseagreen', size=14)))
        
        
        try:
            if "Mercury" in bodyoptions:
                fig.add_trace(graph['bodytraces'][0][0])
                fig.add_trace(graph['bodytraces'][1])
            if "Venus" in bodyoptions:
                fig.add_trace(graph['bodytraces'][2][0])
                fig.add_trace(graph['bodytraces'][3])

            if "Mars" in bodyoptions:
                fig.add_trace(graph['bodytraces'][4][0])
                fig.add_trace(graph['bodytraces'][5])
        except:
            pass
            
            
        if "Title" in plotoptions:
            try:
                titledate = datetime.datetime.strptime(infodata['processday'][0], "%Y-%m-%dT%H:%M:%S%z")
            except:
                try:
                    titledate = datetime.datetime.strptime(infodata['processday'][0], "%Y-%m-%d %H:%M:%S%z")
                except:
                    titledate = datetime.datetime.strptime(infodata['processday'][0], "%Y-%m-%dT%H:%M:%S")
            datetitle = datetime.datetime.strftime(titledate, "%Y-%m-%d")
            fig.update_layout(title=datetitle)
            fig.add_annotation(text="HEEQ", xref="paper", yref="paper", x=1.1, y=-0.1, showarrow=False)
        

        # Adjust the subplot size
        fig.update_layout(height=800, width=800)
        
    
    return fig, {'display': 'block'}, insitufig, marks



def get_modelkwargs_ranges(fittingstate_values):


    ensemble_size = fittingstate_values[0]
        
    model_kwargs = {
        "ensemble_size": ensemble_size, #2**17
        "iparams": {
            "cme_longitude": {
                "maximum": fittingstate_values[1][1],
                "minimum": fittingstate_values[1][0]
            },
            "cme_latitude": {
                "maximum": fittingstate_values[2][1],
                "minimum": fittingstate_values[2][0]
            },
            "cme_inclination": {
                "distribution": "uniform",
                "maximum": fittingstate_values[3][1],
                "minimum": fittingstate_values[3][0]
            },
            "cme_diameter_1au": {
                "maximum": fittingstate_values[4][1],
                "minimum": fittingstate_values[4][0]
            },
            "cme_aspect_ratio": {
                "maximum": fittingstate_values[5][1],
                "minimum": fittingstate_values[5][0]
            },
            "cme_launch_radius": {
                "distribution": "uniform",
                "maximum": fittingstate_values[6][1],
                "minimum": fittingstate_values[6][0]
            },
            "cme_launch_velocity": {
                "maximum": fittingstate_values[7][1],
                "minimum": fittingstate_values[7][0]
            },
            "t_factor": {
                "maximum": fittingstate_values[11][1],
                "minimum": fittingstate_values[11][0],
            },
            "cme_expansion_rate": {
                "distribution": "uniform",
                "maximum": fittingstate_values[8][1],
                "minimum": fittingstate_values[8][0],
            },
            "magnetic_decay_rate": {
                "distribution": "uniform",
                "maximum": fittingstate_values[12][1],
                "minimum": fittingstate_values[12][0],
            },
            "magnetic_field_strength_1au": {
                "maximum": fittingstate_values[13][1],
                "minimum": fittingstate_values[13][0],
            },
            "background_drag": {
                "distribution": "uniform",
                "maximum": fittingstate_values[9][1],
                "minimum": fittingstate_values[9][0],
            },
            "background_velocity": {
                "distribution": "uniform",
                "maximum": fittingstate_values[10][1],
                "minimum": fittingstate_values[10][0],
            }
        }
    }
    
    for param, values in model_kwargs["iparams"].items():
        if values["maximum"] == values["minimum"]:
            values["distribution"] = "fixed"
            values["default_value"] = values["minimum"]
            del values["maximum"]
            del values["minimum"]
    
    return model_kwargs



def starmap(func, args):
    return [func(*_) for _ in args]

def save(path, extra_args):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        print('Created directory ' + path)
    
    with open(path, "wb") as fh:
        p.dump(extra_args, fh)
        print('Saved to ' + path)

def offwebfit(t_launch, eventinfo, graphstore, multiprocessing, t_s, t_e, t_fit, njobs, itermin, itermax, n_particles, *model_kwargs):
    
    output_file = None
    iter_i = 0 # keeps track of iterations
    hist_eps = [] # keeps track of epsilon values
    hist_time = [] # keeps track of time
    
    balanced_iterations = 3
    time_offsets = [0]
    eps_quantile = 0.25
    epsgoal = 0.25
    kernel_mode = "cm"
    random_seed = 42
    summary_type = "norm_rmse"
    fit_coord_system = 'HEEQ'
    sc = eventinfo['sc'][0]
    
    model_kwargs = model_kwargs[0]
    
    outputfile = eventinfo['id'][0]+'_HEEQ'
    current_datetime = datetime.datetime.now()
    current_time = current_datetime.strftime("%Y%m%d%H%M")
    outputpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashcore/output/"))
    
    if isinstance(outputfile, str):
        outputfilecode = outputpath +'/' + outputfile + "_" + current_time + "/"
    elif isinstance(outputfile, list) and len(outputfile) == 1 and isinstance(outputfile[0], str):
        outputfilecode = outputpath + outputfile[0] + "_" + current_time + "/"    
    
    
    
    base_fitter = BaseMethod()
    base_fitter.initialize(t_launch, coreweb.ToroidalModel, model_kwargs)
    base_fitter.add_observer(sc, t_fit, t_s, t_e)
    
    t_launch = sanitize_dt(t_launch)
    
    
    if multiprocessing == True:

        #global mpool
        mpool = mp.Pool(processes=njobs) # initialize Pool for multiprocessing
        processes.append(mpool)
    data_obj = FittingData(base_fitter.observers, fit_coord_system, graphstore)
    data_obj.generate_noise("psd",60)
   
    kill_flag = False
    pcount = 0
    timer_iter = None
    
    try:
        for iter_i in range(iter_i, itermax):
            # We first check if the minimum number of 
            # iterations is reached.If yes, we check if
            # the target value for epsilon "epsgoal" is reached.
            reached = False

            if iter_i >= itermin:
                if hist_eps[-1] < epsgoal:
                    print("Fitting terminated, target RMSE reached: eps < ", epsgoal)
                    kill_flag = True
                    break    
                    
            print("Running iteration " + str(iter_i))        
                    
            
            timer_iter = time.time()

            # correct observer arrival times

            if iter_i >= len(time_offsets):
                _time_offset = time_offsets[-1]
            else:
                _time_offset = time_offsets[iter_i]

            data_obj.generate_data(_time_offset)
            #print(data_obj.data_b)
            #print(data_obj.data_dt)
            #print(data_obj.data_o)
            #print('success datagen')


            if len(hist_eps) == 0:
                eps_init = data_obj.sumstat(
                    [np.zeros((1, 3))] * len(data_obj.data_b), use_mask=False
                )[0]
                # returns summary statistic for a vector of zeroes for each observer                
                hist_eps = [eps_init, eps_init * 0.98]
                #hist_eps gets set to the eps_init and 98% of it
                hist_eps_dim = len(eps_init) # number of observers
                
                print("Initial eps_init = ", eps_init)
                

                model_obj_kwargs = dict(model_kwargs)
                model_obj_kwargs["ensemble_size"] = n_particles
                model_obj = base_fitter.model(t_launch, **model_obj_kwargs) # model gets initialized
            sub_iter_i = 0 # keeps track of subprocesses 

            _random_seed = random_seed + 100000 * iter_i # set random seed to ensure reproducible results
            # worker_args get stored

            worker_args = (
                    iter_i,
                    t_launch,
                    base_fitter.model,
                    model_kwargs,
                    model_obj.iparams_arr,
                    model_obj.iparams_weight,
                    model_obj.iparams_kernel_decomp,
                    data_obj,
                    summary_type,
                    hist_eps[-1],
                    kernel_mode,
                )
            

            print("Starting simulations")

            if multiprocessing == True:
                print("Multiprocessing is used")
                _results = mpool.starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(njobs)]) # starmap returns a function for all given arguments
            else:
                print("Multiprocessing is not used")
                _results = starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(njobs)]) # starmap returns a function for all given arguments

            # the total number of runs depends on the ensemble size set in the model kwargs and the number of jobs
            total_runs = njobs * int(model_kwargs["ensemble_size"])  #
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
                epses_temp = np.zeros((pcount, hist_eps_dim), model_obj.dtype)
                for i in range(0, len(_results)):
                    particles_temp[
                        sum(pcounts[:i]) : sum(pcounts[: i + 1])
                    ] = _results[i][0] # results of current iteration are stored
                    epses_temp[sum(pcounts[:i]) : sum(pcounts[: i + 1])] = _results[
                        i
                    ][1] # errors of current iteration are stored
                    
                sys.stdout.flush()    
                print(f"Step {iter_i}:{sub_iter_i} with ({pcount}/{n_particles}) particles", end='\r')
                  # Flush the output buffer to update the line immediately


                if pcount > n_particles:
                    print(str(pcount) + ' reached particles                     ')
                    break
                # if ensemble size isn't reached, continue
                # random seed gets updated

                _random_seed = (
                    random_seed + 100000 * iter_i + 1000 * (sub_iter_i + 1)
                )

                if multiprocessing == True:
                    _results_ext = mpool.starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(njobs)]) # starmap returns a function for all given arguments
                else:
                    _results_ext = starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(njobs)]) # starmap returns a function for all given arguments

                _results.extend(_results_ext) #results get appended to _results
                sub_iter_i += 1
                # keep track of total number of runs
                total_runs += njobs * int(model_kwargs["ensemble_size"])  #

                if pcount == 0:
                    print("No hits, aborting                ")
                    kill_flag = True
                    break

            if kill_flag:
                break

            if pcount > n_particles: # no additional particles are kept
                particles_temp = particles_temp[:n_particles]

            # if we're in the first iteration, the weights and kernels have to be initialized. Otherwise, they're updated. 
            if iter_i == 0:
                model_obj.update_iparams(
                    particles_temp,
                    update_weights_kernels=False,
                    kernel_mode=kernel_mode,
                ) # replace iparams_arr by particles_temp
                model_obj.iparams_weight = (
                    np.ones((n_particles,), dtype=model_obj.dtype) / n_particles
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

                hist_eps.append(new_eps)
                
            elif isinstance(eps_quantile, list) or isinstance(
                eps_quantile, np.ndarray
            ):
                eps_quantile_eff = eps_quantile ** (1 / hist_eps_dim)  #
                _k = len(eps_quantile_eff)  #
                new_eps = np.array(
                    [
                        np.quantile(epses_temp, eps_quantile_eff[i], axis=0)[i]
                        for i in range(_k)
                    ]
                )
                hist_eps.append(new_eps)
                
            print(f"Setting new eps: {hist_eps[-2]} => {hist_eps[-1]}")
                

            hist_time.append(time.time() - timer_iter)
            
            print(
                f"Step {iter_i} done, {total_runs / 1e6:.2f}M runs in {time.time() - timer_iter:.2f} seconds, (total: {time.strftime('%Hh %Mm %Ss', time.gmtime(np.sum(hist_time)))})"
            )
            
            
            iter_i = iter_i + 1  # iter_i gets updated

            # save output to file 
            if outputfilecode:
                output_file = os.path.join(
                    outputfilecode, "{0:02d}.pickle".format(iter_i - 1)
                )

                extra_args = {"t_launch": t_launch,
                  "model_kwargs": model_kwargs,
                  "hist_eps": hist_eps,
                  "hist_eps_dim": hist_eps_dim,
                  "base_fitter": base_fitter,
                  "model_obj": model_obj,
                  "data_obj": data_obj,
                  "epses": epses_temp,
                 }

                save(output_file, extra_args)
    finally:
        for process in processes:
            process.terminate()
        pass
    
    
    return output_file



def create_movie(degmove, timeres, deltatime, longmove_array, results, plottheme, graphstore, reference_frame, rinput, lonput, latput, currenttimeslider, eventinfo, launchlabel, plot_options, spacecraftoptions, bodyoptions,  insitu, positions, view_legend_insitu, camera, posstore, *modelstatevars):

    if timeres == None:
        timeres = 30
        
    ks = int(deltatime* 60/timeres)

    deg_per = degmove / ks
    currentcam = camera


    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Format the datetime as a string
    current_time = current_datetime.strftime("%Y%m%d%H%M")

    #print(currentcam)

    path = 'src/coreweb/dashcore/temp/' + current_time + '/'

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    
    for i in range(0,ks):
        currentcam[2] = currentcam[2] + deg_per

        sys.stdout.flush()    
        print(f"{i}/{ks}", end='\r')

        timeslide = i*0.5

        fig = check_animation(longmove_array, results, plottheme, graphstore, reference_frame, rinput, lonput, latput, timeslide, eventinfo, launchlabel, plot_options, spacecraftoptions, bodyoptions,  insitu, positions, view_legend_insitu, currentcam.copy(), posstore, *modelstatevars)
        
        fig.write_image(path + "fig_" + str(i) + ".png")


    return



def extract_row(row):
    
    params = [
        row['Longitude'],
        row['Latitude'],
        row['Inclination'],
        row['Diameter 1 AU'],
        row['Aspect Ratio'],
        row['Launch Radius'],
        row['Launch Velocity'],
        row['Expansion Rate'],
        row['Background Drag'],
        row['Background Velocity'],
        row['T_Factor'],
        row['Magnetic Decay Rate'],
        row['Magnetic Field Strength 1 AU'],        
    ]

    return params





def allropeplotter(modelstatevars, timeslide):

    figs = []

    fontsize = 28  # Define the fontsize variable

    iparams = get_iparams_live(*modelstatevars)

    iparamlist, highincflag = ropechecker(iparams)
    if highincflag == False:
        fluxtypes = [
            ["SWN","",""], # RH-1
            ["NES","",""], # RH-2
            ["NWS","",""], # LH-1
            ["SEN","",""] # LH-2
            ]
        subplot_titles= ["Low Inclination Fluxrope Types","","",
                         "","","",
                         "","","",
                         "","",""]
        
    else:
        fluxtypes = [
            ["WNE","",""], # RH-1
            ["ESW","",""], # RH-2
            ["ENW","",""], # LH-1
            ["WSE","",""] # LH-2
            ]
        subplot_titles= ["High Inclination Fluxrope Types","","",
                         "","","",
                         "","","",
                         "","",""]
    
    names = ['Br', 'Bt', 'Bn']
    
    modelvartypes = []

    specs = [
        [{"type": "scene", "rowspan": 4},{"type": "scene"}, {"type": "xy"}],
        [None, {"type": "scene"}, {"type": "xy"}],
        [None, {"type": "scene"}, {"type": "xy"}],
        [None, {"type": "scene"}, {"type": "xy"}],
    ]

    fig = make_subplots(
            rows=4,
            cols=3,
            specs=specs,
            subplot_titles=subplot_titles,
            column_widths=[0.5, 0.25, 0.25],
            )

    template = "none"  
    bg_color = 'rgba(0,0,0,0)'
    line_color = 'black'
    line_colors =['#c20078','#f97306', '#069af3', '#000000']
    eventshade = "LightSalmon"
    framecolor = 'rgba(100, 100, 100, 0.8)'
    cmecolor = 'rgba(100, 100, 100, 0.8)'

    roundedlaunch = datetime.datetime(2012,12,21,6)
    insituend = roundedlaunch + datetime.timedelta(hours=100)
    resolution = datetime.timedelta(minutes=1)
    t_data = [roundedlaunch + i * resolution for i in range(int((insituend - roundedlaunch).total_seconds() / resolution.total_seconds()))]

    for i, fluxtype in enumerate(fluxtypes):

        #fig.show()

        model_obj = coreweb.ToroidalModel(roundedlaunch, **iparamlist[i]) # model gets initialized
        model_obj.generator()

        ### polar

        model_obj.propagator(roundedlaunch + datetime.timedelta(hours=timeslide))
            
        if i == 0:
            wf_model = model_obj.visualize_shape(iparam_index=0)  
        
            wf_array = np.array(wf_model)

            # Extract x, y, and z data from wf_array
            x = wf_array[:,:,0].flatten()
            y = wf_array[:,:,1].flatten()
            z = wf_array[:,:,2].flatten()

            # Create a 3D wireframe plot using plotly
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                            line=dict(width=1, color=cmecolor),
                            showlegend=False), row=1, col=1)

            # Transpose the wf_array to extract wireframe points along the other direction
            x_wire = wf_array[:,:,0].T.flatten()
            y_wire = wf_array[:,:,1].T.flatten()
            z_wire = wf_array[:,:,2].T.flatten()

            # Create another 3D wireframe plot using plotly
            fig.add_trace(go.Scatter3d(x=x_wire, y=y_wire, z=z_wire, mode='lines',
                            line=dict(width=1, color=cmecolor),
                            showlegend=False), row=1, col=1)
        
            # Create data for the Sun
            sun_trace = go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=8, color='yellow'),
                name='Sun',
                showlegend=False,
            )

            fig.add_trace(sun_trace, row=1, col=1)

            rinput = 0.7
            lonput = 0
            latput = 0

            longmove_array = get_longmove_array(0, rinput,lonput,latput, None)

            x,y,z = longmove_array[60*int(timeslide)] #sphere2cart(float(rinput), np.deg2rad(-float(latput)+90), np.deg2rad(float(lonput)))
            fig.add_trace(
                go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode='markers', 
                    marker=dict(size=3, 
                                symbol='square',
                                color='red'),
                    name="SYN",
                    customdata=np.vstack((rinput, latput, lonput)).T,
                    showlegend=False,
                    legendgroup = '1',
                ), row=1, col=1)
            

            # Create data for concentrical circles
            circle_traces = []
            radii = [0.3, 0.5, 0.8]  # Radii for the concentrical circles
            for r in radii:
                theta = np.linspace(0, 2 * np.pi, 100)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = np.zeros_like(theta)
                circle_trace = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color='gray'),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(circle_trace, row=1, col=1)

                # Add labels for the circles next to the line connecting Sun and Earth
                label_x = 0  # x-coordinate for label position
                label_y = r  # y-coordinate for label position
                label_trace = go.Scatter3d(
                    x=[label_x], y=[label_y], z=[0],
                    mode='text',
                    text=[f'{r} AU'],
                    textposition='middle left',
                    textfont=dict(size=fontsize),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                #fig.add_trace(label_trace, row=1, col=1)

            
            
            
            
            # Create data for the AU lines and their labels
            num_lines = 8
            for l in range(num_lines):
                angle_degrees = -180 + (l * 45)  # Adjusted angle in degrees (-180 to 180)
                angle_radians = np.deg2rad(angle_degrees)
                x = [0, np.cos(angle_radians)]
                y = [0, np.sin(angle_radians)]
                z = [0, 0]
                au_line = go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color=framecolor),
                    name=f'{angle_degrees}°',
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(au_line, row=1, col=1)

                # Add labels for the AU lines
                label_x = 1.1 * np.cos(angle_radians)
                label_y = 1.1 * np.sin(angle_radians)
                label_trace = go.Scatter3d(
                    x=[label_x], y=[label_y], z=[0],
                    mode='text',
                    text=[f'+/{angle_degrees}°' if angle_degrees == -180 else f'{angle_degrees}°'],
                    textposition='middle center',
                    textfont=dict(size=22),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(label_trace, row=1, col=1)

            ran = 1.2

            # Set the layout
            fig.update_annotations(font_size=fontsize)
            fig.update_layout(
                template=template, 
                plot_bgcolor=bg_color,  # Background color for the entire figure
                scene=dict(
                    xaxis=dict(tickfont=dict(size=fontsize),showticklabels=False, showgrid=False, zeroline=False, showline=False, title='', showspikes=False, range=[-ran, ran]),  # Adjust the range as needed
                    yaxis=dict(tickfont=dict(size=fontsize),showticklabels=False, showgrid=False, zeroline=False, showline=False, title='', showspikes=False, range=[-ran, ran]),  # Adjust the range as needed
                    zaxis=dict(tickfont=dict(size=fontsize),showticklabels=False, showgrid=False, zeroline=False, showline=False, title='', showspikes=False, range=[-ran, ran]),  # Adjust the range as needed
                    aspectmode='cube',
                    camera=dict(eye=dict(x=1.5, y=0, z=0.7)),
                    bgcolor=bg_color,
                ),
                height=1200,
                width=1800,
                legend=dict(font=dict(size=fontsize))
            )

        outa = np.array(model_obj.simulator(t_data, longmove_array), dtype=object)

        outa = np.squeeze(outa[0])

        rtn_bx, rtn_by, rtn_bz = hc.convert_HEEQ_to_RTN_mag(longmove_array[:, 0], longmove_array[:, 1], longmove_array[:, 2], outa[:, 0],outa[:, 1],outa[:, 2])
        outa[:, 0],outa[:, 1],outa[:, 2] = rtn_bx, rtn_by, rtn_bz

        outa[outa==0] = np.nan


        # Find the range of non-NaN values
        valid_indices = np.where(~np.isnan(outa[:, 0]))[0]
        min_index = valid_indices.min()
        max_index = valid_indices.max()

        show_legend = True if i == 0 else False

        fig.add_trace(
            go.Scatter(
                x=t_data[min_index:max_index + 1],  # Use the valid range of indices
                y=outa[min_index:max_index + 1, 0],
                line=dict(color=line_colors[0], width=3, #dash='dot'
                          ),
                name=names[0],
                showlegend=show_legend, 
            ),
            row=i + 1, col=3
        )

        fig.add_trace(
            go.Scatter(
                x=t_data[min_index:max_index + 1],  # Use the valid range of indices
                y=outa[min_index:max_index + 1, 1],
                line=dict(color=line_colors[1], width=3, #dash='dot'
                          ),
                name=names[1],
                showlegend=show_legend, 
            ),
            row=i + 1, col=3
        )

        fig.add_trace(
            go.Scatter(
                x=t_data[min_index:max_index + 1],  # Use the valid range of indices
                y=outa[min_index:max_index + 1, 2],
                line=dict(color=line_colors[2], width=3, #dash='dot'
                          ),
                name=names[2],
                showlegend=show_legend, 
            ),
            row=i + 1, col=3
        )

        fig.add_trace(
            go.Scatter(
                x=t_data[min_index:max_index + 1],  # Use the valid range of indices
                y=np.sqrt(np.sum(outa[min_index:max_index + 1]**2, axis=1)),
                line=dict(color=line_colors[3], width=3, #dash='dot'
                          ),
                name='Btot',
                showlegend=show_legend, 
            ),
            row=i + 1, col=3
        )

        # Sample a subset of time steps for tick labels
        max_ticks = 10  # Adjust the number of ticks as needed
        sampled_ticks = t_data[::len(t_data) // max_ticks]

        # Update x-axis tick labels
        tick_labels = [("+ " + str(int((i - roundedlaunch).total_seconds()/3600)) + " h") for i in sampled_ticks]

        fig.update_xaxes(tickvals=sampled_ticks, ticktext=tick_labels, row=i + 1, col=3, tickfont=dict(size=fontsize))



        fig.update_yaxes(title_text='B [nT]', row=i + 1, col=3, title_font=dict(size=fontsize)) #, range=[min_b_data - y_range_padding, max_b_data + y_range_padding])
        fig.update_yaxes(showgrid=True, zeroline=False, showticklabels=True,
                        showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
                        spikethickness=1, row=i + 1, col=3, tickfont=dict(size=fontsize))
        fig.update_xaxes(showgrid=True, zeroline=False, showticklabels=True, rangeslider_visible=False,
                        showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
                        spikethickness=1, row=i + 1, col=3, tickfont=dict(size=fontsize))
        
        



    return fig