import dash
from dash import dcc, html, Output, Input, State, callback, no_update
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import dash_bootstrap_components as dbc

import datetime
import functools

import pandas as pd
import pickle as p
import numpy as np

import os
import sys

import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns 

import matplotlib.dates as mdates

import astrospice
from astropy.time import Time, TimeDelta
import astropy.units as u
from sunpy.coordinates import HeliographicStonyhurst, HeliocentricEarthEcliptic
from astrospice.net.reg import RemoteKernel, RemoteKernelsBase

import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup

import heliosat

import coreweb.dashcore.utils.heliocats as hc
import coreweb.methods.data_frame_transforms as dft

from ...methods.method import BaseMethod

import cdflib
import base64
import io
import tempfile
import re
from sunpy.time import parse_time
from scipy.io import readsav

import matplotlib.pyplot as plt


import copy

def convert_HEE_to_HEEQ_single(obsdate_mjd, x, y, z):
    '''
    for Wind positions: convert HEE to HAE to HEEQ
    '''

    print('conversion HEE to HEEQ')                                
    
    jd = Time(obsdate_mjd + 2400000.5, format='mjd').jd
    mjd = obsdate_mjd

    w_hee = [x, y, z]

    # HEE to HAE        

    # define T00 and UT
    T00 = (mjd - 51544.5) / 36525.0          
    dobj = Time(obsdate_mjd, format='mjd')
    UT = dobj.datetime.hour + dobj.datetime.minute / 60. + dobj.datetime.second / 3600. # time in UT in hours   

    # lambda_sun in Hapgood, equation 5, here in rad
    M = np.radians(357.528 + 35999.050 * T00 + 0.04107 * UT)
    LAMBDA = 280.460 + 36000.772 * T00 + 0.04107 * UT        
    lambda_sun = np.radians((LAMBDA + (1.915 - 0.0048 * T00) * np.sin(M) + 0.020 * np.sin(2 * M)))

    # S-1 Matrix equation 12 hapgood 1992, change sign in lambda angle for inversion HEE to HAE instead of HAE to HEE
    c, s = np.cos(-(lambda_sun + np.radians(180))), np.sin(-(lambda_sun + np.radians(180)))
    Sm1 = np.array(((c, s, 0), (-s, c, 0), (0, 0, 1)))
    w_hae = np.dot(Sm1, w_hee)

    # HAE to HEEQ

    iota = np.radians(7.25)
    omega = np.radians((73.6667 + 0.013958 * ((mjd + 3242) / 365.25)))                      
    theta = np.arctan(np.cos(iota) * np.tan(lambda_sun - omega))                       

    # quadrant of theta must be opposite lambda_sun minus omega; Hapgood 1992 end of section 5   
    # get lambda-omega angle in degree mod 360 and theta in degrees
    lambda_omega_deg = np.mod(np.degrees(lambda_sun) - np.degrees(omega), 360)
    theta_node_deg = np.degrees(theta)

    # if the 2 angles are close to similar, so in the same quadrant, then theta_node = theta_node +pi           
    if np.logical_or(abs(lambda_omega_deg - theta_node_deg) < 1, abs(lambda_omega_deg - 360 - theta_node_deg) < 1): 
        theta = theta + np.pi                                                                                                          

    # rotation around Z by theta
    c, s = np.cos(theta), np.sin(theta)
    S2_1 = np.array(((c, s, 0), (-s, c, 0), (0, 0, 1)))

    # rotation around X by iota  
    iota = np.radians(7.25)
    c, s = np.cos(iota), np.sin(iota)
    S2_2 = np.array(((1, 0, 0), (0, c, s), (0, -s, c)))

    # rotation around Z by Omega  
    c, s = np.cos(omega), np.sin(omega)
    S2_3 = np.array(((c, s, 0), (-s, c, 0), (0, 0, 1)))

    # matrix multiplication to go from HAE to HEEQ components                
    [x_heeq, y_heeq, z_heeq] = np.dot(np.dot(np.dot(S2_1, S2_2), S2_3), w_hae) 

    print('HEE to HEEQ done')  
    
    return x_heeq, y_heeq, z_heeq

############################################################
############################################################
######################## DASH UTILS ########################
############################################################
############################################################

def create_nav_link(icon, label, href):
    
    '''
    Used to create the link to pages.
    '''
    
    return dcc.Link(
        dmc.Group(
            [
                dmc.ThemeIcon(
                    DashIconify(icon=icon, width=18),
                    size=40,
                    radius=40,
                    variant="light",
                    style={"backgroundColor": "#eaeaea", "color": "black"}
                ),
                dmc.Text(label, size="l", color="gray", weight=500),
            ],
            style={"display": "flex", 
                   "alignItems": "center", 
                   "justifyContent": "flex-end", 
                   #"border": "1px solid black", 
                   "padding": "28px"
                  },
        ),
        href=href,
        style={"textDecoration": "none",
               #"marginTop": 40
              },
    )


def create_double_slider(mins, maxs, values, step, label, ids, html_for, marks=None, unit=""):
    '''
    Creates a double slider with label.
    '''
    
    slider_label = dbc.Label( f"{label}: {values[0]}, {values[1]} {unit}", id=html_for, style={"font-size": "12px"})
    
    return html.Div(
    [
        slider_label,
        dcc.RangeSlider(id=ids, min=mins, max=maxs, step=step, value=values, marks=marks, persistence=True),
    ],
    className="mb-3",
)

def create_single_slider(mins, maxs, values, step, label, ids, html_for, marks, unit):
    '''
    Creates a single slider with label.
    '''
    
    slider_label = dbc.Label(f"{label}: {values}{unit}", id=html_for, style={"font-size": "12px"})
    if marks == None:
        slider = dcc.Slider(id=ids, min=mins, max=maxs, step=step, value=values, persistence=True)
    else:
        slider = dcc.Slider(id=ids, min=mins, max=maxs, step=step, value=values, marks=marks, persistence=True)
    return html.Div([slider_label, slider], className="mb-3")



def custom_indicator(val, prev):
    '''
    creates a custom indicator to visualize the error.
    '''
    
    if isinstance(val, list):
        val = val[0]
    if isinstance(prev, list):
        prev = prev[0]

    if isinstance(val, np.ndarray):
        val = val[0]
    if isinstance(prev, np.ndarray):
        prev = prev[0]

    if prev is None:
        color = 'lightgrey'
        symbol = '-'
        prev = 0
    elif val - prev == 0:
        color = 'lightgrey'
        symbol = '-'
    elif val - prev > 0:
        color = 'red'
        symbol = '▲'
    else:
        color = 'green'
        symbol = '▼'

    val = np.around(val, 3)
    prev = np.around(prev, 3)
    diff = np.around(val - prev,3)

    card = dbc.Card(
        body=True,
        className='text-center',
        children=[
            dbc.CardBody([
                html.H5('RMSE', className='text-dark font-medium'),
                html.H2(val, className='text-black font-large'),
                html.H6(f'{symbol} {diff}', className='text-small', style={'color': color})
            ])
        ]
    )
    return card
    
    


def make_progress_graph(progress, total, rmse, rmse_prev, iteration, status):
    '''
    creates the card that visualizes the fitting progress
    '''
    
    if status == None:
        status = 3
    
    progressperc = (progress / total) * 100
    if progressperc > 100:
        progressperc = 100
        
    if status == 0:
         progress_graph = html.Div([
            html.Div(
                id="alert-div",
                className="alert alert-primary",
                children=f"ⓘ Run fitting process or load existing fit!",
                style={
                    "overflowX": "auto",
                    "whiteSpace": "nowrap",
                },
            )
         ]
         )
        
    elif status == 1:
        # Secondary alert with progress bar and metric card
        progress_graph = html.Div([
            
            dbc.Row([
                dbc.Col([
                    html.Br(),
                    html.Div(
                        id="alert-div",
                        className="alert alert-dark",
                        children=f" Running iteration: {iteration +1 }",
                        style={
                            "overflowX": "auto",
                            "whiteSpace": "nowrap",
                        },                    
                    ),
                    #html.Br(),
                    html.Div(f"{progress}/{total}"),
                    dbc.Progress(
                        id='progress-bar',
                        value=progressperc,
                        max=100,
                        style={"height": "30px"}
                    ),
                ], width=8),  # Set width to half of the row's width (12-column grid system)
                dbc.Col([custom_indicator(rmse, rmse_prev)
                    
                ], width=4)  # Set width to half of the row's width (12-column grid system)
            ]),
        ], style={"max-height": "250px"#, "overflow-y": "auto"
                 })
    elif status == 2:
        progress_graph = html.Div([
                html.Div(
                    id="alert-div",
                    className="alert alert-dark",
                    children=f"✅ Reached target RMSE!",
                    style={
                        "overflowX": "auto",
                        "whiteSpace": "nowrap",
                    },
                ),
                dbc.Row([
                    dbc.Col([
                        html.Div(f"{progress}/{total}"),
                        dbc.Progress(
                            id='progress-bar',
                            value=progressperc,
                            max=100,
                            style={"height": "30px"}
                        ),
                    ], width=8),  
                    dbc.Col([custom_indicator(rmse, rmse_prev)

                    ], width=4) 
                ]),
            ], style={"max-height": "250px"
                     })
        
    elif status == 3:
        progress_graph = html.Div([
                html.Div(
                    id="alert-div",
                    className="alert alert-dark",
                    children=f"✅ Reached maximum number of iterations!",
                    style={
                        "overflowX": "auto",
                        "whiteSpace": "nowrap",
                    },
                ),
                dbc.Row([
                    dbc.Col([
                        html.Div(f"{progress}/{total}"),
                        dbc.Progress(
                            id='progress-bar',
                            value=progressperc,
                            max=100,
                            style={"height": "30px"}
                        ),
                    ], width=8),  # Set width to half of the row's width (12-column grid system)
                    dbc.Col([custom_indicator(rmse, rmse_prev)

                    ], width=4)  # Set width to half of the row's width (12-column grid system)
                ]),
            ], style={"max-height": "250px"#, "overflow-y": "auto"
                     })
        
    elif status == 4:
        progress_graph = html.Div([
                html.Div(
                    id="alert-div",
                    className="alert alert-dark",
                    children=f"❌ No hits, aborting!",
                    style={
                        "overflowX": "auto",
                        "whiteSpace": "nowrap",
                    },
                ),
            ], style={"max-height": "250px"#, "overflow-y": "auto"
                     })

    # Combine the secondary alert and progress graph components
    return html.Div([progress_graph])





############################################################
############################################################
######################### PLOTTING #########################
############################################################
############################################################





def plot_body3d(data_list, nowdate, color, sc, legendgroup = None):
    '''
    plots the current 3d position for a body
    '''
    
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
    
    x_now, y_now, z_now, now_time = now_data['x'], now_data['y'], now_data['z'], now_data['time']
    
    r_now, lon_now, lat_now = now_data['r'], now_data['lon'], now_data['lat']
    
    times_now_list = now_time.tolist()

    now_time_str = [time.strftime("%Y-%m-%d %H:%M:%S") for time in times_now_list]
    
    if legendgroup == None:
        trace = go.Scatter3d(x=x_now, y=y_now, z=z_now,
                         mode='markers', 
                         marker=dict(size=4, 
                                     #symbol='square',
                                     color=color),
                         name=sc, 
                         customdata=np.vstack((r_now, lon_now, lat_now )).T,  # Custom data for r, lat, lon values
                         showlegend=True, 
                         hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                         + sc + "</extra>", text=now_time_str),
    
    else:
        trace = go.Scatter3d(x=x_now, y=y_now, z=z_now,
                         mode='markers', 
                         marker=dict(size=4, 
                                     #symbol='square',
                                     color=color),
                         name=sc, 
                         customdata=np.vstack((r_now, lon_now, lat_now )).T,  # Custom data for r, lat, lon values
                         showlegend=True, 
                         hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                         + sc + "</extra>", text=now_time_str, legendgroup = legendgroup),
            

    return trace




def process_coordinates(data_list, date, nowdate, color, sc, legendgroup = None):
    '''
    plot spacecraft 3d position from previously loaded data
    '''
    
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
    
    if legendgroup == None:
        traces = [
                go.Scatter3d(x=x_past, y=y_past, z=z_past,
                            mode='lines', 
                            line=dict(color=color), 
                            name=sc + '_past_100', 
                            customdata=np.vstack((r_past, lat_past, lon_past)).T,  # Custom data for r, lat, lon values
                            showlegend=False, 
                            hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                            + sc + "</extra>", text=times_past_str),
                go.Scatter3d(x=x_future, y=y_future, z=z_future,
                            mode='lines', 
                            line=dict(color=color, dash='dash'), 
                            name=sc + '_future_100', 
                            customdata=np.vstack((r_future, lat_future, lon_future)).T,  # Custom data for r, lat, lon values
                            showlegend=False, 
                            hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                            + sc + "</extra>", text=times_future_str),
                go.Scatter3d(x=x_now, y=y_now, z=z_now,
                            mode='markers', 
                            marker=dict(size=3, 
                                        symbol='square',
                                        color=color),
                            name=sc, 
                            customdata=np.vstack((r_now, lat_now, lon_now)).T,  # Custom data for r, lat, lon values
                            showlegend=True, 
                            hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                            + sc + "</extra>", text=now_time_str),
                
            ]
        
    else:
        traces = [
                go.Scatter3d(x=x_past, y=y_past, z=z_past,
                            mode='lines', 
                            line=dict(color=color), 
                            name=sc + '_past_100', 
                            customdata=np.vstack((r_past, lat_past, lon_past)).T,  # Custom data for r, lat, lon values
                            showlegend=False, 
                            hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                            + sc + "</extra>", text=times_past_str),
                go.Scatter3d(x=x_future, y=y_future, z=z_future,
                            mode='lines', 
                            line=dict(color=color, dash='dash'), 
                            name=sc + '_future_100', 
                            customdata=np.vstack((r_future, lat_future, lon_future)).T,  # Custom data for r, lat, lon values
                            showlegend=False, 
                            hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                            + sc + "</extra>", text=times_future_str),
                go.Scatter3d(x=x_now, y=y_now, z=z_now,
                            mode='markers', 
                            marker=dict(size=3, 
                                        symbol='square',
                                        color=color),
                            name=sc, 
                            customdata=np.vstack((r_now, lat_now, lon_now)).T,  # Custom data for r, lat, lon values
                            showlegend=True, 
                            legendgroup = legendgroup,
                            hovertemplate="%{text}<br><b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                            + sc + "</extra>", text=now_time_str),
                
            ]

    return traces




############################################################
############################################################
##################### INSITU DATA LOAD #####################
############################################################
############################################################


def process_sav(list_of_names, list_of_contents):
    '''
    used to process uploaded cdf files
    '''
    
    content_type, content_string = list_of_contents.split(',')
    decoded = base64.b64decode(content_string)
    # Create a temporary file to save the uploaded data
    with tempfile.NamedTemporaryFile(delete=False, suffix=".cdf") as tmp_file:
        tmp_file.write(decoded)
        sav_data = readsav(tmp_file.name)

    name = list_of_names.split('/')[-1]
    parts = name.split('_')
    distance = parts[3].split('.')[0]
    direction = parts[1]
    
    
    date_format = '%Y/%m/%d %H:%M:%S.%f'
    starttime = datetime.datetime.strptime(sav_data.time[0].decode('utf-8'), date_format)
    endtime = datetime.datetime.strptime(sav_data.time[-1].decode('utf-8'), date_format)
    
    eventbegin = datetime.datetime.strptime(sav_data.mctime[0].decode('utf-8'), date_format)
    eventend = datetime.datetime.strptime(sav_data.mctime[-1].decode('utf-8'), date_format)
    
    time_int = []
    
    while starttime <= endtime:
        time_int.append(starttime)
        starttime += datetime.timedelta(minutes=30)
        
    ll = np.zeros(np.size(sav_data.ibx),dtype=[('time',object),('br', float),('bt', float),('bn', float)] )
    
    ll = ll.view(np.recarray)  
    
    ll.time = time_int[:np.size(sav_data.ibx)]
    ll.br = sav_data.ibx
    ll.bt = sav_data.iby
    ll.bn = sav_data.ibz
    
    uploadpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/uploaded")) 
    
    filename = name[:-4] + '.pickle'
    
    p.dump(ll, open(uploadpath + '/'+ filename, "wb"))
    print("Created pickle file from sav: " +filename)
            
    
    return eventbegin, eventend, "SYN", filename, direction, distance


def process_cdf(list_of_names, list_of_contents):
    '''
    used to process uploaded cdf files
    '''
    list_of_names, list_of_contents, firstdate, spacecraft = filter_and_sort_files(list_of_names, list_of_contents)
    
    data = cdf_to_data(list_of_names, list_of_contents, spacecraft)
    
    uploadpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/uploaded")) 
    
    filename = spacecraft + '_' + firstdate + '.pickle'
    
    p.dump(data, open(uploadpath + '/'+ filename, "wb"))
    print("Created pickle file from cdf: " +filename)
            
    
    return firstdate, spacecraft.upper(), filename

def cdf_to_data(list_of_names, list_of_contents, spacecraft):
    
    '''
    used to read data from cdf
    '''
    br1 = np.zeros(0)
    bt1 = np.zeros(0)
    bn1 = np.zeros(0)
    time1 = np.zeros(0,dtype=[('time',object)])
    
    for i, name in enumerate(list_of_names):
        content_type, content_string = list_of_contents[i].split(',')
        decoded = base64.b64decode(content_string)
        # Create a temporary file to save the CDF content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cdf") as tmp_file:
            tmp_file.write(decoded)
            cdf_file = cdflib.CDF(tmp_file.name)  
            
            if (spacecraft == "SOLO") or (spacecraft == "solo"):
                b = cdf_file.varget('B_RTN')
                time = cdf_file.varget('EPOCH')
            elif (spacecraft == "PSP") or (spacecraft == "psp"):
                b = cdf_file.varget('psp_fld_l2_mag_RTN_1min')
                time = cdf_file.varget('epoch_mag_RTN_1min')
            elif (spacecraft == "WIND") or (spacecraft == "wind"):
                b = cdf_file.varget('BRTN')
                time = cdf_file.varget('Epoch')
            
            
        br = b[:, 0]
        bt = b[:, 1]
        bn = b[:, 2]

        br1 = np.append(br1, br)
        bt1 = np.append(bt1, bt)
        bn1 = np.append(bn1, bn)

        t1 = parse_time(cdflib.cdfastropy.convert_to_astropy(time, format=None)).datetime
        time1 = np.append(time1,t1)
    
        
    starttime = time1[0].replace(hour = 0, minute = 0, second=0, microsecond=0)
    endtime = time1[-1].replace(hour = 0, minute = 0, second=0, microsecond=0)
    
    if starttime == endtime:
        endtime = endtime + datetime.timedelta(days=1)
        
    time_int = []
    while starttime < endtime:
        time_int.append(starttime)
        starttime += datetime.timedelta(minutes=10)
        
    time_int_mat = mdates.date2num(time_int)
    time1_mat = mdates.date2num(time1)
    
    ll = np.zeros(np.size(time_int),dtype=[('time',object),('br', float),('bt', float),('bn', float)] )
    
    ll = ll.view(np.recarray)  
    
    # replace all unreasonable large and small values with nan
    thresh = 1e6
    br1[br1 > thresh] = np.nan
    br1[br1 < -thresh] = np.nan
    bt1[bt1 > thresh] = np.nan
    bt1[bt1 < -thresh] = np.nan
    bn1[bn1 > thresh] = np.nan
    bn1[bn1 < -thresh] = np.nan
    
    ll.time = time_int
    ll.br = np.interp(time_int_mat, time1_mat[~np.isnan(br1)], br1[~np.isnan(br1)])
    ll.bt = np.interp(time_int_mat, time1_mat[~np.isnan(bt1)], bt1[~np.isnan(bt1)])
    ll.bn = np.interp(time_int_mat, time1_mat[~np.isnan(bn1)], bn1[~np.isnan(bn1)])
    
    return ll

def filter_and_sort_files(file_list, content_list):
    
    '''
    used to filter and sort loaded files
    '''
    
    # Create a dictionary to store filenames by spacecraft
    spacecraft_files = {}

    ## Initialize variables to keep track of the most frequent spacecraft and its date
    max_spacecraft = None
    min_date = None

    # Iterate through the input lists
    for filename, content in zip(file_list, content_list):
        # Check if the filename contains 'mag'
        if 'mag' in filename:
            # Extract the spacecraft identifier (e.g., 'psp')
            spacecraft = filename.split('_')[0]

            # Extract the date from the filename using a regular expression
            date_match = re.search(r'(\d{8})', filename)
            if date_match:
                date = date_match.group(0)
            else:
                date = None

            # Add the filename, content, and date to the spacecraft's list
            if spacecraft in spacecraft_files:
                spacecraft_files[spacecraft].append((filename, content, date))
            else:
                spacecraft_files[spacecraft] = [(filename, content, date)]

            # Update the most recent date if needed
            if date and (min_date is None or date < min_date):
                max_spacecraft = spacecraft
                min_date = date

    # Find the spacecraft with the most items
    most_files = spacecraft_files[max_spacecraft]

    # Sort the filenames for the most frequent spacecraft
    most_files.sort()

    # Extract the filenames, content, and date for the most frequent spacecraft
    filenames, contents, date = zip(*most_files)

    return filenames, contents, min_date, max_spacecraft


#@functools.lru_cache()    
def get_rt_data(sc, insitubegin, insituend, plushours):
    '''
    used to load insitudata for the graphstore from helioforecast
    '''
    
    if sc == "NOAA_RTSW":
        url = 'https://helioforecast.space/static/sync/insitu_python/noaa_rtsw_last_35files_now.p'
    elif sc == "STEREO-A_beacon":
        url = 'https://helioforecast.space/static/sync/insitu_python/stereoa_beacon_rtn_last_35days_now.p'
        
    file = urllib.request.urlopen(url)    
    data, dh = p.load(file)
    
    # Extract relevant fields
    time = data['time']
    bx = data['bx']
    by = data['by']
    bz = data['bz']
    x = data['x'] 
    y = data['y'] 
    z = data['z'] 

    # Filter data based on insitubegin
    begin_index = np.where(time >= insitubegin + datetime.timedelta(hours=24))[0][0]
    time = time[begin_index:]
    bx = bx[begin_index:]
    by = by[begin_index:]
    bz = bz[begin_index:]
    x = x[begin_index:]
    y = y[begin_index:]
    z = z[begin_index:]

    # insitubegin = insitubegin.replace(tzinfo=None)
    # insituend = insituend.replace(tzinfo=None)
    
    # # Ensure time ends with insituend
    # while time[-1] < insituend:
    #     #print(time[-1])
    #     #print(insituend)

    #     time = np.append(time, time[-1]+ (time[-1]-time[-2]))
    #     bx = np.append(bx, np.nan)
    #     by = np.append(by, np.nan)
    #     bz = np.append(bz, np.nan)
    #     x = np.append(x, 1.)
    #     y = np.append(y, 0.)
    #     z = np.append(z, 0.)
    
    # # Find indices within the specified time range
    # mask = (time >= insitubegin) & (time <= insituend)

    #print(x)
    #print(time)
    
    # heeq_bx, heeq_by, heeq_bz = hc.convert_RTN_to_HEEQ_mag(x[mask], y[mask], z[mask], bx[mask], by[mask], bz[mask])
    # b_HEEQ = np.column_stack((heeq_bx, heeq_by, heeq_bz))
    
    # dt = time[mask]
    # b_RTN = np.column_stack((bx[mask], by[mask], bz[mask]))
    # pos = np.column_stack((x[mask], y[mask], z[mask]))
    
    heeq_bx, heeq_by, heeq_bz = hc.convert_RTN_to_HEEQ_mag(x, y, z, bx, by, bz)
    b_HEEQ = np.column_stack((heeq_bx, heeq_by, heeq_bz))
    
    dt = time
    b_RTN = np.column_stack((bx, by, bz))
    pos = np.column_stack((x, y, z))

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

@functools.lru_cache()    
def get_uploaddata(filename):
    
    '''
    used to generate the insitudata for the graphstore from upload (app.py)
    '''
    uploadpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/uploaded"))

    data = p.load(open(uploadpath + '/' + filename, "rb" ) )    

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
    archivepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/archive"))
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
        distance = int(parts[3].split('.')[0])
        direction = int(parts[1])
        
        # Create an array with NaN values
        posdata = np.empty((desired_length, 3))

        posdata[:, 0], posdata[:, 1], posdata[:, 2] = sphere2cart(float(distance)*0.00465047, np.deg2rad(-float(0)+90), np.deg2rad(float(direction)))
        
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
                          
    return b_HEEQ, b_RTN, dt, pos


@functools.lru_cache()    
def get_archivedata(sc, insitubegin, insituend):
    
    '''
    used to generate the insitudata for the graphstore from archive (app.py)
    '''
    
    archivepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data/archive"))
        
    if (sc == 'BepiColombo') or (sc == 'BEPI'):
        filertn = '/bepi_ob_2019_now_rtn.p'
        
    elif sc == 'DSCOVR':
        file = ''
        
    elif sc == 'MESSENGER':
        filertn = '/messenger_2007_2015_sceq_removed.p'
        fileheeq = '/messenger_2007_2015_heeq_removed.p'
        
    elif sc == 'PSP':
        filertn = '/psp_2018_now_rtn.p'
        fileheeq = '/psp_2018_now_heeq.p'
    
    elif (sc == 'SolarOrbiter') or (sc == 'SOLO'):
        filertn = '/solo_2020_now_rtn.p'
        fileheeq = '/solo_2020_now_heeq.p'
    
    elif (sc == 'STEREO-A') or (sc == "STEREO_A"):
        filertn = '/stereoa_2007_now_rtn.p'
        fileheeq = '/stereoa_2007_now_heeq.p'
        
    elif (sc =='VEX-A') or (sc == "VEX"):
        filertn = '/vex_2007_2014_sceq_removed.p'
        fileheeq = '/vex_2007_2014_heeq_removed.p'
        
    elif sc == 'Wind':
        filertn = '/wind_1995_now_rtn.p'
        fileheeq = '/wind_1995_now_heeq.p'
        
    [data,dataheader]=p.load(open(archivepath + filertn, "rb" ) )
    
    # Extract relevant fields
    time = data['time']
    bx = data['bx']
    by = data['by']
    bz = data['bz']
    x = data['x'] * 6.68459e-9
    y = data['y'] * 6.68459e-9
    z = data['z'] * 6.68459e-9
    
    insitubegin = insitubegin.replace(tzinfo=None)
    insituend = insituend.replace(tzinfo=None)

    # Find indices within the specified time range
    mask = (time >= insitubegin) & (time <= insituend)
    
    try:
        [data,dataheader]=p.load(open(archivepath + fileheeq, "rb" ) ) 
    
        # Extract relevant fields
        time_heeq = data['time']
        heeq_bx = data['bx']
        heeq_by = data['by']
        heeq_bz = data['bz']
        x_heeq = data['x']
        y_heeq = data['y']
        z_heeq = data['z']
        
        #print(heeq_bx)

        # Find indices within the specified time range
        mask_heeq = (time_heeq >= insitubegin) & (time_heeq <= insituend)
        b_HEEQ = np.column_stack((heeq_bx[mask_heeq], heeq_by[mask_heeq], heeq_bz[mask_heeq]))
        
    except:
        
        heeq_bx, heeq_by, heeq_bz = hc.convert_RTN_to_HEEQ_mag(x[mask], y[mask], z[mask], bx[mask], by[mask], bz[mask])
        b_HEEQ = np.column_stack((heeq_bx, heeq_by, heeq_bz))
        
    #heeq_bx, heeq_by, heeq_bz = hc.convert_RTN_to_HEEQ(x[mask], y[mask], z[mask], bx[mask], by[mask], bz[mask])
    #print(heeq_bx)

    # Extract data within the time range
    dt = time[mask]
    b_RTN = np.column_stack((bx[mask], by[mask], bz[mask]))
    pos = np.column_stack((x[mask], y[mask], z[mask]))
    #print(pos)
    #print(bx[mask])
    #print(dt)
    return b_HEEQ, b_RTN, dt, pos


@functools.lru_cache()    
def get_insitudata(sc, insitubegin, insituend):
    
    '''
    used to generate the insitudata for the graphstore (app.py)
    '''
        
    if (sc == 'BepiColombo') or (sc == 'BEPI'):
        reference_frame = 'GSE'
        observer = 'Bepi'
    elif sc == 'DSCOVR':
        reference_frame = 'GSE'
        observer = 'DSCOVR'
    elif sc == 'MESSENGER':
        reference_frame = 'MSGR_RTN'
        observer = 'Mes'
    elif sc == 'PSP':
        reference_frame = 'SPP_RTN'
        observer = 'PSP'
    elif (sc == 'SolarOrbiter') or (sc == 'SOLO'):
        reference_frame = 'SOLO_SUN_RTN'
        observer = 'SolO'
    elif (sc == 'STEREO-A') or (sc == "STEREO_A"):
        reference_frame = 'STAHGRTN'
        observer = 'STA'
    elif sc == 'VEX-A':
        reference_frame = 'VSO'
        observer = 'VEX'
    elif sc == 'Wind':
        reference_frame = 'GSE'
        observer = 'Wind'
            
            
    observer_obj = getattr(heliosat, observer)() 
    
    t, b_HEEQ = observer_obj.get([insitubegin,insituend], "mag", reference_frame='HEEQ', as_endpoints=True)
    _, b_RTN = observer_obj.get([insitubegin,insituend], "mag", reference_frame=reference_frame, as_endpoints=True)
    dt = [datetime.datetime.utcfromtimestamp(ts) for ts in t]

    pos = observer_obj.trajectory(dt, reference_frame='HEEQ')
    
    if np.shape(pos)[0] is not np.shape(dt):
        pos = observer_obj.trajectory(dt, reference_frame='HEEQ', smoothing="gaussian")

    
    return b_HEEQ, b_RTN, dt, pos


############################################################
############################################################
######################## DATA LOAD #########################
############################################################
############################################################


def load_body_data(mag_coord_system, date, datafile = None):
    '''
    Used in generate_graphstore to load the body data for later 3d plotting.
    Can use both archive and astrospice.
    '''
    dt = TimeDelta(0.5 * u.hour)
    delta = datetime.timedelta(days=10)
    now_time = Time(date, scale='utc')
    start_time, end_time = now_time - delta, now_time + delta
    frame = HeliographicStonyhurst()
    
    times = Time(np.arange(start_time, end_time, dt))
    
    planets = [1, # Mercury
              2, #Venus
              4, #Mars
              3, # Earth
              ]
    colors = ['slategrey',
             'darkgoldenrod',
             'red',
             'mediumseagreen']
    names = ['Mercury', 'Venus', 'Mars', 'Earth']
    
    dicc = {}
        
    ########## BODY
    
    for i, planet in enumerate(planets):
        
        try:
            data = getarchivecoords(names[i], begin= date - delta, end = date + delta, arrays =  '30T', datafile = datafile)
            
            print(names[i] + ' traces obtained from Archive/Helioforecast')
        
        except Exception as e:
            print(e)
            
            data = np.zeros(np.size(times),dtype=[('time',object),('r','f8'),('lon','f8'),('lat','f8'),('x','f8'),('y','f8'),('z','f8')])
            data = data.view(np.recarray)
        
            coords = astrospice.generate_coords(planet, times)
            coords = coords.transform_to(frame)

            data.time = times.to_datetime()
            data.r = coords.radius.to(u.au).value
            data.lon = coords.lon.value #degrees
            data.lat = coords.lat.value
            [data.x, data.y, data.z] = sphere2cart(data.r, np.deg2rad(-data.lat+90), np.deg2rad(data.lon))
            
            print(names[i] + ' traces obtained through astrospice')
        
        dicc[names[i]] = {'data': data, 'color': colors[i]}

    
    return dicc
    
def load_pos_data(mag_coord_system, sc, date, datafile = None, deltadays = 10):
    
    '''
    used to load the data of a specific spacecraft (manually)
    '''
    
    dt = TimeDelta(0.5 * u.hour)
    delta = datetime.timedelta(days=deltadays)
    now_time = Time(date, scale='utc')
    start_time, end_time = now_time - delta, now_time + delta
    frame = HeliographicStonyhurst()
    
    times = Time(np.arange(start_time, end_time, dt))
    
    ########## SPACECRAFT
    
    spacecraft_info = {
        'SolarOrbiter': ('solar orbiter', 'coral', 'Solar orbiter'),
        'SOLO': ('solar orbiter', 'coral', 'Solar orbiter'),
        'PSP': ('psp', 'black', 'Solar probe plus'),
        'STEREO-A': ('stereo-a', 'darkred', 'Stereo ahead'),
        'STEREO_A': ('stereo-a', 'darkred', 'Stereo ahead'),
        'BEPI': ('mpo', 'blue', 'Bepicolombo mpo'),
    }
    
    kernel, color, generator = spacecraft_info[sc]
    
    try:
        
        data = getarchivecoords(sc, begin= date - delta, end = date + delta, arrays = '30T', datafile = datafile)
        types = 'Data Archive/Helioforecast'
        
    except:
        kernelss = astrospice.registry.get_kernels(kernel, 'predict')[0]
        coords = astrospice.generate_coords(generator, times).transform_to(frame)
        
        data = np.zeros(np.size(times),dtype=[('time',object),('r','f8'),('lon','f8'),('lat','f8'),('x','f8'),('y','f8'),('z','f8')])
        data = data.view(np.recarray)
        data.time = times.to_datetime()
        data.r = coords.radius.to(u.au).value
        data.lon = coords.lon.value #degrees
        data.lat = coords.lat.value
        [data.x, data.y, data.z] = sphere2cart(data.r, np.deg2rad(-data.lat+90), np.deg2rad(data.lon))
        
        types = 'astrospice'

    
    return {'data': data, 'color': color}, types
    
def get_posdata(mag_coord_system, sc, date, datafile = None, threed = False):
    '''
    used to load the traces for the 2d plot for a manually selected sc
    '''
    dt = TimeDelta(6 * u.hour)
    delta = datetime.timedelta(days=100)
    now_time = Time(date, scale='utc')
    start_time, end_time = now_time - delta, now_time + delta
    frame = HeliographicStonyhurst()
    
    times_past = Time(np.arange(start_time, now_time, dt))
    times_future = Time(np.arange(now_time, end_time, dt))
    
    spacecraft_info = {
        'SolarOrbiter': ('solar orbiter', 'coral', 'Solar orbiter'),
        'SOLO': ('solar orbiter', 'coral', 'Solar orbiter'),
        'PSP': ('psp', 'black', 'Solar probe plus'),
        'STEREO-A': ('stereo-a', 'darkred', 'Stereo ahead'),
        'STEREO_A': ('stereo-a', 'darkred', 'Stereo ahead'),
        'BEPI': ('mpo', 'blue', 'Bepicolombo mpo'),
    }

    if sc in spacecraft_info:
        kernel, color, generator = spacecraft_info[sc]
        try:
            coords_past = getarchivecoords(sc, begin= date - delta, end = date, arrays = '6h', datafile = datafile)
            coords_future = getarchivecoords(sc, begin= date, end = date + delta, arrays = '6h', datafile = datafile)
            r_now, lon_now, lat_now = getarchivecoords(sc, begin= date, end = None, arrays = None, datafile = datafile)
            r_now, lon_now, lat_now = [r_now], [lon_now], [lat_now]
            types = 'Data Archive/Helioforecast'

            r_past, lon_past, lat_past, times_past = coords_past.r, coords_past.lon, coords_past.lat, coords_past.time
            r_future, lon_future, lat_future, times_future = coords_future.r, coords_future.lon, coords_future.lat, coords_future.time

            # Convert Time objects to formatted strings
            times_past_str = [time.strftime("%Y-%m-%d %H:%M:%S") for time in times_past]
            times_future_str = [time.strftime("%Y-%m-%d %H:%M:%S") for time in times_future]
            now_time_str = now_time.strftime("%Y-%m-%d %H:%M:%S")
            
        except:
            #kernelss = astrospice.registry.get_kernels(kernel, 'predict')[0]
            coords_past = astrospice.generate_coords(generator, times_past).transform_to(frame)
            coords_future = astrospice.generate_coords(generator, times_future).transform_to(frame)
            coords_now = astrospice.generate_coords(generator, now_time).transform_to(frame)
            types = 'astrospice'

            r_past, lon_past, lat_past = coords_past.radius.to(u.au).value, coords_past.lon.value, coords_past.lat.value
            r_future, lon_future, lat_future = coords_future.radius.to(u.au).value, coords_future.lon.value, coords_future.lat.value
            r_now, lon_now, lat_now = coords_now.radius.to(u.au).value, coords_now.lon.value, coords_now.lat.value

            # Convert Time objects to formatted strings
            times_past_str = [time.datetime.strftime("%Y-%m-%d %H:%M:%S") for time in times_past]
            times_future_str = [time.datetime.strftime("%Y-%m-%d %H:%M:%S") for time in times_future]
            now_time_str = now_time.datetime.strftime("%Y-%m-%d %H:%M:%S")

    if threed == False:
        traces = [
            go.Scatterpolar(r=r_past, theta=lon_past, mode='lines', line=dict(color=color), name=sc + '_past_100', showlegend=False, hovertemplate="%{text}<br>%{r:.1f} AU<br>%{theta:.1f}°<extra>" + sc + "</extra>", text=times_past_str),
            go.Scatterpolar(r=r_future, theta=lon_future, mode='lines', line=dict(color=color, dash='dash'), name=sc + '_future_100', showlegend=False, hovertemplate="%{text}<br>%{r:.1f} AU<br>%{theta:.1f}°<extra>" + sc + "</extra>", text=times_future_str),
            go.Scatterpolar(r=r_now, theta=lon_now, mode='markers', marker=dict(size=8, symbol='square', color=color), name=sc, showlegend=False, hovertemplate="%{text}<br>%{r:.1f} AU<br>%{theta:.1f}°<extra>" + sc + "</extra>", text=[now_time_str]),
        ]
    return traces, types


def getarchivecoords(body, begin, end = None, arrays = None, datafile = None):
    
    '''
    Function to create coordinates from archive.
    '''
    
    if (body == 'BepiColombo') or (body == 'BEPI'):
        bodyindex = 4
        
    elif body == 'PSP':
        bodyindex = 0
        
    elif (body == 'SolarOrbiter') or (body == 'SOLO'):
        bodyindex = 1
        
    elif (body == 'STEREO-A') or (body == "STEREO_A"):
        bodyindex = 2
        
    elif body == 'Wind':
        bodyindex = 5
        
    elif (body == 'STEREO-B') or (body == "STEREO_B"):
        bodyindex = 3
    
    elif body == 'Earth':
        bodyindex = 6
        
    elif body == 'Mercury':
        bodyindex = 7

    elif body == 'Venus':
        bodyindex = 8
        
    elif body == 'Mars':
        bodyindex = 9
    
    
    # Extract relevant fields
    time = datafile[bodyindex]['time']
    r = datafile[bodyindex]['r']
    lon = datafile[bodyindex]['lon']
    lat = datafile[bodyindex]['lat']
    x = datafile[bodyindex]['x']
    y = datafile[bodyindex]['y']
    z = datafile[bodyindex]['z']
    
    insitubegin = begin.replace(tzinfo=None)

    if arrays is not None:
        if end == None:
            start_date = begin - datetime.timedelta(days=10)
            end_date = begin  + datetime.timedelta(days=10)
        else:
            start_date = begin
            end_date = end
        
        mask = (time >= start_date) & (time <= end_date)
        
        df = pd.DataFrame({
            'time': time[mask],
            'r': r[mask],
            'lon': lon[mask],
            'lat': lat[mask],
            'x': x[mask],
            'y': y[mask],
            'z': z[mask]
        })

        # Round the 'time' column to the specified resolution
        df['time'] = df['time'].dt.round('10T')

        # Resample data using arrays as timedelta
        df = df.set_index('time').resample(rule=arrays).interpolate()

        # Reset the index and convert the 'time' column to datetime objects
        df = df.reset_index()

        # Convert the DataFrame back to a recarray
        time_stamps = df['time']
        dt_lst= [element.to_pydatetime() for element in list(time_stamps)] #extract timestamps in datetime.datetime format
        
        data = np.zeros(len(dt_lst),dtype=[('time',object), ('r', float),('lon', float),('lat', float), ('x', float),('y', float),('z', float)])
        
        data = data.view(np.recarray)
        
        data.time = dt_lst
        data.x = df['x']
        data.y = df['y']
        data.z = df['z']
        data.lat = df['lat']
        data.lon = df['lon']
        data.r = df['r']
        
        return data
        
    elif end is not None:
        insituend = end.replace(tzinfo=None)
        # Find indices within the specified time range
        mask = (time >= insitubegin) & (time <= insituend)
        return r[mask], lon[mask], lat[mask]
    else:
        # Find the index of the closest available time
        closest_index = np.argmin(np.abs(time - insitubegin))
        return r[closest_index], lon[closest_index], lat[closest_index]
    
    

def getbodytraces(mag_coord_system, sc, date, threed = False, datafile = None):
    
    '''
    This function is used at startup to obtain the 2d traces.
    Can use both archive and astrospice.
    '''
    
    
    try:
        date_object = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S%z")
    except:
        try:
            date_object = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        except:
            date_object = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
            
    now_time = Time(date_object, scale='utc')
    frame = HeliographicStonyhurst()
    
    traces = []
    
    planets = [1, # Mercury
              2, #Venus
              4, #Mars
              ]
    colors = ['slategrey',
             'darkgoldenrod',
             'red']
    names = ['Mercury', 'Venus', 'Mars']
    
    for i, planet in enumerate(planets):
        try:
            
            r, lon, lat = getarchivecoords(names[i], date_object, end = None, arrays = None, datafile = datafile)
            if i == 0:
                print("Loading body traces from Data Archive/Helioforecast")
            
        except Exception as e:
            coords = astrospice.generate_coords(planet, now_time)
            coords = coords.transform_to(frame)

            r = coords.radius.to(u.au).value
            lon = coords.lon.value #degrees
            lat = coords.lat.value
            
            if i == 0:
                print("Loading body traces through astrospice: ", e)
        
        if threed == False:
            
            try:
                trace = go.Scatterpolar(r=r, theta=lon, mode='markers', marker=dict(size=10, color=colors[i]), name = names[i], showlegend=False, hovertemplate="%{r:.1f} AU<br>%{theta:.1f}°"+
                        "<extra></extra>"),
                traces.append(trace)

                nametrace = go.Scatterpolar(r=r + 0.03, theta=lon + 0.03, mode='text', text=names[i],textposition='top right', showlegend=False, hovertemplate = None, hoverinfo = "skip", textfont=dict(color=colors[i], size=14))
                traces.append(nametrace)
            except:
                trace = go.Scatterpolar(r=[r], theta=[lon], mode='markers', marker=dict(size=10, color=colors[i]), name = names[i], showlegend=False, hovertemplate="%{r:.1f} AU<br>%{theta:.1f}°"+
                        "<extra></extra>"),
                traces.append(trace)

                nametrace = go.Scatterpolar(r=[r + 0.03], theta=[lon + 0.03], mode='text', text=names[i],textposition='top right', showlegend=False, hovertemplate = None, hoverinfo = "skip", textfont=dict(color=colors[i], size=14))
                traces.append(nametrace)
        else:
            x,y,z = sphere2cart(r, np.deg2rad(-lat+90), np.deg2rad(lon))
            
            scs = names[i]

            trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=4, color=colors[i]),
                name=names[i],
                customdata=np.vstack((r, lat, lon)).T,  # Custom data for r, lat, lon values
                hovertemplate="<b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" + sc + "</extra>",
                text=names[i]  # Text to display in the hover label
            )
            traces.append(trace)

        
    return traces





############################################################
############################################################
###################### GENERAL UTILS #######################
############################################################
############################################################

def cart2sphere(x,y,z):
    r = np.sqrt(x**2+ y**2 + z**2)           
    theta = np.arctan2(z,np.sqrt(x**2+ y**2))
    phi = np.arctan2(y,x)                    
    return (r, theta, phi)
    

def sphere2cart(r,lat,lon):
    x = r * np.sin( lat ) * np.cos( lon )
    y = r * np.sin( lat ) * np.sin( lon )
    z = r * np.cos( lat )
    return (x, y,z)

def interpolate_points(point1, point2, num_points, multiplicator=1):
    # Generate an array of linearly spaced values between 0 and multiplicator
    t_values = np.linspace(0, multiplicator, num_points)

    # Use linear interpolation to find points along the line
    interpolated_points = (1 - t_values)[:, np.newaxis] * point1 + t_values[:, np.newaxis] * point2

    return interpolated_points

class Event:
    
    def __init__(self, begin, end, idd, sc):
        self.begin = begin
        self.end = end
        self.duration = self.end-self.begin
        self.id = idd 
        self.sc = sc
    def __str__(self):
        return self.id

def get_catevents(sc, year, month, day):
    '''
    Returns events from helioforecast.space filtered by year, month, day, and sc.
    Used during startup.
    '''
    url = 'https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v22.csv'
    icmecat = pd.read_csv(url)
    starttime = icmecat.loc[:, 'icme_start_time']
    idd = icmecat.loc[:, 'icmecat_id']
    sc_insitu = icmecat.loc[:, 'sc_insitu']
    endtime = icmecat.loc[:, 'mo_end_time']
    mobegintime = icmecat.loc[:, 'mo_start_time']

    evtList = []
    dateFormat = "%Y/%m/%d %H:%M"
    begin = pd.to_datetime(starttime, format=dateFormat)
    mobegin = pd.to_datetime(mobegintime, format=dateFormat)
    end = pd.to_datetime(endtime, format=dateFormat)
    
    if sc == "STEREO A":
        sc = "STEREO-A"

    for i, event in enumerate(mobegin):
        if (year is None or mobegin[i].year == year) and \
           (month is None or mobegin[i].month == int(month)) and \
           (day is None or mobegin[i].day == int(day)) and \
           (sc is None or sc_insitu[i] == sc):
            
            evtList.append(str(Event(mobegin[i], end[i], idd[i], sc_insitu[i])))


    if len(evtList) == 0:
        evtList = ['No events returned', ]

    return evtList

def load_cat_id(idd):
    '''
    Returns from helioforecast.space the event with a given ID.
    '''
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
    
    i = np.where(idds == idd)[0]
    return Event(mobegin[i], end[i], idds[i], sc[i])
        
def load_cat(date):
    '''
    Returns from helioforecast.space the event list for a given day
    '''
    url='https://helioforecast.space/static/sync/icmecat/HELIO4CAST_ICMECAT_v22.csv'
    icmecat=pd.read_csv(url)
    starttime = icmecat.loc[:,'icme_start_time']
    idd = icmecat.loc[:,'icmecat_id']
    sc = icmecat.loc[:, 'sc_insitu']
    endtime = icmecat.loc[:,'mo_end_time']
    mobegintime = icmecat.loc[:, 'mo_start_time']
    
    evtList = []
    dateFormat="%Y/%m/%d %H:%M"
    begin = pd.to_datetime(starttime, format=dateFormat)
    mobegin = pd.to_datetime(mobegintime, format=dateFormat)
    end = pd.to_datetime(endtime, format=dateFormat)

    
    for i, event in enumerate(mobegin):
        if (mobegin[i].year == date.year and mobegin[i].month == date.month and mobegin[i].day == date.day and mobegin[i].hour == date.hour):
            return Event(mobegin[i], end[i], idd[i], sc[i])
        
        
        
def loadpickle(path=None, number=-1):

    """ Loads the filepath of a pickle file. """
    
    path = '/' + path + '/'
    #print(path)
    
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output")) + path
    #print(path)

    # Get the list of all files in path
    dir_list = sorted(os.listdir(path))

    resfile = []
    respath = []
    # we only want the pickle-files
    for file in dir_list:
        if file.endswith(".pickle") and not file.endswith("ensembles.pickle"):
            resfile.append(file) 
            respath.append(os.path.join(path,file))
            
    filepath = path + resfile[number]

    return filepath


        
def load_fit(name, graph):
    
    filepath = loadpickle(name)
    ensemble_filepath = filepath.split('.')[0] + '_ensembles.pickle'
    
    # read from pickle file
    file = open(filepath, "rb")
    data = p.load(file)
    file.close()
    
    if os.path.exists(ensemble_filepath):
        pass
    else:
        print('Generating and converting ensembles!')
        # Generate the ensemble data and save it to the file
        ensemble_HEEQ, ensemble_HEEQ_data = generate_ensemble(filepath, graph['t_data'], graph['pos_data'], reference_frame='HEEQ', reference_frame_to='HEEQ', max_index=data['model_obj'].ensemble_size)
        ensemble_RTN, ensemble_RTN_data = generate_ensemble(filepath, graph['t_data'], graph['pos_data'], reference_frame='HEEQ', reference_frame_to='RTN', max_index=data['model_obj'].ensemble_size)

        ensemble_data = {
            'ensemble_HEEQ': ensemble_HEEQ,
            'ensemble_HEEQ_data': ensemble_HEEQ_data,
            'ensemble_RTN': ensemble_RTN,
            'ensemble_RTN_data': ensemble_RTN_data
        }
        with open(ensemble_filepath, 'wb') as ensemble_file:
            p.dump(ensemble_data, ensemble_file)
            
            
    
    observers = data['data_obj'].observers
    
    ###### tablenew
    
    t0 = data['t_launch']
    
    for i, observer in enumerate(observers):
        
        obs_dic = {
                "spacecraft": [""],
                "ref_a": [""],
                "ref_b": [""],
                "t_1": [""],
                "t_2": [""],
                "t_3": [""],
                "t_4": [""],
                "t_5": [""],
                "t_6": [""],
            }
        
        if observer[0] == "SOLO":
            obs_dic['spacecraft'] = ['SolarOrbiter']
        elif observer[0] == "BEPI":
            obs_dic['spacecraft'] = ['BepiColombo']
        else:
            obs_dic['spacecraft'] = [observer[0]]
            
        obs_dic['ref_a'] = [observer[2].strftime("%Y-%m-%d %H:%M")]
        obs_dic['ref_b'] = [observer[3].strftime("%Y-%m-%d %H:%M")]
        
        for j, dt in enumerate(observer[1]):
            t_key = "t_" + str(j + 1)
            obs_dic[t_key] = [dt.strftime("%Y-%m-%d %H:%M")]         
        df = pd.DataFrame(obs_dic)
        
        if i == 0:
            tablenew = df
        else:
            pd.concat([tablenew, df_new_row])
            
    ####### modelsliders
    
    long_new = [data['model_kwargs']['iparams']['cme_longitude']['minimum'], data['model_kwargs']['iparams']['cme_longitude']['maximum']]
    
    # Create a dictionary to store the results
    result_dict = []
    mag_result_dict = []

    # Iterate through each key in iparams
    for key, value in data['model_kwargs']['iparams'].items():
        # Get the maximum and minimum values if they exist, otherwise use the default_value
        maximum = value.get('maximum', value.get('default_value'))
        minimum = value.get('minimum', value.get('default_value'))
        # Create the long_new list
        var_new = [minimum, maximum]
        
        # Check if the key starts with 'mag' or 't_fac' and append it to the corresponding dictionary
        if key.startswith('mag') or key.startswith('t_fac'):
            mag_result_dict.append(var_new)
        else:
            result_dict.append(var_new)
            
            
    ######## particle slider
    
    given_values = [265, 512, 1024, 2048]
    closest_value = min(given_values, key=lambda x: abs(x - len(data['epses'])))

    
    partslid_new = given_values.index(closest_value)
    
    
    
    ######## reference frame
    
    refframe_new = data['data_obj'].reference_frame
    
    ####### "fitter-radio", 'n_jobs' no update
    ####### n_iter
    
    n_iternew = [len(data['hist_eps'])+1,len(data['hist_eps'])+1]
    
    #######ensemble size
    
    ens = data['model_kwargs']['ensemble_size']
    
    if ens == int(2**16):
        ens_new = 16
    elif ens == int(2**17):
        ens_new = 17
    elif ens == int(2**18):
        ens_new = 18
        
    ####### resulttab
    
    iparams_arrt = data["model_obj"].iparams_arr
    resdf = pd.DataFrame(iparams_arrt)
    
    rescols = resdf.columns.values.tolist()
    
    # drop first column
    resdf.drop(resdf.columns[[0]], axis=1, inplace=True)
    # rename columns
    resdf.columns = ['Longitude', 'Latitude', 'Inclination', 'Diameter 1 AU', 'Aspect Ratio', 'Launch Radius', 'Launch Velocity', 'T_Factor', 'Expansion Rate', 'Magnetic Decay Rate', 'Magnetic Field Strength 1 AU', 'Background Drag', 'Background Velocity']
    
    
    # Scatter plot matrix using go.Splom
    statsfig = go.Figure(data=go.Splom(
        dimensions=[dict(label=col, values=resdf[col]) for col in resdf.columns],
        diagonal_visible=False,
        marker=dict(size=5, symbol='cross', line=dict(width=1, color='black'),
                   ),
    showupperhalf = False))
    
    # Add histograms on the diagonal
    #for i in range(len(resdf.columns)):
    #    statsfig.add_trace(go.Histogram(x=resdf.iloc[:, i], xaxis=f"x{i + 1}", yaxis=f"y{i + 1}"))


    # Customizing the axis and grid styles
    statsfig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    statsfig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Update layout to control height and width
    statsfig.update_layout(height=2500, width=2500)
    

    # Add 'eps' column from data["epses"]
    resepses = data["epses"]
    num_rows = min(len(resepses), len(resdf))
    resdf.insert(0, 'RMSE Ɛ', resepses[:num_rows])

    # Calculate Twist number

    delta = resdf["Aspect Ratio"].values
    h = (delta - 1) ** 2 / (1 + delta) ** 2
    Efac = np.pi * (1 + delta) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))

    twist = resdf["T_Factor"].values / Efac

    resdf.insert(len(resdf.columns), 'Number of Twists', twist[:num_rows])      

    resdf['Launch Time'] = t0.strftime("%Y-%m-%d %H:%M")

    
    # Calculate statistics
    mean_values = np.mean(resdf, axis=0)
    std_values = np.std(resdf, axis=0)
    median_values = resdf.median()
    min_values = np.min(resdf, axis=0)
    max_values = np.max(resdf, axis=0)
    q1_values = resdf.quantile(0.25)
    q3_values = resdf.quantile(0.75)
    skewness_values = resdf.skew()
    kurtosis_values = resdf.kurt()
    
    mean_row = pd.DataFrame(
        [mean_values, std_values, median_values, min_values, max_values,
         q1_values, q3_values, skewness_values,
         kurtosis_values],
        columns=resdf.columns
    )

    mean_row['Launch Time'] = t0.strftime("%Y-%m-%d %H:%M")

    
    # Add the index column
    resdf.insert(0, 'Index', range(0, num_rows ))

    # Round all values to 2 decimal points
    resdfnew = round_dataframe(resdf)
    
    ###### stattab
    
   # Add the index column
    mean_row.insert(0, 'Index', ["Mean", "Standard Deviation", "Median", "Minimum", "Maximum", "Q1", "Q3", "Skewness", "Kurtosis"],)

    # Round all values to 2 decimal points
    mean_rownew = round_dataframe(mean_row)
    
    mean_row_df = pd.DataFrame([mean_rownew.iloc[0]], columns=resdf.columns)
    # Concatenate resdf and mean_row_df along the rows (axis=0) and reassign it to resdf
    resdffinal = pd.concat([resdfnew, mean_row_df], axis=0)

    return tablenew, *result_dict, *mag_result_dict, partslid_new, no_update, no_update, no_update, n_iternew, ens_new, resdffinal.to_dict("records"), t0, mean_rownew.to_dict("records"), statsfig

def round_dataframe(df):
    return df.applymap(lambda x: round(x, 2) if isinstance(x, float) else x)




def get_iparams(row):
    
    model_kwargs = {
        "ensemble_size": int(1), #2**17
        "iparams": {
            "cme_longitude": {
                "distribution": "fixed",
                "default_value": row['Longitude']
            },
            "cme_latitude": {
                "distribution": "fixed",
                "default_value": row['Latitude']
            },
            "cme_inclination": {
                "distribution": "fixed",
                "default_value": row['Inclination']
            },
            "cme_diameter_1au": {
                "distribution": "fixed",
                "default_value": row['Diameter 1 AU']
            },
            "cme_aspect_ratio": {
                "distribution": "fixed",
                "default_value": row['Aspect Ratio']
            },
            "cme_launch_radius": {
                "distribution": "fixed",
                "default_value": row['Launch Radius']
            },
            "cme_launch_velocity": {
                "distribution": "fixed",
                "default_value": row['Launch Velocity']
            },
            "t_factor": {
                "distribution": "fixed",
                "default_value": row['T_Factor']
            },
            "cme_expansion_rate": {
                "distribution": "fixed",
                "default_value": row['Expansion Rate']
            },
            "magnetic_decay_rate": {
                "distribution": "fixed",
                "default_value": row['Magnetic Decay Rate']
            },
            "magnetic_field_strength_1au": {
                "distribution": "fixed",
                "default_value": row['Magnetic Field Strength 1 AU']
            },
            "background_drag": {
                "distribution": "fixed",
                "default_value": row['Background Drag']
            },
            "background_velocity": {
                "distribution": "fixed",
                "default_value": row['Background Velocity']
            }
        }
    }
    
    row_launch = datetime.datetime.strptime(row['Launch Time'], "%Y-%m-%d %H:%M")
    
    return model_kwargs, row_launch



def get_iparams_live(*modelstatevars):
    
    model_kwargs = {
        "ensemble_size": int(1), #2**17
        "iparams": {
            "cme_longitude": {
                "distribution": "fixed",
                "default_value": modelstatevars[0]
            },
            "cme_latitude": {
                "distribution": "fixed",
                "default_value": modelstatevars[1]
            },
            "cme_inclination": {
                "distribution": "fixed",
                "default_value": modelstatevars[2]
            },
            "cme_diameter_1au": {
                "distribution": "fixed",
                "default_value": modelstatevars[3]
            },
            "cme_aspect_ratio": {
                "distribution": "fixed",
                "default_value": modelstatevars[4]
            },
            "cme_launch_radius": {
                "distribution": "fixed",
                "default_value": modelstatevars[5]
            },
            "cme_launch_velocity": {
                "distribution": "fixed",
                "default_value": modelstatevars[6]
            },
            "t_factor": {
                "distribution": "fixed",
                "default_value": modelstatevars[10]
            },
            "cme_expansion_rate": {
                "distribution": "fixed",
                "default_value": modelstatevars[7]
            },
            "magnetic_decay_rate": {
                "distribution": "fixed",
                "default_value": modelstatevars[11]
            },
            "magnetic_field_strength_1au": {
                "distribution": "fixed",
                "default_value": modelstatevars[12]
            },
            "background_drag": {
                "distribution": "fixed",
                "default_value": modelstatevars[8]
            },
            "background_velocity": {
                "distribution": "fixed",
                "default_value": modelstatevars[9]
            }
        }
    }
    
    return model_kwargs
    

    



def generate_ensemble(path: str, dt: datetime.datetime, posdata, reference_frame: str="HCI", reference_frame_to: str="HCI", perc: float=0.95, max_index=None) -> np.ndarray:
    
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

    
    ensemble_data = []
    

    try:
        ftobj = BaseMethod(path) # load Fitter from path
    except:
        ftobj = BaseMethod(path.replace('_ensembles_GSM', ''))

    # simulate flux ropes using iparams from loaded fitter
    ensemble = np.squeeze(np.array(ftobj.model_obj.simulator(dt, posdata)[0]))

    # how much to keep of the generated ensemble
    if max_index is None:
        max_index = ensemble.shape[1]

    ensemble = ensemble[:, :max_index, :]
    

    
    #ensemble[np.where(ensemble == 0)] = np.nan
    fig = go.Figure()
    # transform frame
    if reference_frame != reference_frame_to:
        x,y,z = hc.separate_components(posdata)        
        for k in range(0, ensemble.shape[1]):
            #if np.sum(~np.isnan(ensemble[:, k, :])) > 5000:
                #print(str(k)+': ')
                #print(np.sum(~np.isnan(ensemble[:, k, :])))
                #print(ensemble[:, k, :])
            sys.stdout.write(f"\r{k+1}/{ensemble.shape[1]}")
            sys.stdout.flush()
            bx,by,bz = hc.separate_components(ensemble[:, k, :])
            #plt.figure()
            
            #plt.show()

            rtn_bx, rtn_by, rtn_bz = hc.convert_HEEQ_to_RTN_mag(x,y,z, bx,by,bz, printer = False)
            fig.add_trace(go.Scatter(x=dt, y=rtn_bx,
                    mode='lines'))
            if reference_frame_to == "RTN":
                ensemble[:, k, :] = hc.combine_components(rtn_bx, rtn_by, rtn_bz)

            if reference_frame_to == "GSM":
                gsm_bx, gsm_by, gsm_bz = dft.RTN_to_GSM(x, y, z, rtn_bx,rtn_by,rtn_bz, dt)
                ensemble[:, k, :] = hc.combine_components(gsm_bx, gsm_by, gsm_bz)
        # Print a new line after progress is complete
        print()
    fig.show()
    
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
        
    return ensemble_data, ensemble



class BepiPredict(RemoteKernelsBase):
    '''
    enable handling Bepi Positions
    '''
    
    body = 'mpo'
    type = 'predict'

    def get_remote_kernels(self):
        """
        Returns
        -------
        list[RemoteKernel]
        """
        page = urlopen('https://naif.jpl.nasa.gov/pub/naif/BEPICOLOMBO/kernels/spk/')
        soup = BeautifulSoup(page, 'html.parser')

        kernel_urls = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href is not None and href.startswith('bc'):
                fname = href.split('/')[-1]
                matches = self.matches(fname)
                if matches:
                    kernel_urls.append(
                        RemoteKernel(f'https://naif.jpl.nasa.gov/pub/naif/BEPICOLOMBO/kernels/spk/{href}', *matches[1:]))

        return kernel_urls

    @staticmethod
    def matches(fname):
        """
        Check if the given filename matches the pattern of this kernel.

        Returns
        -------
        matches : bool
        start_time : astropy.time.Time
        end_time : astropy.time.Time
        version : int
        """
        # Example filename: bc_mpo_fcp_00154_20181020_20251102_v01.bsp 
        fname = fname.split('_')
        if (len(fname) != 7 or
                fname[0] != 'bc' or
                fname[1] != 'mpo' or
                fname[2] != 'fcp'):
            return False

        start_time = Time.strptime(fname[4], '%Y%m%d')
        end_time = Time.strptime(fname[5], '%Y%m%d')
        version = int(fname[6][1:3])
        return True, start_time, end_time, version



def round_to_hour_or_half(dt):
    
    '''
    round launch datetime
    '''
    
    remainder = dt.minute % 30
    if remainder < 15:
        dt -= datetime.timedelta(minutes=remainder)
    else:
        dt += datetime.timedelta(minutes=30 - remainder)
    return dt.replace(second=0, microsecond=0)



    
