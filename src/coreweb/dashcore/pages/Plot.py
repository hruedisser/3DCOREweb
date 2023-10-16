from dash import dcc, html, Input, Output, State, callback, register_page
import dash_mantine_components as dmc
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly_express as px
import plotly.figure_factory as ff

import numpy as np

import matplotlib 

import datetime

import plotly.graph_objs as go

import coreweb
from coreweb.dashcore.assets.config_sliders import modelstate
from coreweb.dashcore.utils.utils import *

import coreweb.dashcore.utils.heliocats as hc

from dash_iconify import DashIconify
import dash_bootstrap_components as dbc

register_page(__name__, icon="streamline:money-graph-analytics-business-product-graph-data-chart-analysis", order=2)

#matplotlib.use('agg')

app = dash.get_app()

reload_icon = dmc.ThemeIcon(
                     DashIconify(icon='ci:arrows-reload-01', style={"color": "black"}),
                     size=40,
                     radius=40,
                     variant="light",
                     style={"backgroundColor": "#eaeaea", "marginRight": "12px"},
                 )

plotoptionrow = dbc.Row([
            dbc.Col(
                dmc.CheckboxGroup(
                    id="plotoptions_posfig",
                    label="Options for plotting",
                    orientation="horizontal",
                    withAsterisk=False,
                    offset="md",
                    mb=10,
                    children=[
                        dmc.Checkbox(label="Title", value="title", color="green"),
                        dmc.Checkbox(label="Latitudinal Grid", value="latgrid", color="green"),
                        dmc.Checkbox(label="Longitudinal Grid", value="longgrid", color="green"),
                        dmc.Checkbox(label="Trajectories", value="trajectories", color="green", disabled = False),
                        dmc.Checkbox(label="Synthetic Event", value="cme", color="green"),
                        dmc.Checkbox(label="Catalog Event", value="catalogevent", color="green"),
                    ],
                    value=["longgrid", "title", "trajectories", "cme", "catalogevent"],
                ),
                width=10,
            ),
                dbc.Col(
                [
                    dmc.SegmentedControl(
                        id = "segmented",
                        value="2D",
                        data = [
                            {"value": "2D", "label": "2D"},
                            {"value": "3D", "label": "3D"},
                            {"value": "coronograph", "label": "📹", "disabled": True},
                        ],
                        style={"marginRight": "12px"},
                    )
                ],
                width=2,  # Adjust the width of the right-aligned columns
                style={"display": "flex", "justify-content": "flex-end"},
            ),
        ], justify="between",  # Distribute the columns
           align="end",  # Align content at the bottom of the row
        )



bodyoptionrow = dbc.Row([
            dbc.Col(
                dmc.CheckboxGroup(
                    id="bodyoptions_posfig",
                    label="Bodies",
                    orientation="horizontal",
                    withAsterisk=False,
                    offset="md",
                    mb=10,
                    children=[
                        dmc.Checkbox(label="Sun", value="showsun", color="green"),
                        dmc.Checkbox(label="Mercury", value="mercury", color="green"),
                        dmc.Checkbox(label="Venus", value="venus", color="green"),
                        dmc.Checkbox(label="Earth", value="earth", color="green"),
                        #dmc.Checkbox(label="Mars", value="mars", color="green"),
                    ],
                    value=["showsun", "earth", "mercury", "venus"],
                ),
                width=12,  # Adjust the width of the CheckboxGroup column
            ),
            
        ], justify="start",  
           align="end",  # Align content at the bottom of the row
        )

spacecraftoptionrow = dbc.Row([
            dbc.Col(
                dmc.CheckboxGroup(
                    id="spacecraftoptions_posfig",
                    label="Spacecraft",
                    orientation="horizontal",
                    withAsterisk=False,
                    offset="md",
                    mb=10,
                    children=[
                        dmc.Checkbox(label="SOLO", value="SOLO", color="green"),
                        dmc.Checkbox(label="PSP", value="PSP", color="green"),
                        dmc.Checkbox(label="BEPI", value="BEPI", color="green"),
                        dmc.Checkbox(label="Wind", value="Wind", color="green"),
                        dmc.Checkbox(label="STEREO-A", value="STEREO-A", color="green"),
                        dmc.Checkbox(label="Synthetic Spacecraft", value="SYN", color="green"),
                    ],
                ),
                width=8,  # Adjust the width of the CheckboxGroup column
            ),
                                     
                ], justify="between",  
                   align="end",  # Align content at the bottom of the row
                )







layout = html.Div(
    [
        plotoptionrow,
        bodyoptionrow,
        spacecraftoptionrow,
        dbc.Row(
            [
                dmc.Text("Δt"), 
                dcc.Slider(
                    id="time_slider",
                    min=0,
                    max=168,
                    step=0.5,
                    value=5,
                    marks = {i: '+' + str(i)+'h' for i in range(0, 169, 12)},
                    persistence=True,
                    persistence_type='session',
                ),
            ],
            id = 'timesliderdiv'
        ),
        dbc.Row(
            [
                dmc.Text("Δt"), 
                dcc.Slider(
                    id="corono_slider",
                    min=0,
                    max=12,
                    step=0.5,
                    value=5,
                    marks = {i: '+' + str(i)+'h' for i in range(0, 13)},
                    persistence=True,
                    persistence_type='session',
                ),
            ],
            id = 'coronosliderdiv'
        ),
        dcc.Graph(id="posfig", style={'display': 'none'}),  # Hide the figure by default        
        dcc.Graph(id="sliderfiginsitu"),
    
        ]
)


@callback(
    Output("plotoptions_posfig", "children"),
    Output("timesliderdiv", "style"),
    Output("coronosliderdiv", "style"),
    Output("spacecraftoptions_posfig", "style"),
    Input("segmented", "value"),
    State("posstore", "data"),
)
def update_plot_options(dim, posstore):
    if dim == "2D":
        plot_options = [
            dmc.Checkbox(label="Title", value="title", color="green"),
            dmc.Checkbox(label="AU axis", value="axis", color="green"),
            dmc.Checkbox(label="Trajectories", value="trajectories", color="green", disabled=False),
            dmc.Checkbox(label="Parker Spiral", value="parker", color="green", disabled=False),
            dmc.Checkbox(label="Synthetic Event", value="cme", color="green"),
            dmc.Checkbox(label="Catalog Event", value="catalogevent", color="green"),
        ]
        timeslider_style = {"visibility": "hidden"}
        coronoslider_style = {"visibility": "hidden"}
        spacecraft_options_style = {"visibility": "visible"}
        plot_options_posfig_style = {
            "visibility": "visible",
            "min-width": "250px",
            "marginLeft": "0px",
            "marginRight": "12px",
        }

    elif dim == "3D":
        plot_options = [
            dmc.Checkbox(label="Title", value="title", color="green"),
            dmc.Checkbox(label="Timer", value="datetime", color="green"),
            dmc.Checkbox(label="Latitudinal Grid", value="latgrid", color="green"),
            dmc.Checkbox(label="Longitudinal Grid", value="longgrid", color="green"),
            dmc.Checkbox(label="Trajectories", value="trajectories", color="green", disabled=False),
            dmc.Checkbox(label="Synthetic Event", value="cme", color="green"),
            dmc.Checkbox(label="Catalog Event", value="catalogevent", color="green"),
        ]
        timeslider_style = {"visibility": "visible"}
        coronoslider_style = {"visibility": "hidden"}
        spacecraft_options_style = {"visibility": "visible"}
        plot_options_posfig_style = {
            "visibility": "visible",
            "min-width": "250px",
            "marginLeft": "0px",
            "marginRight": "12px",
        }

    elif dim == "coronograph":
        plot_options = []
        timeslider_style = {"visibility": "hidden"}
        coronoslider_style = {"visibility": "visible"}
        spacecraft_options_style = {"visibility": "hidden"}
        plot_options_posfig_style = {"visibility": "hidden"}

    else:
        plot_options = []
        timeslider_style = {"visibility": "hidden"}
        coronoslider_style = {"visibility": "visible"}
        spacecraft_options_style = {"visibility": "visible"}
        plot_options_posfig_style = {
            "visibility": "visible",
            "min-width": "250px",
            "marginLeft": "0px",
            "marginRight": "12px",
        }

    return (
        plot_options,
        timeslider_style,
        coronoslider_style,
        spacecraft_options_style,
    )






@callback(
    Output("spacecraftoptions_posfig", "children"),
    Output("spacecraftoptions_posfig", "value"),
    [Input("posstore", "data")]
)
def update_spacecraft_options(posstore):
    spacecraft_options = [
        dmc.Checkbox(label="SOLO", value="SOLO", color="green", disabled = "SOLO" not in posstore, checked="SOLO" in posstore
),
        dmc.Checkbox(label="PSP", value="PSP", color="green", disabled = "PSP" not in posstore, checked="PSP" in posstore),
        dmc.Checkbox(label="BEPI", value="BEPI", color="green", disabled = "BEPI" not in posstore, checked="BEPI" in posstore),
        dmc.Checkbox(label="Wind", value="Wind", color="green", disabled = "Wind" not in posstore, checked="Wind" in posstore),
        dmc.Checkbox(label="STEREO-A", value="STEREO-A", color="green", disabled = "STEREO-A" not in posstore, checked="STEREO-A" in posstore),
        dmc.Checkbox(label="Synthetic Spacecraft", value="SYN", color="green"),
    ],
    
    # Create updated spacecraft options list with disabled and checked properties
    spacecraft_options_value = [key for key in posstore]
    
    return spacecraft_options[0], spacecraft_options_value



@app.long_callback(
    Output("posfig", "figure"),
    Output("posfig", "style"),  # Add this Output to control the display style
    Output("sliderfiginsitu", "figure"),
    Output("time_slider", "marks"),
    State("posstore", "data"),
    State("rinput", "value"),
    State("lonput", "value"),
    State("latput", "value"),
    Input('reload-button', "n_clicks"),
    Input("toggle-range", "value"),
    Input("time_slider", "value"),
    Input("segmented", "value"),
    Input("graphstore", "data"),
    Input("event-info", "data"),
    Input("launch-label", "children"),
    Input("plotoptions_posfig", "value"),
    Input("spacecraftoptions_posfig", "value"),
    Input("bodyoptions_posfig", "value"),
    Input("reference_frame", "value"),
    *[
            Input(id, "value") for id in modelstate
        ],
    running=[
        (Output(id, "disabled"), True, False) for id in modelstate 
    ],
    
)
def update_posfig(posstore, rinput, lonput, latput, nclicks, togglerange, timeslider, dim, graph, infodata, launchlabel,plotoptions, spacecraftoptions, bodyoptions, refframe, *modelstatevars):
    
    if launchlabel == "Launch Time:":
        raise PreventUpdate

    marks = {i: '+' + str(i)+'h' for i in range(0, 169, 12)}
    
    if "catalogevent" in plotoptions:
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
                
    if "cme" in plotoptions:
        iparams = get_iparams_live(*modelstatevars)
        model_obj = coreweb.ToroidalModel(roundedlaunch, **iparams) # model gets initialized
        model_obj.generator()
        
    
    
    ################################################################
    ############################ INSITU ############################
    ################################################################
                
    
    if (graph is {}) or (graph is None): 
        insitufig = {}
    
    else:
        try:
            insitufig = go.Figure(graph['fig'])
        except:
            raise PreventUpdate   
        
        
        if "title" in plotoptions:
            insitufig.update_layout(title=infodata['id'][0]+'_'+refframe)
        
        if "catalogevent" in plotoptions:
            insitufig.add_vrect(
                    x0=begin,
                    x1=end,
                    fillcolor="LightSalmon", 
                    opacity=opac,
                    layer="below",
                    line_width=0
            )
            
        if "cme" in plotoptions:
            # Create ndarray with dtype=object to handle ragged nested sequences
            if sc == "SYN":
                try:
                    # Calculate the desired length
                    desired_length = len(graph['t_data'])

                    # Create an array with NaN values
                    pos_array = np.empty((desired_length, 3))

                    pos_array[:, 0], pos_array[:, 1], pos_array[:, 2] = sphere2cart(rinput, np.deg2rad(-latput+90), np.deg2rad(lonput))  # Assign a different value (e.g., 0.9) to the third vector
                    
                    #print(pos_array)

                    outa = np.array(model_obj.simulator(graph['t_data'], pos_array), dtype=object)
                    
                    
                
                except:
                    pass
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
        
        if "cme" in plotoptions:
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
            
            
        if "catalogevent" in plotoptions:
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

        if "showsun" in bodyoptions:

            # Create data for the Sun
            sun_trace = go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=8, color='yellow'),
                name='Sun'
            )

            fig.add_trace(sun_trace)

        if "earth" in bodyoptions:

            # Create data for the Earth
            earth_trace = go.Scatter3d(
                x=[1], y=[0], z=[0],
                mode='markers',
                marker=dict(size=4, color='mediumseagreen'),
                name='Earth'
            )

            fig.add_trace(earth_trace)
                        
        if "mercury" in bodyoptions:
            fig.add_trace(plot_body3d(graph['bodydata']['Mercury']['data'], roundedlaunch + datetime.timedelta(hours=timeslider), 'slategrey', 'Mercury')[0])
            
            
        if "venus" in bodyoptions:
            fig.add_trace(plot_body3d(graph['bodydata']['Venus']['data'], roundedlaunch + datetime.timedelta(hours=timeslider), 'darkgoldenrod', 'Venus')[0])
            
        if "mars" in bodyoptions:
            fig.add_trace(plot_body3d(graph['bodydata']['Mars']['data'], roundedlaunch + datetime.timedelta(hours=timeslider), 'red', 'Mars')[0])
            
        if spacecraftoptions is not None:
            for scopt in spacecraftoptions:
                if scopt == "SYN":
                    try:
                        
                        x,y,z = sphere2cart(rinput, np.deg2rad(-latput+90), np.deg2rad(lonput))
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
                    except:
                        pass
                    
                else:                    
                    traces = process_coordinates(posstore[scopt]['data']['data'], roundedlaunch, roundedlaunch + datetime.timedelta(hours=timeslider), posstore[scopt]['data']['color'], scopt)

                    if "trajectories" in plotoptions:
                        fig.add_trace(traces[0])
                        fig.add_trace(traces[1])

                    fig.add_trace(traces[2])


        if "longgrid" in plotoptions:
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
                
        if "latgrid" in plotoptions:
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

        if "title" in plotoptions:
            fig.update_layout(title=str(roundedlaunch + datetime.timedelta(hours=timeslider)))
            fig.add_annotation(text="HEEQ", xref="paper", yref="paper", x=1.1, y=0.1, showarrow=False)
            
        if "datetime" in plotoptions:
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
        
        if "axis" in plotoptions:
            showticks = True
        else:
            showticks = False
            
            
        if "parker" in plotoptions:
                
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
                if scopt == "SYN":
                    fig.add_trace(
                        go.Scatterpolar(
                            r=[rinput], 
                            theta=[lonput], 
                            mode='markers', 
                            marker=dict(size=8, symbol='square', color='red'), 
                            name="SYN",
                            showlegend=False, 
                            hovertemplate="%{r:.1f} AU<br>%{theta:.1f}°<extra>" + "SYN" + "</extra>"
                        )
                    )
                else:            
                    if "trajectories" in plotoptions:
                        fig.add_trace(posstore[scopt]['traces'][0])
                        fig.add_trace(posstore[scopt]['traces'][1])

                    fig.add_trace(posstore[scopt]['traces'][2])

        if "showsun" in bodyoptions:
            # Add the sun at the center
            fig.add_trace(go.Scatterpolar(r=[0], theta=[0], mode='markers', marker=dict(color='yellow', size=10, line=dict(color='black', width=1)), showlegend=False, hovertemplate="%{r:.1f} AU<br>%{theta:.1f}°"+
                    "<extra></extra>"))
            # Add label "Sun" next to the sun marker
            fig.add_trace(go.Scatterpolar(r=[0.03], theta=[15], mode='text', text=['Sun'],textposition='top right', showlegend=False, hovertemplate = None, hoverinfo = "skip", textfont=dict(color='black', size=14)))


        if "earth" in bodyoptions:# Add Earth at radius 1
            fig.add_trace(go.Scatterpolar(r=[1], theta=[0], mode='markers', marker=dict(color='mediumseagreen', size=10), showlegend=False, hovertemplate="%{r:.1f} AU<br>%{theta:.1f}°"+
                    "<extra></extra>"))
            fig.add_trace(go.Scatterpolar(r=[1.03], theta=[1], mode='text', text=['Earth'],textposition='top right', name = 'Earth', showlegend=False, hovertemplate = None, hoverinfo = "skip",  textfont=dict(color='mediumseagreen', size=14)))
        
        
        try:
            if "mercury" in bodyoptions:
                fig.add_trace(graph['bodytraces'][0][0])
                fig.add_trace(graph['bodytraces'][1])
            if "venus" in bodyoptions:
                fig.add_trace(graph['bodytraces'][2][0])
                fig.add_trace(graph['bodytraces'][3])

            if "mars" in bodyoptions:
                fig.add_trace(graph['bodytraces'][4][0])
                fig.add_trace(graph['bodytraces'][5])
        except:
            pass
            
            
        if "title" in plotoptions:
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
        
    else:
        # Specify the time range for downloading
        start_time = "2022-08-01"
        end_time = "2022-08-02"

        # Define the time range
        time_range = a.Time(start_time, end_time)

        # Create a query for SOHO LASCO images
        lasco_query = Fido.search(time_range, a.Instrument.lasco)
        
        print(type(lasco_query))
        print(lasco_query)

        # Download the data
        #files = Fido.fetch(lasco_query, max_conn=1)
        # Plot the image
        fig = plt.figure()
        
        x = np.arange(0,4*np.pi,0.1)   # start,stop,step
        y = np.sin(x)
        
        
        plt.plot(x,y)
    
        
        fig = tls.mpl_to_plotly(fig)
    
    return fig, {'display': 'block'}, insitufig, marks