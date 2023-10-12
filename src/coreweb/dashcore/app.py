import dash
from dash import dcc, html, Output, Input, State, callback, long_callback, ctx
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import dash_bootstrap_components as dbc
from dash.long_callback import DiskcacheLongCallbackManager
from dash.exceptions import PreventUpdate
import dash_daq as daq

import base64

import plotly.graph_objs as go

import pickle
import os 
import sys
import pandas as pd
import json

import re
import time
import datetime

from coreweb.dashcore.utils.utils import *
from coreweb.dashcore.utils.plotting import *
from coreweb.dashcore.utils.main_fitting import *

from coreweb.dashcore.assets.config_sliders import *

from coreweb.methods.abc_smc import abc_smc_worker
from coreweb.model import SimulationBlackBox, set_random_seed
from coreweb.methods.data import FittingData

from heliosat.util import sanitize_dt
from typing import Any, Optional, Sequence, Tuple, Union

import multiprocess as mp # ing as mp
import diskcache

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = dash.Dash(__name__, use_pages=True,external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],long_callback_manager=long_callback_manager)

app.config.suppress_callback_exceptions = True

################# components
############################

success_icon = dmc.ThemeIcon(
                     DashIconify(icon='mdi:check-bold', width=18, color="black"),
                     size=40,
                     radius=40,
                     style={"backgroundColor": "#eaeaea"},
                 )

fail_icon = dmc.ThemeIcon(
                     DashIconify(icon='ph:x-bold', width=18, color="black"),
                     size=40,
                     radius=40,
                     style={"backgroundColor": "#eaeaea"},
                 )

# Create a manager to share data between processes
manager = mp.Manager()
processes = []

launchdatepicker = html.Div(
    [
    html.Label(id="launch-label", 
               children="Launch Time:", 
               style={"font-size": "12px", 
                     }),
    html.Br(),
    dcc.Slider(
        id="launch_slider",
        min=-120,
        max=-2,
        step=0.5,
        value=-24,
        marks = {i: str(i)+'h' for i in [-2, -24, -48, -72, - 96, -120]},
        persistence=True,
        persistence_type='session',
    ),
    
    ],
    style= {"margin": "20px",
                    "maxWidth": "310px",
                    "overflowX": "auto",
                    "whiteSpace": "nowrap",
                },
)




reference_frame = html.Div(
    [
        dbc.Row(
            [
                html.Label(children="Reference Frame:", style={"font-size": "12px"}),
                html.Br(),
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id="reference_frame",
                            options=[
                                {"label": "HEEQ", "value": "HEEQ"},
                                {"label": "RTN", "value": "RTN"},
                            ],
                            value="RTN",
                            clearable = False,
                            persistence = True,
                        ),
                    ],
                    width={"size": 6},  # Adjust the width as needed
                ),
                dbc.Col(
                    dmc.SegmentedControl(
                        id = "toggle-range",
                        value=0,
                        data = [
                            {"value": 0, "label": "←"},
                            {"value": 180, "label": "→"},
                        ],
                        color="white",
                        style={"marginRight": "12px"},
                    ),
                    width={"size": 3},  # Adjust the width as needed
                ),
                dbc.Col(
                    html.Div(id = "loadgraphstoreicon", 
                             children = "",
                             style={"marginRight": "12px", "marginLeft": "12px"},
                            ),
                    width={"size": 3},  # Adjust the width as needed
                ),
            ],
            justify="between",  # Distribute the columns to the edges of the row
        ),
    ],
    style={"marginLeft": "20px", "marginBottom": "20px", "marginRight": "20px","maxWidth": "310px", "whiteSpace": "nowrap"},
    className="mb-3",
)



modelsliders = html.Div(
    [create_single_slider(
        var['min'],
        var['max'],
        var['def'],
        var['step'],
        var['var_name'],
        var['variablename'],
        var['variablename']+ 'label',
        var['marks'],
        var['unit']
    ) for var in modelslidervars],
    style= {"margin": "20px",
                    "maxWidth": "310px",
                    "overflowX": "auto",
                    "whiteSpace": "nowrap",
                },
)


magsliders = html.Div(
    [create_single_slider(
        var['min'],
        var['max'],
        var['def'],
        var['step'],
        var['var_name'],
        var['variablename'],
        var['variablename']+ 'label',
        var['marks'],
        var['unit']
    ) for var in magslidervars],
    style= {"margin": "20px",
                    "maxWidth": "310px",
                    "overflowX": "auto",
                    "whiteSpace": "nowrap",
                },
)

##################### Layout
############################


topbar = dmc.Navbar(
    height=110,
    style={"backgroundColor": "#f8f9fa"},
    children=[
        html.Div(
            style={"display": "flex", "gap": "20px", "justifyContent": "flex-end" },
            children=[
                    create_nav_link(
                        icon=page["icon"], label=page["name"], href=page["path"]
                    )
                    for page in dash.page_registry.values() 
                ],
        ),
    ],
)

reload_icon = dmc.ThemeIcon(
                     DashIconify(icon='ci:arrows-reload-01', style={"color": "black"}),
                     size=40,
                     radius=40,
                     variant="light",
                     style={"backgroundColor": "#eaeaea", "marginRight": "12px"},
                 )

sidebar = dmc.Navbar(
    fixed=True,
    width={"base": 350},
    position={"top": 1, "left": 1},
    height=2500,
    style={"backgroundColor": "#f8f9fa","maxHeight": "calc(100vh - 0px)", "overflowY": "auto"},  # Set maximum height and enable vertical overflow
            
    children=[
        dmc.ScrollArea(
            offsetScrollbars=True,
            type="scroll",
            style={"height": "100%"},  # Set the height of the scroll area
            children=[
                html.A(
                    html.H2(
                        "3DCOREweb",
                        className="display-4",
                        style={"marginBottom": 20, "marginLeft": 20, "marginTop": 20},
                    ),
                    href="https://iopscience.iop.org/article/10.3847/1538-4365/abc9bd/meta",
                    target="_blank",  # Open link in a new tab
                    style={"textDecoration": "none", "color": "inherit"},
                ),
                html.Hr(style={"marginLeft": 20}),
                html.P(
                    "Reconstruct CMEs using the 3D Coronal Rope Ejection Model",
                    className="lead",
                    style={"marginLeft": 20},
                ),
                dmc.Divider(
                    label="CME Event",
                    style={"marginBottom": 20, "marginTop": 20, "marginLeft": 20, "marginRight": 20},
                ),
                dbc.Button(
                    id="event-alert-div",
                    className="alert alert-primary",
                    children="No event selected",
                    style={
                        "marginLeft": "20px",
                        "marginRight": "20px",
                        "maxWidth": "310px",
                        "minWidth": "310px",
                        "overflowX": "auto",
                        "whiteSpace": "nowrap",
                    },
                    title="Download"
                ),
                reference_frame,                
                launchdatepicker,
                dbc.Row(dbc.Col(
                    [
                        dbc.Button(
                            children=[reload_icon],
                            id='reload-button',
                            n_clicks=0,  # Initialize the click count to 0
                            style={"border": "none", 
                                   "background-color": "transparent", 
                                   "padding": "0",  # Set padding to 0
                                   "width": "auto",  # Allow the button to adjust its width based on content
        
                                   "visibility": "hidden", 
                                  },
                        ),
                        dbc.FormFloating(
                            [
                                dbc.Input(id="rinput",type="number", value=0.8, style={"width": "80px"}, persistence=True, persistence_type='session',),
                                dbc.Label("r [AU]", style={"font-size": "12px"}),
                            ],
                            id="r_input",
                            style={"height": "30px", 
                                   "visibility": "hidden"
                                  },  # Initially hidden
                        ),
                        dbc.FormFloating(
                            [
                                dbc.Input("lonput",type="number", value=45, style={"width": "80px"}, persistence=True, persistence_type='session',),
                                dbc.Label("lon [°]", style={"font-size": "12px"}),
                            ],
                            id="lon_input",
                            style={"height": "30px", 
                                   "visibility": "hidden"
                                  },  # Initially hidden
                        ),
                        dbc.FormFloating(
                            [
                                dbc.Input("latput",type="number", value=0, style={"width": "80px"}, persistence=True, persistence_type='session',),
                                dbc.Label("lat [°]", style={"font-size": "12px"}),
                            ],
                            id="lat_input",
                            style={"height": "30px", 
                                   "visibility": "hidden"
                                  },  # Initially hidden
                        ),
                    ],
                    width=12,  # Adjust the width of the right-aligned columns
                    style={"display": "flex", "marginLeft": 20, "marginBottom": 20, 
                          },
                    
                )),
                
                
                dmc.Divider(
                    label="Model Parameters",
                    style={"marginBottom": 20, "marginTop": 20, "marginLeft": 20, "marginRight": 20},
                ),
                modelsliders,
                dmc.Divider(
                    label="Magnetic Field Parameters",
                    style={"marginBottom": 20, "marginTop": 20, "marginLeft": 20, "marginRight": 20},
                ),
                magsliders,
            ],
        )
    ],
)

app.layout = dmc.Container(
    [dcc.Location(id='url', refresh=False),
     topbar,
     sidebar,
     dmc.Container(
         dash.page_container,
         size="lg",
         pt=20,
         style={"marginLeft": 340,
                "marginTop": 20},
     ),
    # dcc.Store stores the event info
    dcc.Store(id='event-info', storage_type='local'),
    dcc.Store(id='graphstore', storage_type='local'),
    dcc.Store(id='posstore', storage_type='local'),
    dcc.Download(id="download-model"),
    dmc.Container([
        dbc.Row([
            dbc.Col([], width=6),
            dbc.Col([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files', style={'font-weight': 'bold'})
                    ]),
                    style={
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '.5px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'borderColor': 'gray',
                        'margin-left': '20px',
                        'margin-right': '30px',
                    },
                    multiple=True
                ),
            ], width=6),
         ],
        ),
    ],
         id = 'upload-container',
         size="lg",
         pt=20,
         style={"marginLeft": 340,
                "visibility": "hidden",
               },
    ),
     dmc.Container([
         dataarchive,
    ],
         id = 'dataarchive',
         size="lg",
         pt=20,
         style={"marginLeft": 340,
               },
    ),
    ],
    fluid=True,
)

################## callbacks
############################


@callback(
    Output("download-model", "data"),
    Input("event-alert-div", "n_clicks"),
    State("rinput", "value"),
    State("lonput", "value"),
    State("latput", "value"),
    State("event-info", "data"),
    State("launch-label", "children"),
    State("launch_slider", "value"),
    State("reference_frame","value"),
    *[
            State(id, "value") for id in modelstate
        ],
)
def download_model(n_clicks, rinput, lonput, latput, eventinfo, launchlabel, launchvalue, refframe, *modelstatevars):
    if (n_clicks == None) or (n_clicks) == 0:
        raise PreventUpdate
    
    if (launchlabel == 'Launch Time:'):
        raise PreventUpdate
        
    datetime_format = "Launch Time: %Y-%m-%d %H:%M"
    t_launch = datetime.datetime.strptime(launchlabel, datetime_format)
    roundedlaunch = round_to_hour_or_half(t_launch) 
    
    iparams = get_iparams_live(*modelstatevars)
    
    contents = {
        'modelstatevars': modelstatevars,
        'iparams': iparams,
        'roundedlaunch': roundedlaunch.strftime(datetime_format),  # Convert to string
        'launchvalue': launchvalue,
        'refframe': refframe,
        'eventinfo': eventinfo,    
        'rinput': rinput,    
        'lonput': lonput,    
        'latput': latput,    
    }
    
    # Convert the contents dictionary to a JSON string
    contents_json = json.dumps(contents)
    
    return dict(content=contents_json, filename="model_"+eventinfo['id'][0]+".txt")

@callback(
    Output("launch-label", "children"),
    Output("event-alert-div", "children"),
    Input("event-info", "data"),
    Input("launch_slider", "value"),
)
def update_launch_label(data, slider_value):
    if data == None:
        return "Launch Time:", 'No event selected'
    else:
        try:
            datetime_value = data['begin'][0]
            dateFormat = "%Y-%m-%dT%H:%M:%S%z"
            dateFormat2 = "%Y-%m-%d %H:%M:%S"
            dateFormat3 = "%Y-%m-%dT%H:%M:%S"
            try:
                input_datetime = datetime.datetime.strptime(datetime_value, dateFormat2)
            except ValueError:
                try:
                    input_datetime = datetime.datetime.strptime(datetime_value, dateFormat)
                except ValueError:
                    input_datetime = datetime.datetime.strptime(datetime_value, dateFormat3)

            hours = slider_value
            launch = input_datetime + datetime.timedelta(hours=hours)
            roundedlaunch = round_to_hour_or_half(launch) 
            launch_formatted = roundedlaunch.strftime("%Y-%m-%d %H:%M")
            return f"Launch Time: {launch_formatted}", data['id']
        except:
            return "Launch Time:", 'No event selected'
    
@app.callback(
    Output('longit', 'min'),
    Output('longit', 'max'),
    [Input('toggle-range', 'value')]
)
def update_slider_range(toggle_value):
    if toggle_value == 0:
        return -180, 180
    else:
        return 0, 360
    
##slider callbacks

def create_callback(var, index):
    html_for = var['variablename'] + 'label'
    ids = var['variablename']
    unit = var['unit']
    func_name = f"update_slider_label{index}"
    
    def callback_func(value, label=f"{var['var_name']}"):
        return f"{label}: {value} {unit}"
    
    callback_func.__name__ = func_name
    app.callback(Output(html_for, "children"), [Input(ids, "value")])(callback_func)
    
callbacks = []

for i, var in enumerate(modelslidervars):
    create_callback(var, i)
    
for i, var in enumerate(magslidervars):
    create_callback(var, i + len(modelslidervars))
    
##############long callbacks
############################

def starmap(func, args):
    return [func(*_) for _ in args]

def save(path: str, extra_args: Any) -> None:
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        print('created directory ' + path)
    
    with open(path, "wb") as fh:
        pickle.dump(extra_args, fh)
        print('saved to ' + path)
        
        
        
# Create a callback to control the visibility of the floating forms
@callback(
    Output("r_input", "style"),
    Output("lon_input", "style"),
    Output("lat_input", "style"),
    Output('reload-button', "style"),
    Input("spacecraftoptions_posfig", "value")
)
def update_floating_forms(spacecraft_options):
    # Check if "Synthetic Spacecraft" is in the list of selected spacecraft options
    synthetic_spacecraft_checked = "SYN" in spacecraft_options
    
    # Update the visibility style of the floating forms accordingly
    visibility_style = {"height": "30px", "visibility": "visible"} if synthetic_spacecraft_checked else {"height": "30px", "visibility": "hidden"}
    
    button_style = {"border": "none", 
                    "background-color": "transparent", 
                    "padding": "0",  # Set padding to 0
                    "width": "auto",  # Allow the button to adjust its width based on content
                    "visibility": "visible"} if synthetic_spacecraft_checked else {"border": "none", "background-color": "transparent", "visibility": "hidden", "marginRight": "5px"}

    return visibility_style, visibility_style, visibility_style, button_style


        
@app.long_callback(
    Output("insitufitgraph", "figure"),
    Input("spacecrafttable", "cellValueChanged"),
    Input("graphstore", "data"),
    Input("plotoptions", "value"),
    Input("spacecrafttable", "rowData"),
    Input("event-info", "data"),
    State("restable", "selectedRows"),
    Input("plt-synresults", "n_clicks"),
    State("launch-label", "children"),
    State("fit_dropdown", "value"),
    State("reference_frame", "value"),
    running = [
        (Output("insitufitspinner", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"}
        ), 
    ]
)
def plot_insitufig(_, graph, plotoptions, tabledata, infodata, selectedrows, nclicks, launchlabel, name, refframe):
    
    if (graph is {}) or (graph is None):  # This ensures that the function is not executed when no figure is present
        fig = {}
        return fig
    
    try:
        fig = go.Figure(graph['fig'])
    except:
        raise PreventUpdate       
    
    triggered_id = ctx.triggered_id
    
    sc = infodata['sc'][0]
    
    if triggered_id == None:
        raise PreventUpdate
        
    if "fittingresults" in plotoptions:
        
        filepath = loadpickle(name)
    
        # read from pickle file
        file = open(filepath, "rb")
        data = p.load(file)
        file.close()
        
        #refframe = data['data_obj'].reference_frame
        
        if sc == "SYN":
            raise PreventUpdate
            
        else:
            ensemble_filepath = filepath.split('.')[0] + '_ensembles.pickle'
            with open(ensemble_filepath, 'rb') as ensemble_file:
                ensemble_data = pickle.load(ensemble_file)    
    
            if refframe == 'HEEQ':
                ed = ensemble_data['ensemble_HEEQ']
            else:
                ed = ensemble_data['ensemble_RTN']
        
        shadow_data = [
            (ed[0][3][0], None, 'black'),
            (ed[0][3][1], 'rgba(0, 0, 0, 0.15)', 'black'),
            (ed[0][2][0][:, 0], None, 'red'),
            (ed[0][2][1][:, 0], 'rgba(255, 0, 0, 0.15)', 'red'),
            (ed[0][2][0][:, 1], None, 'green'),
            (ed[0][2][1][:, 1], 'rgba(0, 255, 0, 0.15)', 'green'),
            (ed[0][2][0][:, 2], None, 'blue'),
            (ed[0][2][1][:, 2], 'rgba(0, 0, 255, 0.15)', 'blue')
        ]            

        for i in range(0, len(shadow_data), 2):
            y1, fill_color, line_color = shadow_data[i]
            y2, _, _ = shadow_data[i + 1]

            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=y1,
                    fill=None,
                    mode='lines',
                    line_color=line_color,
                    line_width=0,
                    showlegend=False
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=y2,
                    fill='tonexty',
                    mode='lines',
                    line_color=line_color,
                    line_width=0,
                    fillcolor=fill_color,
                    showlegend=False
                )
            )
        
        

    if "ensemblemembers" in plotoptions:
        datetime_format = "Launch Time: %Y-%m-%d %H:%M"
        t_launch = datetime.datetime.strptime(launchlabel, datetime_format)
        
        for row in selectedrows:
            iparams = get_iparams(row)
            rowind = row['Index']
            model_obj = coreweb.ToroidalModel(t_launch, **iparams) # model gets initialized
            model_obj.generator()
            
            if sc == "SYN":
                raise PreventUpdate
                
            else:
                outa = np.array(model_obj.simulator(graph['t_data'], graph['pos_data']), dtype=object)
                
                
            outa = np.squeeze(outa[0])
            
            if refframe == "RTN":
                x,y,z = hc.separate_components(graph['pos_data'])
                rtn_bx, rtn_by, rtn_bz = hc.convert_HEEQ_to_RTN_mag(x,y,z, outa[:, 0],outa[:, 1],outa[:, 2])
                outa[:, 0],outa[:, 1],outa[:, 2] = rtn_bx, rtn_by, rtn_bz
            
            outa[outa==0] = np.nan
            
            names = graph['names']
                            
            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=outa[:, 0],
                    line=dict(color='red', width=3, dash='dot'),
                    name=names[0]+'_'+str(rowind),
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=outa[:, 1],
                    line=dict(color='green', width=3, dash='dot'),
                    name=names[1]+'_'+str(rowind),
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=outa[:, 2],
                    line=dict(color='blue', width=3, dash='dot'),
                    name=names[2]+'_'+str(rowind),
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=np.sqrt(np.sum(outa**2, axis=1)),
                    line=dict(color='black', width=3, dash='dot'),
                    name='Btot_'+str(rowind),
                )
            )
                            
                                      
        
        
    if "fittingpoints" in plotoptions:
        
        try:
            t_s, t_e, t_fit = extract_t(tabledata[0])
        
            fig.add_vrect(
                x0=t_s,
                x1=t_s,
                line=dict(color="Red", width=.5),
                name="Ref_A",  # Add label "Ref_A" for t_s
            )

            fig.add_vrect(
            x0=t_e,
            x1=t_e,
            line=dict(color="Red", width=.5),
            name="Ref_B",  # Add label "Ref_B" for t_e
            )

            for idx, line in enumerate(t_fit):
                fig.add_vrect(
                    x0=line,
                    x1=line,
                    line=dict(color="Black", width=.5),
                    name=f"t_{idx + 1}",  # Add labels for each line in t_fit
                )
        except:
            print('No fitting points to plot')
            pass
        
    if "title" in plotoptions:
        fig.update_layout(title=infodata['sc'][0])
    
    if "catalogevent" in plotoptions:
        sc = infodata['sc'][0]
        begin = infodata['begin'][0]
        end = infodata['end'][0]
        
        if infodata['id'][0] == 'I':
            opac = 0
        else:
            opac = 0.3

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
                
        fig.add_vrect(
                x0=begin,
                x1=end,
                fillcolor="LightSalmon", 
                opacity=opac,
                layer="below",
                line_width=0
        )
    return fig


@app.long_callback(
    Output("graphstore", "data"),
    Output("posstore", "data"),
    Output("loadgraphstoreicon", "children"),
    Input("event-info","data"),
    Input("reference_frame","value"),
    State("posstore", "data"),
    running = [
        (Output("loadgraphstoreicon", "children"),
            dbc.Spinner(), " "
        ), 
    ],
    
)
def generate_graphstore(infodata, reference_frame, posstore):
    newhash = infodata['id']
    
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    if (newhash == "No event selected") or (newhash == None):
        return {}, {}, " "
    

    if reference_frame == "HEEQ":
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
    data_file_path = os.path.join(os.path.dirname(__file__), "data", f"{newhash[0]}.pkl")

    
    if os.path.exists(data_file_path) and not (sc == "NOAA_RTSW" or sc == "STEREO-A_beacon" or infodata['loaded'] is not False):
        # Load data from the file

        with open(data_file_path, 'rb') as file:
            saved_data = pickle.load(file)
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
            b_data_HEEQ, b_data_RTN, t_data, pos_data = get_uploaddata(infodata['loaded'])
        else:
            try:
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

                    elif (sc == "NOAA_RTSW") or (sc == "STEREO-A_beacon"):
                        print('Loading realtime data...')
                        b_data_HEEQ, b_data_RTN, t_data, pos_data = get_rt_data(sc, insitubegin, insituend)
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
                    return {}, {}, fail_icon

        if reference_frame == "HEEQ":
            b_data = b_data_HEEQ
        else:
            b_data = b_data_RTN
            
            
        # Check for archive path
        archivepath = os.path.join(os.path.dirname(__file__), "data", "archive")
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
        except:
            bodydata = None
            print('Failed to load body data')


        scs = ["SOLO", "PSP", "BEPI", "STEREO-A"]

        for scc in scs:

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
                print("Failed to load data for ", scc, ":", e)


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
        if not (sc == "NOAA_RTSW" or sc == "STEREO-A_beacon"):
            
            with open(data_file_path, 'wb') as file:
                pickle.dump(saved_data, file)
        else:
            # Get the current date and time as a string
            current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

            # Append the current time to the file name
            file_name, file_extension = os.path.splitext(data_file_path)
            updated_file_path = f"{file_name}_{current_time}{file_extension}"

            with open(updated_file_path, 'wb') as file:
                pickle.dump(saved_data, file)
    try:
        view_legend_insitu = True
        fig = plot_insitu(names, t_data, b_data, view_legend_insitu) 
    except Exception as e:
        print("An error occurred:", e)
        return {}, {},fail_icon
    
    
    return {'fig': fig,  'b_data_HEEQ': b_data_HEEQ, 'b_data_RTN': b_data_RTN, 't_data': t_data, 'pos_data': pos_data, 'names': names, 'bodytraces': bodytraces, 'bodydata': bodydata}, posstore, success_icon


# Define a callback to handle the cancellation button
@app.callback(
    Output("cancel_button", "disabled"),
    Input("cancel_button", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_fit(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    
    global processes
    for process in processes:
        process.terminate()  # Terminate all processes
    processes = []  # Clear the list of processes
    return True


@app.long_callback(
    output = Output("statusplaceholder", "children"),
    inputs = Input("run_button", "n_clicks"),
    state=[
        State("graphstore", "data"),
        State("launch-label-fit", "children"),
        State("spacecrafttable", "rowData"),
        *[
            State(id, "value") for id in fittingstate
            if id != "launch-label" and id != "spacecrafttable"
        ],
        State("event-info", "data")
    ],
    running = [        
        (Output("run_button", "disabled"), True, False),
        (Output("cancel_button", "disabled"), False, True),
    ],
    cancel=[Input("cancel_button", "n_clicks")],
    progress=Output("statusplaceholder", "children"),
    progress_default=make_progress_graph(0, 512, 0, 0, 0, 0),
    prevent_initial_call=True,
    interval=1000,
)
def main_fit(set_progress, n_clicks, graphstore, *fittingstate_values):
    global manager, processes
    
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
    
    try:
        base_fitter, fit_coord_system, multiprocessing, itermin, itermax, n_particles, outputfile, njobs, model_kwargs, t_launch  = extract_fitvars(fittingstate_values)
    
    except:
        return (make_progress_graph(0, 512, 0, 0, 0, 0))
    
    t_launch = sanitize_dt(t_launch)
    
    
    if multiprocessing == True:
        #global mpool
        mpool = mp.Pool(processes=njobs) # initialize Pool for multiprocessing
        processes.append(mpool)
    #print('FittingDATA')
    data_obj = FittingData(base_fitter.observers, fit_coord_system, graphstore)
    #print('generate noise')
    data_obj.generate_noise("psd",60)
   
    kill_flag = False
    killstatus = None
    pcount = 0
    timer_iter = None
    
    try:
        for iter_i in range(iter_i, itermax):
            # We first check if the minimum number of 
            # iterations is reached.If yes, we check if
            # the target value for epsilon "epsgoal" is reached.
            reached = False
            #print(str(iter_i) + ' iter_i')

            if iter_i >= itermin:
                if hist_eps[-1] < epsgoal:
                    try:
                        return make_progress_graph(pcount, n_particles, hist_eps[-1], hist_eps[-2], iter_i, 2)
                    except:
                        pass
                    kill_flag = True
                    killstatus = 2
                    break    
                    
            print("running iteration " + str(iter_i))        
                    
            
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
                
                #print("initial eps_init = ", eps_init)
                

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
            
            print("starting simulations")

            try:
                set_progress(make_progress_graph(0, n_particles, hist_eps[-1], hist_eps[-2], iter_i-1, 1))
            except:
                set_progress(make_progress_graph(0, n_particles, hist_eps[-1], eps_init, iter_i-1, 1))
                pass

            if multiprocessing == True:
                print("multiprocessing is used")
                _results = mpool.starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(njobs)]) # starmap returns a function for all given arguments
            else:
                print("multiprocessing is not used")
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
                try:
                    set_progress(make_progress_graph(pcount, n_particles, hist_eps[-1], hist_eps[-2], iter_i, 1))
                except:
                    set_progress(make_progress_graph(pcount, n_particles, hist_eps[-1], eps_init, iter_i, 1))
                    pass

                for i in range(0, len(_results)):
                    particles_temp[
                        sum(pcounts[:i]) : sum(pcounts[: i + 1])
                    ] = _results[i][0] # results of current iteration are stored
                    epses_temp[sum(pcounts[:i]) : sum(pcounts[: i + 1])] = _results[
                        i
                    ][1] # errors of current iteration are stored
                    
                    
                #print(
                #    f"step {iter_i}:{sub_iter_i} with ({pcount}/{n_particles}) particles"
                #)

                if pcount > n_particles:
                    print(str(pcount) + ' reached particles')
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
                    print("no hits, aborting")
                    kill_flag = True
                    killstatus = 4
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

            hist_time.append(time.time() - timer_iter)
            iter_i = iter_i + 1  # iter_i gets updated

            # save output to file 
            if outputfile:
                output_file = os.path.join(
                    outputfile, "{0:02d}.pickle".format(iter_i - 1)
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
    
    
    return make_progress_graph(pcount, n_particles, hist_eps[-1], hist_eps[-2], iter_i, killstatus)
