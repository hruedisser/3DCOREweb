import dash
from dash import dcc, html, Input, Output, State, callback, register_page, no_update, ctx, long_callback
import dash_mantine_components as dmc
import plotly.express as px
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from dash_iconify import DashIconify

from coreweb.dashcore.assets.config_sliders import modelslidervars, magslidervars, fittingstate, modelstate
from coreweb.dashcore.utils.utils import load_fit, round_to_hour_or_half

from dash.long_callback import CeleryLongCallbackManager

from celery import Celery

import pickle

import diskcache
import time
import os

import pandas as pd

import datetime
import functools 

from coreweb.dashcore.utils.utils import create_double_slider, get_insitudata, make_progress_graph
from coreweb.dashcore.utils.plotting import plot_insitu

register_page(__name__, icon="mdi:chart-histogram", order=1)

df = pd.DataFrame({
            "spacecraft": [""],
            "ref_a": [""],
            "ref_b": [""],
            "t_1": [""],
            "t_2": [""],
            "t_3": [""],
            "t_4": [""],
            "t_5": [""],
            "t_6": [""],
            "t_7": [""],
            "t_8": [""],
        })

################ COMPONENTS
###########################



fitterradio = html.Div(
                            [
                                dbc.Label("Fitter", style={"font-size": "12px"}),
                                dcc.RadioItems(
                            options=[
                                {"label": " ABC-SMC", "value": "abc-smc"},
                            ],
                            id="fitter-radio",
                            value="abc-smc",
                            persistence=True,
                            inline=True,
                        ),
                            ],
                            className="form-group",
                        )

multiprocesscheck = html.Div(
    [
        dbc.Label("Number of Jobs", style={"font-size": "12px"}),
        html.Br(),
        html.Div(
            [
                dcc.Input(
                    id='n_jobs',
                    type='number',
                    min=1,
                    max=300,
                    step=1,
                    value=4,
                    style={"margin-right": "10px"}
                ),
                dcc.Checklist(
                    id="multiprocesscheck",
                    options=[{"label": " Multiprocessing", "value": "multiprocessing"}],
                    value=["multiprocessing"],
                    persistence=True,
                )
            ],
            style={"display": "inline-flex", "align-items": "center"}
        )
    ],
    className="form-group",
)



numiter = create_double_slider(1,15, [8,12], 1, 'Number of Iterations', 'n_iter', 'n_iter', {i: str(i) for i in range(1, 16, 1)})

particlenum = html.Div(
                            [
                                dbc.Label("Number of Particles",style={"font-size": "12px"}),
                                dcc.Slider(
                                    id="particle-slider",
                                    min=0,
                                    max=3,
                                    marks={
                                        0: "265",
                                        1: "512",
                                        2: "1024",
                                        3: "2048"
                                    },
                                    value=1,
                                    step=None,
                                    included=False,
                                    updatemode="drag",
                                    persistence=True,
                                ),
                            ],
                            className="form-group",
                        )

ensemblenum = html.Div(
                            [
                                dbc.Label("Ensemble Size",style={"font-size": "12px"}),
                                dcc.Slider(
                                    id="ensemble-slider",
                                    min=16,
                                    max=18,
                                    marks={
                                        16: "2^16",
                                        17: "2^17",
                                        18: "2^18"
                                    },
                                    value=16,
                                    step=None,
                                    included=False,
                                    updatemode="drag",
                                    persistence=True,
                                ),
                            ],
                            className="form-group",
                        )

timergroup = dbc.Row([
            dbc.Col([
                dbc.ButtonGroup(
                [
                    dbc.Button("Run", id = 'run_button', color="primary"),
                    dbc.Button("Cancel", id = 'cancel_button', color="secondary", disabled=True),
                ],
                className="mr-2",
            ),
            ],
                width=6,
            ),
                dbc.Col(
                [
                    html.Div(id='elapsed-time'),   
                ],
                width=6,  # Adjust the width of the right-aligned columns
                style={"display": "flex", "justify-content": "flex-end"},
            ),
        ], justify="between",  # Distribute the columns
           align="end",  # Align content at the bottom of the row
        )


tabform = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                dbc.Form(
                    [
                        fitterradio,
                        html.Br(),
                        multiprocesscheck,
                        html.Br(),
                        numiter,
                        particlenum,
                        html.Br(),
                        ensemblenum,
                        html.Br(),
                    ]
                ),
                style={"max-height": "400px", "overflow-y": "auto"},
            ),
            timergroup,
            
            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every 1 second
                n_intervals=0
            ),
            
        ]
    ),
    className="mt-3",
)


modelsliders_double = html.Div(
    [create_double_slider(
        var['min'],
        var['max'],
        var['doubl_def'],
        var['step'],
        var['var_name'],
        var['variablename_double'],
        var['variablename_double']+ 'label',
        var['marks'],
        var['unit']
    ) for var in modelslidervars],
    style= {"margin": "20px",
                    "maxWidth": "310px",
                    "overflowX": "auto",
                    "whiteSpace": "nowrap",
                },
)


magsliders_double = html.Div(
    [create_double_slider(
        var['min'],
        var['max'],
        var['doubl_def'],
        var['step'],
        var['var_name'],
        var['variablename_double'],
        var['variablename_double']+ 'label',
        var['marks'],
        var['unit']
    ) for var in magslidervars],
    style= {"margin": "20px",
                    "maxWidth": "310px",
                    "overflowX": "auto",
                    "whiteSpace": "nowrap",
                },
)

launchdatepicker = html.Div(
    [
    html.Label(id="launch-label-fit", 
               children="Launch Time:", 
               style={"font-size": "12px", 
                     }),
    html.Br(),
    dcc.Slider(
        id="launch_slider_fit",
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

@callback(
    Output("launch-label-fit", "children"),
    Input("event-info", "data"),
    Input("launch_slider_fit", "value"),
)
def update_launch_label(data, slider_value):
    if data == None:
        return "Launch Time:", 
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
            return f"Launch Time: {launch_formatted}"
        except:
            return "Launch Time:"

tabparam = dbc.Card(
    dbc.CardBody(
        [
           launchdatepicker,
           dmc.Divider(
                    label="Model Parameters",
                ),
                modelsliders_double,
                dmc.Divider(
                    label="Magnetic Field Parameters",
                ),
                magsliders_double,
        ]
    ),
    className="mt-3",style={"max-height": "400px", "overflow-y": "auto"},
)

tabload = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                dbc.Form(
                    [
                        dcc.Dropdown(id="fit_dropdown", options=[], placeholder="Select a fit", value=None,persistence=True, persistence_type='session',)
                    ]
                ),
            ),
            html.Br(),
            dbc.Button("Load Results", id = 'loadfit_button', color="primary", disabled=False),
        ]
    ),
    className="mt-3",
)


# Create the Accordion component
accordion = dbc.Accordion(
    [
        dbc.AccordionItem(tabparam, title="Parameters"),
        dbc.AccordionItem(tabform, title="Fitting"),
        dbc.AccordionItem(tabload, title="Load"),
    ],
        start_collapsed=True,

)

statusplaceholder = html.Div(
            [make_progress_graph(0, 512, 0, 0, 0, 0)
            ],id="statusplaceholder",
)

# Create the Spacecraft Table

columnDefs = [
    {
        "headerName": "Spacecraft",
        "field": "spacecraft", 
        "checkboxSelection": True,
        "cellEditor": "agSelectCellEditor",
        "cellEditorParams": {
            'values': ["BepiColombo",
                       "DSCOVR",
                       "PSP",
                       "SolarOrbiter",
                       "STEREO A",
                       "STEREO B",
                       "Wind"]}
        },
    {
        "headerName": "Reference A",
        "field": "ref_a",
    },
    {
        "headerName": "Reference B",
        "field": "ref_b",
    },
    {
        "headerName": "t_1",
        "field": "t_1",
    },
    {
        "headerName": "t_2",
        "field": "t_2",
    },
    {
        "headerName": "t_3",
        "field": "t_3",
    },
    {
        "headerName": "t_4",
        "field": "t_4",
    },
    {
        "headerName": "t_5",
        "field": "t_5",
    },
    {
        "headerName": "t_6",
        "field": "t_6",
    },
    {
        "headerName": "t_7",
        "field": "t_7",
    },
    {
        "headerName": "t_8",
        "field": "t_8",
    },
]

defaultColDef = {
    "resizable": True,
    "editable": True,
    "minWidth": 180,
    "flex":1,
}

rowData = df.to_dict("records")

groupies = dbc.Row([
            dbc.Col(
                dbc.ButtonGroup(
                    [
                        dbc.Button(
                            id="add-sc",
                            children="Add Spacecraft",
                            color="primary",
                            disabled=True,
                        ),
                        dbc.Button(
                            id="delete-sc",
                            children="Delete Spacecraft",
                            color="secondary",
                            disabled=True,
                        ),],
                    className="mr-2",
                ),
                width=10,
            ),
                dbc.Col(
                [
                    dbc.ButtonGroup(
                        [
                            dbc.Button("Auto", "ld-evt", color="primary"),
                            dbc.Button("Load", id = "ld-rowdata", color="primary"),
                            dbc.Button("Save", id = "sv-rowdata", color="secondary"),
                        ],
                        className="mr-2",
                    )
                ],
                width=2,  # Adjust the width of the right-aligned columns
                style={"display": "flex", "justify-content": "flex-end"},
            ),
        ], justify="between",  # Distribute the columns
           align="end",  # Align content at the bottom of the row
        )




spacecrafttable = dag.AgGrid(
    id="spacecrafttable",
    className="ag-theme-alpine",
    columnDefs=columnDefs,
    rowData=rowData,
    defaultColDef=defaultColDef,
    dashGridOptions={"undoRedoCellEditing": True},
    style={"max-height": "200px", "overflow-y": "auto"},
    persistence=True,
    persistence_type='session',
)

table = dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        spacecrafttable,
                                        html.Hr(),
                                        groupies,
                                    ], 
                                ),
                            ],
                        )

# Create the fittingform layout
fittingform = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(accordion, width=6),  # Set the accordion width to half of the screen (6 out of 12 columns)
                dbc.Col(statusplaceholder,width=6),  # Set the Markdown width to half of the screen (6 out of 12 columns)
            ]
        )
    ]
)
    

insitufitgraphcontainer = html.Div(
    [
        dmc.CheckboxGroup(
            id="plotoptions",
            label="Options for plotting",
            #description="This is anonymous",
            orientation="horizontal",
            withAsterisk=False,
            offset="md",
            mb=10,
            children=[
                dmc.Checkbox(label="Fitting Points", value="fittingpoints", color="green"),
                dmc.Checkbox(label="Catalog Event", value="catalogevent", color="green"),
                dmc.Checkbox(label="Title", value="title", color="green"),
                dmc.Checkbox(label="Fitting Results", value="fittingresults", 
                             color="green", disabled=True),
                dmc.Checkbox(label="Ensemble Members", value="ensemblemembers", color="green", disabled=True),
            ],
            value=[ "catalogevent"],
        ),
        dbc.Spinner(id="insitufitspinner"),
        dcc.Graph(id="insitufitgraph"),
    ],
    id = "insitufitgraphcontainer",
)



# Create the Spacecraft Table

resultstatcolumnDefs = [
    {
        "headerName": "Statistic",
        "field": "Index",
    },
    {
        "headerName": 'RMSE Ɛ',
        "field": 'RMSE Ɛ',
    },
    {
        "headerName": 'Longitude',
        "field": 'Longitude',
    },
    {
        "headerName": 'Latitude',
        "field": 'Latitude',
    },
    {
        "headerName": 'Inclination',
        "field": 'Inclination',
    },
    {
        "headerName": 'Diameter 1 AU',
        "field": 'Diameter 1 AU',
    },
    {
        "headerName": 'Aspect Ratio',
        "field": 'Aspect Ratio',
    },
    {
        "headerName": 'Launch Radius',
        "field": 'Launch Radius',
    },
    {
        "headerName": 'Launch Velocity',
        "field": 'Launch Velocity',
    },
    {
        "headerName": 'T_Factor',
        "field": 'T_Factor',
    },
    {
        "headerName": 'Expansion Rate',
        "field": 'Expansion Rate',
    },
    {
        "headerName": 'Magnetic Decay Rate',
        "field": 'Magnetic Decay Rate',
    },
    {
        "headerName": 'Magnetic Field Strength 1 AU',
        "field": 'Magnetic Field Strength 1 AU',
    },
    {
        "headerName": 'Background Drag',
        "field": 'Background Drag',
    },
    {
        "headerName": 'Background Velocity',
        "field": 'Background Velocity',
    },
]


resulttabcolumnDefs = [
    {
        "headerName": "Index",
        "field": "Index",
        "checkboxSelection": True,
        "rowSelection": "multiple",  # Enable multiple row selection
        "sortable": True,  # The index column should not be sortable
        "minWidth": 100,  # Adjust the width of the index column as needed
        "rowMultiSelectWithClick": True,
        "resizable": False,  # Disable resizing of the index column
        "suppressSizeToFit": True  # Avoid the index column from participating in sizeToFit calculations
    },
    {
        "headerName": 'RMSE Ɛ',
        "field": 'RMSE Ɛ',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Longitude',
        "field": 'Longitude',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Latitude',
        "field": 'Latitude',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Inclination',
        "field": 'Inclination',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Diameter 1 AU',
        "field": 'Diameter 1 AU',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Aspect Ratio',
        "field": 'Aspect Ratio',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Launch Radius',
        "field": 'Launch Radius',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Launch Velocity',
        "field": 'Launch Velocity',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'T_Factor',
        "field": 'T_Factor',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Expansion Rate',
        "field": 'Expansion Rate',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Magnetic Decay Rate',
        "field": 'Magnetic Decay Rate',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Magnetic Field Strength 1 AU',
        "field": 'Magnetic Field Strength 1 AU',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Background Drag',
        "field": 'Background Drag',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Background Velocity',
        "field": 'Background Velocity',
        "sortable": True,  # Enable sorting for this column
    },
    {
        "headerName": 'Launch Time',
        "field": 'Launch Time',
        "sortable": True,  # Enable sorting for this column
    },
]

defaultresColDef = {
    "resizable": True,
    "editable": False,
    "flex":1,
    "sortable": True,
    "minWidth": 120,
}

resdf = pd.DataFrame(columns = ['Index','RMSE Ɛ','Longitude', 'Latitude', 'Inclination', 'Diameter 1 AU', 'Aspect Ratio', 'Launch Radius', 'Launch Velocity', 'T_Factor', 'Expansion Rate', 'Magnetic Decay Rate', 'Magnetic Field Strength 1 AU', 'Background Drag', 'Background Velocity', 'Launch Time'] )

rowresData = resdf.to_dict("records")

defaultstatColDef = {
    "resizable": True,
    "editable": False,
    "flex":1,
    "sortable": True,
    "minWidth": 150,
}

download_icon = dmc.ThemeIcon(
                     DashIconify(icon='bxs:download', width=20, color="white"),
                     radius=0,
                     style={"backgroundColor": "transparent"},
                 )



restable = dag.AgGrid(
    id="restable",
    className="ag-theme-alpine",
    columnDefs=resulttabcolumnDefs,
    rowData=rowresData,
    defaultColDef=defaultresColDef,
    csvExportParams={
                "fileName": "ensemble_members.csv",
            },
    style={"max-height": "250px", "overflow-y": "auto"},
    persistence=True,
    persistence_type='session',
    dashGridOptions={"rowSelection":"multiple"},
)

statstable = dag.AgGrid(
    id="statstable",
    className="ag-theme-alpine",
    columnDefs=resultstatcolumnDefs,
    rowData=rowresData,
    defaultColDef=defaultstatColDef,
    csvExportParams={
                "fileName": "statistics.csv",
            },
    style={"max-height": "250px", "overflow-y": "auto"},
    persistence=True,
    persistence_type='session',
)

restabtable = html.Div(
    [
        
     html.Hr(),
     html.Br(),
        html.H5("Results", className="display-10"),
        html.Br(),
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.H6("Ensemble Members", className="display-10"),
                        restable,
                        html.Div(  # Create a new container to hold the button and apply the style
                            [ 
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button(
                                            id="ld-synresults",
                                            children="Adjust Sliders",
                                            color="primary",
                                            size="md",
                                            className="mt-3",
                                            disabled=True,
                                        ),
                                        dbc.Button(
                                            id="plt-synresults",
                                            children="Plot Synthetic Insitu",
                                            color="primary",
                                            size="md",
                                            className="mt-3",
                                            disabled=True,
                                        ),
                                        dbc.Button(
                                            id="download-csv",
                                            children=download_icon,
                                            color="primary",
                                            size="md",
                                            className="mt-3",
                                        )
                                    ],
                                    className="mr-2",
                                )
                            ],
                            style={"text-align": "right"},  # Align the button to the right
                        ),
                        html.H6("Statistics", className="display-10"),
                        statstable,
                        dcc.Graph(id="statsgraph"),
                    ]
                ),
            ]
        ),
    ],
    id="restabtable-card",
    style={"display": "none"},
)

#################### LAYOUT
###########################

# Define the app layout
layout = dbc.Container(
    [html.Br(),
     html.H2("Numerical Fitting", 
             className="display-10"),
     html.Br(),
     html.Hr(),
     html.Br(),
     fittingform,
     html.Br(),
     html.Hr(),
     html.Br(),
     html.H5("Observers", 
             className="display-10"),
     html.Br(),
     table,
     html.Br(),
     html.Hr(),
     html.Br(),
     html.H5("Insitu Data", 
             className="display-10"),
     html.Br(),
     insitufitgraphcontainer,
     restabtable,
     ]
)


################# FUNCTIONS
###########################




################# CALLBACKS
###########################

@callback(
    Output('elapsed-time', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('run_button', 'n_clicks_timestamp'),
    State('run_button', 'disabled'),
)
def update_elapsed_time(n, button_timestamp, run_button_disabled):
    if (button_timestamp is None) or (run_button_disabled == False):
        return ''

    if n > 0:
        current_timestamp = int(time.time() * 1000)  # Current time in milliseconds
        elapsed_time_ms = current_timestamp - button_timestamp
        elapsed_time_sec = elapsed_time_ms // 1000  # Convert milliseconds to seconds
        return f'{elapsed_time_sec} s'

    return ''  # Display nothing initially


@callback(
    Output("restable", "exportDataAsCsv"),
    Output("statstable", "exportDataAsCsv"),
    Input("download-csv", "n_clicks"),
)
def export_data_as_csv(n_clicks):
    if n_clicks:
        return True, True
    return False, False


@callback(
    Output('longit_double', 'min'),
    Output('longit_double', 'max'),
    [Input('toggle-range', 'value')]
)
def update_slider_range(toggle_value):
    if toggle_value == 0:
        return -180, 180
    else:
        return 0, 360
    
##slider callbacks

def create_doublecallback(var, index):
    html_for = var['variablename_double'] + 'label'
    ids = var['variablename_double']
    unit = var['unit']
    func_name = f"update_double_slider_label{index}"
    
    def callback_func(values, label=f"{var['var_name']}"):
        return f"{label}: {values[0]}, {values[1]} {unit}"
    
    callback_func.__name__ = func_name
    callback(Output(html_for, "children"), [Input(ids, "value")])(callback_func)
    
callbacks = []

for i, var in enumerate(modelslidervars):
    create_doublecallback(var, i)
    
for i, var in enumerate(magslidervars):
    create_doublecallback(var, i + len(modelslidervars))    
    

@callback(
    *[
            Output(id, "value") for id in modelstate
        ],
    Output("launch_slider", "value"),
    Input("ld-synresults", "n_clicks"),
    State("restable", "selectedRows"),
    State("launch_slider_fit","value"),
    State("event-info", "data"),
)
def update_buttons(nlicks, selected_rows, sliderval, infodata):
    if (nlicks == None) or (nlicks) == 0:
            raise PreventUpdate
    
    row = selected_rows[0]
    values_to_return = []
    
    date1 = round_to_hour_or_half(datetime.datetime.fromisoformat(infodata['begin'][0]))
    date2 = datetime.datetime.strptime(row.get('Launch Time'), '%Y-%m-%d %H:%M')
    
    new_sliderval = (date2.replace(tzinfo=None) - date1.replace(tzinfo=None)).total_seconds() / 3600
    
    for id in modelstate:
        key = {
            'longit': 'Longitude',
            'latitu': 'Latitude',
            'inc': 'Inclination',
            'dia': 'Diameter 1 AU',
            'asp': 'Aspect Ratio',
            'l_rad': 'Launch Radius',
            'l_vel': 'Launch Velocity',
            'exp_rat': 'Expansion Rate',
            'b_drag': 'Background Drag',
            'bg_vel': 'Background Velocity',
            't_fac': 'T_Factor',
            'mag_dec': 'Magnetic Decay Rate',
            'mag_strength': 'Magnetic Field Strength 1 AU',
        }.get(id)
        
        if key is not None:
            value = row.get(key, no_update)
            values_to_return.append(value)
        else:
            values_to_return.append(no_update)

    return values_to_return + [new_sliderval]



@callback(
    Output("plt-synresults", "disabled"),
    Output("ld-synresults", "disabled"),
    Input("restable", "selectedRows"),
)
def update_buttons(selected_rows):
    plot_button_disabled = True
    another_button_disabled = True

    if selected_rows:
        plot_button_disabled = False
        if len(selected_rows) == 1:
            another_button_disabled = False

    return plot_button_disabled, another_button_disabled


@callback(
    Output("fit_dropdown", "options"),
    Input("event-info", "data"),
    Input("reference_frame", "value"),
)
def update_fit_dropdown(data, refframe):
    ids = data['id'][0] + '_' #+ refframe[0]
    
    outputpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output/"))
    options = [f for f in os.listdir(outputpath) if os.path.isdir(os.path.join(outputpath, f)) and f.startswith(ids)],
    return options[0]

# add or delete rows of table
@callback(
    output = [
        Output("spacecrafttable", "deleteSelectedRows"),
        Output("spacecrafttable", "rowData"),
        *[
            Output(id, "value") for id in fittingstate
            if id != "launch-label" and id != "spacecrafttable" and id != "reference_frame"
        ],
        Output("restabtable-card", "style"),
        Output("restable", "rowData"),
        Output("launch_slider_fit", "value"),
        Output("statstable", "rowData"),
        Output("statsgraph", "figure"),
        Output("plotoptions", "children"),
        Output("restable", "csvExportParams"),
        Output("statstable", "csvExportParams"),
    ],
    inputs = [
        Input("ld-rowdata","n_clicks"),
        Input("ld-evt","n_clicks"),
        Input("loadfit_button", "n_clicks"),
    Input("delete-sc", "n_clicks"),
    Input("add-sc", "n_clicks"),
    Input("sv-rowdata", "n_clicks"),
    ],
    state = [
        State("event-info", "data"),
        State("spacecrafttable", "rowData"),
        State("fit_dropdown", "value"),
        State("graphstore", "data"),
    ],
    prevent_initial_call = True
)
def update_table_or_load(n_ldtable, n_ldevt, n_load, n_dlt, n_add, n_sv_rowdata,  infodata, data, name, graph):

    
    spacetable_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "spacetable.csv"))

    triggered_id = ctx.triggered_id
    
    
    if triggered_id == "loadfit_button":
        
        if name == None:
            raise PreventUpdate
            
        if (n_load == None) or (n_load) == 0:
            raise PreventUpdate
         
        tablenew, *fitting_values, resdfdic, t0, mean_row, statfig = load_fit(name, graph)        

        dtval = infodata['begin'][0]
        dateFormat = "%Y-%m-%dT%H:%M:%S%z"
        dateFormat2 = "%Y-%m-%d %H:%M:%S"
        
        try:
            dtval_in = datetime.datetime.strptime(dtval, dateFormat2)
        except ValueError:
            dtval_in = datetime.datetime.strptime(dtval, dateFormat)
        
        ltval = (t0-dtval_in).total_seconds() // 3600
        
        plotchildren = [
            dmc.Checkbox(label="Fitting Points", value="fittingpoints", color="green"),
            dmc.Checkbox(label="Catalog Event", value="catalogevent", color="green"),
            dmc.Checkbox(label="Title", value="title", color="green"),
            dmc.Checkbox(label="Fitting Results", value="fittingresults", color="green"),
            dmc.Checkbox(label="Ensemble Members", value="ensemblemembers", color="green"),
        ]
        
        
        csvExportParams={
                "fileName": name + "statistics.csv",
            },
        
        
        return False, tablenew.to_dict("records"), *fitting_values, {"display": "block"}, resdfdic, ltval, mean_row, statfig, plotchildren, {"fileName": name + "_ensemblemembers.csv"}, {"fileName": name + "_statistics.csv"}
    
    
    
    if triggered_id == "add-sc":
        
        if (n_add == None) or (n_add) == 0:
            raise PreventUpdate
            
        new_row = {
            "spacecraft": [""],
            "ref_a": [""],
            "ref_b": [""],
            "t_1": [""],
            "t_2": [""],
            "t_3": [""],
            "t_4": [""],
            "t_5": [""],
            "t_6": [""],
            "t_7": [""],
            "t_8": [""],
        }
        df_new_row = pd.DataFrame(new_row)
        updated_table = pd.concat([pd.DataFrame(data), df_new_row])
        return False, updated_table.to_dict("records"), *[no_update] * 19, {"display": "none"}, rowresData, no_update,rowresData, no_update, no_update, no_update, no_update

    elif triggered_id == "delete-sc":
        
        if (n_dlt == None) or (n_dlt) == 0:
            raise PreventUpdate
            
        return True, *[no_update] * 20, {"display": "none"}, rowresData, no_update,rowresData, no_update, no_update, no_update, no_update
    
    elif triggered_id == "ld-evt":
        
        if (n_ldevt == None) or (n_ldevt) == 0:
            raise PreventUpdate
            
        if infodata["sc"] == "":
            raise PreventUpdate
        
        try:
            begin = datetime.datetime.strptime(infodata["begin"][0], "%Y-%m-%d %H:%M:%S")
        except:
            try:
                begin = datetime.datetime.strptime(infodata["begin"][0], "%Y-%m-%dT%H:%M:%S%z")
            except:
                 begin = datetime.datetime.strptime(infodata["begin"][0], "%Y-%m-%dT%H:%M:%S")
                    
        try:
            end = datetime.datetime.strptime(infodata["end"][0], "%Y-%m-%dT%H:%M:%S")
        except:
            try:
                end = datetime.datetime.strptime(infodata["end"][0], "%Y-%m-%dT%H:%M:%S%z")
            except:
                end = datetime.datetime.strptime(infodata["end"][0], "%Y-%m-%d %H:%M:%S")
                
        refa = begin #- datetime.timedelta(hours = 6)
        
        if begin == end:
            refb = refa + datetime.timedelta(hours=20)
        else:
            refb = end #+ datetime.timedelta(hours = 6)
            
        time_difference = (refb - refa)

        # Calculate the interval between each of the four times
        interval = time_difference / 5
        
        # Calculate and round t_1, t_2, t_3, and t_4
        t_1 = refa + interval
        t_2 = t_1 + interval
        t_3 = t_2 + interval
        t_4 = t_3 + interval

        # Round the times to the nearest 30-minute precision
        t_1 = t_1 - datetime.timedelta(minutes=t_1.minute % 30, seconds=t_1.second, microseconds=t_1.microsecond)
        t_2 = t_2 - datetime.timedelta(minutes=t_2.minute % 30, seconds=t_2.second, microseconds=t_2.microsecond)
        t_3 = t_3 - datetime.timedelta(minutes=t_3.minute % 30, seconds=t_3.second, microseconds=t_3.microsecond)
        t_4 = t_4 - datetime.timedelta(minutes=t_4.minute % 30, seconds=t_4.second, microseconds=t_4.microsecond)
        
        #t_1 = begin + datetime.timedelta(hours = 2)
        #t_2 = begin + datetime.timedelta(hours = 4)
        row = {
            "spacecraft": infodata["sc"],
            "ref_a": refa.strftime("%Y-%m-%d %H:%M"),
            "ref_b": refb.strftime("%Y-%m-%d %H:%M"),
            "t_1": t_1.strftime("%Y-%m-%d %H:%M"),
            "t_2": t_2.strftime("%Y-%m-%d %H:%M"),
            "t_3": t_3.strftime("%Y-%m-%d %H:%M"),
            "t_4": t_4.strftime("%Y-%m-%d %H:%M"),
            "t_5": [""],
            "t_6": [""],
            "t_7": [""],
            "t_8": [""],
        
        }
        df_row = pd.DataFrame(row)
        return False, df_row.to_dict("records"), *[no_update] * 19, {"display": "none"},rowresData, no_update,rowresData, no_update, no_update, no_update, no_update  
    
    
    
    if triggered_id == "sv-rowdata":
        if (n_sv_rowdata is None) or (n_sv_rowdata == 0):
            raise PreventUpdate

        # Save the edited table to a CSV file
        df = pd.DataFrame(data)
        df.to_csv(spacetable_path, index=False)
        ('observer settings saved to ' + spacetable_path)

        return no_update
    
    if triggered_id == "ld-rowdata":
        if (n_ldtable is None) or (n_ldtable == 0):
            raise PreventUpdate

        # Check if the file exists
        if os.path.isfile(spacetable_path):
            # Load the table from the CSV file
            df = pd.read_csv(spacetable_path)
            tablenew = df.to_dict("records")
            print('observer settings loaded from ' + spacetable_path)

            return False, tablenew, *[no_update] * 25, no_update, no_update
        