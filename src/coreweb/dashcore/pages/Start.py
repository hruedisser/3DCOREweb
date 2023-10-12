import dash
from dash import dcc, html, Input, Output, State, callback, register_page, no_update, ctx
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

import datetime
import base64
from coreweb.dashcore.utils.utils import get_catevents, load_cat_id
from coreweb.dashcore.assets.config_sliders import modelslidervars, magslidervars, fittingstate, modelstate

from dash.exceptions import PreventUpdate
from coreweb.dashcore.utils.utils import *

import cdflib

from PIL import Image

import json

from PIL import Image
import os


image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "3dcore.png"))

pil_image = Image.open(image_path)



register_page(__name__, path="/", icon="fa-solid:home", order=0)


################ COMPONENTS
###########################
image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "files.png"))
pil_image = Image.open(image_path)

#catalogform

filtergroup = dbc.Row([
            dbc.Col([
            ],
                width=3,
            ),
                dbc.Col(
                [
                    dbc.Stack(
                        [
                            dcc.Dropdown(
                                id="year_dropdown",
                                options=[
                                    {"label": str(year), "value": year}
                                    for year in range(datetime.date.today().year, 2009, -1)
                                ],
                                persistence=True,
                                placeholder="Year",
                                persistence_type="session",
                                style={"margin-right": "10px", "min-width": "100px"}  # Add margin and set minimum width

                            ),
                            dcc.Dropdown(
                                id="month_dropdown",
                                options=[
                                    {"label": month, "value": str(month_num).zfill(2)}
                                    for month_num, month in enumerate(
                                        [
                                            "January",
                                            "February",
                                            "March",
                                            "April",
                                            "May",
                                            "June",
                                            "July",
                                            "August",
                                            "September",
                                            "October",
                                            "November",
                                            "December",
                                        ],
                                        start=1,
                                    )
                                ],
                                persistence=True,
                                placeholder="Month",
                                persistence_type="session",
                                style={"margin-right": "10px", "min-width": "100px"}
                            ),
                            dcc.Dropdown(
                                id="day-dropdown",
                                options=[
                                    {"label": str(day).zfill(2), "value": str(day).zfill(2)}
                                    for day in range(1, 32)
                                ],
                                persistence=True,
                                placeholder="Day",
                                persistence_type="session",
                                style={"min-width": "100px"}
                            ),
                        ], direction="horizontal",
                    )
                ],
                width=9,  # Adjust the width of the right-aligned columns
                style={"display": "flex", "justify-content": "flex-end"},
            ),
        ], justify="between",  # Distribute the columns
           align="end",  # Align content at the bottom of the row
        )



newdatepicker = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Label("Filter by"),
                    width={"size": 6, "order": "first"},  # Left-aligned label
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="sc-dropdown-cat",
                        options=[
                            {"label": spacecraft, "value": spacecraft}
                            for spacecraft in [
                                "BepiColombo",
                                "DSCOVR",
                                "PSP",
                                "SolarOrbiter",
                                "STEREO A",
                                "Wind",
                            ]
                        ],
                        persistence=True,
                        placeholder="Spacecraft",
                        persistence_type="session",
                    ),
                    width={"size": 6, "order": "last"},  # Right-aligned date picker
                    style={"justify-content": "flex-end"},  # Right-align the content
                ),
            ],
            className="mb-2",
        ),
        filtergroup,
    ]
)




customdatepicker = html.Div(
    dbc.Row(
        [
            dbc.Col(
                dbc.Label("Select a day to process", html_for="date_picker"),
                width={"size": 6, "order": "first"},  # Left-aligned label
            ),
            dbc.Col(
                dcc.DatePickerSingle(
                    id="custom_date_picker",
                    min_date_allowed=datetime.date(2010, 1, 1),
                    max_date_allowed=datetime.date.today(),
                    initial_visible_month=datetime.date(2022, 6, 22),
                    date=datetime.date(2022, 6, 22),
                    display_format='YYYY/MM/DD',
                    persistence=True,
                    persistence_type='session',
                ),
                width={"size": 6, "order": "last"},  # Right-aligned date picker
                style={"display": "flex", "justify-content": "flex-end"},  # Right-align the content
            ),
        ],
        className="mb-3",
    )
)

# Generate time options for full and half-hour intervals
time_options = [
    {"label": f"{hour:02}:{minute:02}", "value": f"{hour:02}:{minute:02}"}
    for hour in range(0, 24)
    for minute in [0, 30]
]


collapse = html.Div(
    [
        dbc.Button(
            "Modify Start and End (Optional):",
            id="collapse-button",
            className="d-grid gap-2 col-12 mx-auto",  # Use col-12 to make it span the whole width
            size="sm",
            color="primary",
            n_clicks=0,
            style={
                "marginTop": 20,
                "textAlign": "center",  # Center-align the text
            },
        ),
        dbc.Collapse(
            dbc.Card(dbc.CardBody([
                dbc.Row(
            [
                html.Div(id='start-times'),
            ],
            className="mb-3",
        ),
                
                dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id='start-time',
                        options=time_options,
                        placeholder='Select start time',
                        style={"fontSize": 12},  # Adjust the font size here
                    ),
                    width={"size": 6, "order": "first"},  # Left-aligned label
                ),
                dbc.Col(
                    dcc.Input(
                        id='duration',
                        type='number',
                        size = "sm",
                        placeholder='Enter duration (hours)',
                        style={
                            "borderColor": "lightgray",
                            "backgroundColor": "white",  # Set background color to white
                            "borderRadius": "0.25rem",  # Add rounded corners
                            "width": "100%",  # Make the input span the full width
                            "boxSizing": "border-box",  # Include padding in width calculation
                            "fontSize": 12,  # Adjust the font size here

                        },
                    ),

                    width={"size": 6, "order": "last"},  # Right-aligned date picker
                    style={"display": "flex", "justify-content": "flex-end"},  # Right-align the content
                ),
            ],
            className="mb-3",
        ),
                ]
            
            )),
            id="collapse",
            is_open=False,
        ),
    ]
)
            

# Construct the absolute path to the 'output' directory
output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))

#manualform

#forms


catalogform = dbc.Card(
    dbc.CardBody(
        [
            newdatepicker,
            dcc.Dropdown(
                id="event_dropdown",
                options=[],
                placeholder="Select an event",
                value=None,
                persistence=True,
                persistence_type='session',
                style={"margin-top": "10px"}
            ),
            dbc.Button("Submit", id="submitcatalog", color="primary", style={"marginTop": "12px"}),
        ]
    ),
    className="mt-3",
)

fileform = dbc.Card(
    dbc.CardBody(
        [html.Img(
                 src=pil_image,
                 style={
                     'max-width': '50%',
                     'max-height': '50%',
                     'display': 'block',
                     'margin-left': 'auto',
                     'margin-right': 'auto',
                     'margin-top': 'auto',
                     'margin-bottom': 'auto'
                 }
             )]
    ),
    className="mt-3",
)



manualform = dbc.Card(
    dbc.CardBody(
        [
            customdatepicker,
            dcc.Dropdown(
                id='sc-dropdown',
                options=["BepiColombo",
                       "DSCOVR",
                       "PSP",
                       "SolarOrbiter",
                       "STEREO A",
                       "Wind",
                       "NOAA_RTSW (realtime)",
                       "STEREO-A_beacon (realtime)",
                       "Synthetic Spacecraft",
                        ],
                placeholder='Select a spacecraft'
            ),
            collapse,
            #eventdurationpick,
            dbc.Button("Submit", id="submitmanual", color="primary", style={"marginTop": "12px"}),
            
        ]
    ),
    className="mt-3",
)



# Create the Accordion component
accordion = dbc.Accordion(
    [
        dbc.AccordionItem(catalogform, title="Catalog"),
        dbc.AccordionItem(manualform, title="Manual/Realtime", 
                          #style={"pointer-events": "none", "opacity": 0.5}
                         ),
        dbc.AccordionItem(fileform, title="File", 
                          #style={"pointer-events": "none", "opacity": 0.5}
                         ),
    ],
    id = "accordion",
    #persistence=True,
    #persistence_type='session',
)


#################### LAYOUT
###########################

layout = dbc.Container(
    [
     html.Div(id="downloaded-content"),
     html.Br(),
     html.H2("Get started!", 
             className="display-10"),
     html.Br(),
     html.Hr(),
     html.Br(),
     dbc.Row([
         dbc.Col([
             dcc.Markdown(
                 """ 
                 First choose how to initialize the tool: \n
                 **Catalog:** \n
                 >_Choose an event from the [helioforecast catalog](https://helioforecast.space/icmecat)._ \n
                 **Manual/Realtime:** \n
                 >_Choose a spacecraft and day for your event and start from scratch or work on the [latest available data](https://helioforecast.space/static/sync/NOAA_RTSW_STEREO-A_beacon_now.png)._ \n
                 **File:**\n
                 >_Load model from a txt file or upload spacecraft data cdf files._\n
                 """
             ),
         ], width = 6),
         dbc.Col([html.Div(
         accordion)
                 ], width=6)
     ]),  
    ]
)
    
################# FUNCTIONS
########################### 

def save_widget_state(widget_states, filename):
    with open('output/' + filename, 'w') as file:
        json.dump(widget_states, file)

def load_widget_state(filename):
    with open('output/' + filename, 'r') as file:
        widget_states = json.load(file)
    return widget_states

def get_alternative_sc(sc):
    if sc == 'BepiColombo':
        return "BEPI"
    elif sc == 'DSCOVR':
        return "DSCOVR"
    elif sc == 'PSP':
        return "PSP"
    elif sc == 'SolarOrbiter':
        return "SOLO"
    elif sc == 'STEREO A':
        return "STEREO_A"
    elif sc == 'STEREO B':
        return "STEREO_B"
    elif sc == 'Wind':
        return "Wind"
    elif sc == 'Synthetic Spacecraft':
        return "SYN"
    
def create_event_info(processday, 
                      begin, 
                      end, 
                      sc,
                      ids,
                      loaded = False, 
                     ):
    return {"processday": processday,
            "begin": begin,
            "end": end,
            "sc": sc,
            "id": ids,
            "loaded": loaded,
            "changed": True
           }
    

    
################# CALLBACKS
###########################




@callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@callback(
    Output('start-times', 'children'),
    Input('custom_date_picker', 'date'),
    Input('start-time', 'value'),
    Input('duration', 'value')
)
def update_start_end_times(selected_date, start_time, duration):
    if selected_date and start_time and duration:
        # Convert the selected date to a datetime object
        selected_date = datetime.datetime.strptime(selected_date, '%Y-%m-%d')
        
        # Convert the selected start time to a datetime object
        start_time_parts = start_time.split(':')
        start_time = datetime.time(int(start_time_parts[0]), int(start_time_parts[1]))
        
        # Calculate the end time by adding the duration in hours to the start time
        end_time = datetime.datetime.combine(selected_date, start_time) + datetime.timedelta(hours=duration)
        
        # Format the start and end date-time
        start_date_time_formatted = selected_date.strftime('%Y-%m-%d') + ' ' + start_time.strftime('%H:%M')
        end_date_time_formatted = end_time.strftime('%Y-%m-%d %H:%M')
        
        # Return the formatted text with font size 12
        return html.Div(
            f"{start_date_time_formatted} ---> {end_date_time_formatted}",
            style={"fontSize": 12}
        )
    
    return ''

@callback(
    Output("submitmanual", "disabled"),
    Input("sc-dropdown", "value")
)
def update_submit_button_disabled(sc_value):
    if sc_value is not None:
        return False  # Enable the "Submit" button if a spacecraft is selected
    else:
        return True   # Disable the "Submit" button if no spacecraft is selected

@callback(
    Output("submitcatalog", "disabled"),
    Input("event_dropdown", "value")
)
def update_submit_button_disabled(event_value):
    if event_value is not None:
        return False  # Enable the "Submit" button if an event is selected
    else:
        return True   # Disable the "Submit" button if no event is selected
    

@callback(
    Output("rinput", "value"),
    Output("lonput", "value"),
    Output("latput", "value"),
    Output("event-info", "data"),
    Output("launch_slider", "value", allow_duplicate=True),
    Output("reference_frame","value", allow_duplicate=True),
    *[
            Output(id, "value", allow_duplicate=True) for id in modelstate
        ],
    Input('submitcatalog', 'n_clicks'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    Input('submitmanual', 'n_clicks'),
    State("event_dropdown", "value"),
    State("custom_date_picker", "date"),
    State('sc-dropdown',"value"),
    State('start-times', 'children'),
    prevent_initial_call=True
)
def update_alert_for_init(cat_clicks, list_of_contents, list_of_names, list_of_dates, manual_clicks, cat_event,  manual_date, manual_sc, times):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    event_info = None
    rinput = no_update
    lonput = no_update
    latput = no_update
    launchslider = no_update
    refframe = no_update
    modelstatevars = [no_update] * 13
    
    if 'submitcatalog' in changed_id:
        if cat_event == None:
            event = "No event selected"
            event_info = create_event_info([''],
                                           [''],
                                           [''],
                                           '',
                                           event,
                                          )
        
        else:
            event_obj = load_cat_id(cat_event)
            event = f"{cat_event}"
            event_info = create_event_info(event_obj.begin,
                                           event_obj.begin,
                                           event_obj.end,
                                           event_obj.sc,
                                           event_obj.id,
                                          )
            
    elif 'upload-data' in changed_id:
        if list_of_contents is not None:
            if len(list_of_contents) > 1:
                print('multiple files detected')
            if 'txt' in list_of_names[0]:
                content_type, content_string = list_of_contents[0].split(',')
                decoded = base64.b64decode(content_string)
                txt_data = decoded.decode('utf-8')
                # Parse the JSON data from the txt file
                data_dict = json.loads(txt_data)

                event_info = data_dict['eventinfo']

                rinput = data_dict['rinput']
                lonput = data_dict['lonput']
                latput = data_dict['latput']
                launchslider = data_dict['launchvalue']
                refframe = data_dict['refframe']
                modelstatevars = data_dict['modelstatevars']
            if 'cdf' in list_of_names[0]:
                firstdate, sc, filename = process_cdf(list_of_names, list_of_contents)
                dateFormat = "%Y%m%d"
                input_datetime = datetime.datetime.strptime(firstdate, dateFormat)
                endtime_formatted = input_datetime #+ datetime.timedelta(hours=20)
                input_datetime_formatted = input_datetime.strftime("%Y-%m-%dT%H:%M:%S%z")
                
                event_info = create_event_info([input_datetime_formatted],
                                               [input_datetime_formatted],
                                               [endtime_formatted],
                                               [sc],
                                               [f"ICME_{sc}_CUSTOM_{firstdate}"],
                                               loaded = filename,
                                              )

            else:
                print('datatype not supported, upload cdf or txt file')

        
    elif 'submitmanual' in changed_id:
                
        if (manual_sc == "NOAA_RTSW (realtime)"):
            sc = "NOAA_RTSW"
        elif (manual_sc == "STEREO-A_beacon (realtime)"):
            sc = "STEREO-A_beacon"
            
        else:
            sc = get_alternative_sc(manual_sc)
            
        event = f"ICME_{sc}_CUSTOM_{manual_date.replace('-', '')}"
        dateFormat = "%Y-%m-%d"
        input_datetime = datetime.datetime.strptime(manual_date, dateFormat)
        endtime_formatted = input_datetime #+ datetime.timedelta(hours=20)
        input_datetime_formatted = input_datetime.strftime("%Y-%m-%dT%H:%M:%S%z")

        if times == '':
            event_info = create_event_info([input_datetime_formatted],
                                           [input_datetime_formatted],
                                           [endtime_formatted],
                                           [sc],
                                           [f"ICME_{sc}_CUSTOM_{manual_date.replace('-', '')}"]
                                          )
        else:
            # Convert date and time strings to datetime.datetime objects
            date_time_strings = times['props']['children'].split(' ---> ')
            datetime_objects = [datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M') for dt_str in date_time_strings]
            event_info = create_event_info([input_datetime_formatted],
                                           [datetime_objects[0]],
                                           [datetime_objects[1]],
                                           [sc],
                                           [f"ICME_{sc}_CUSTOM_{manual_date.replace('-', '')}"]
                                          )
    
    return rinput, lonput, latput, event_info, launchslider, refframe, *modelstatevars
    
    

@callback(
    Output("upload-container", "style"),
    Input("accordion", "active_item")
)
def update_upload(activeitem):

    if activeitem == "item-2":
        return {"marginLeft": 340,
                #"marginTop": 20
                "visibility": "visible",
               }
        
    return {"marginLeft": 340,
                #"marginTop": 20
                "visibility": "hidden",
               }

@callback(
    Output("accordion", "active_item"),
    Input("upload-data", "contents")
)
def update_upload(contents):
    
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    if 'upload-data' in changed_id:
        return None
        
        
    return no_update
   
@callback(
    Output("event_dropdown", "options"),
    Input('sc-dropdown-cat', "value"),
    Input('year_dropdown', "value"),
    Input('month_dropdown', "value"),
    Input('day-dropdown', "value"),
)
def update_event_dropdown(sc, year, month, day):
    options = get_catevents(sc, year, month, day)
    return options