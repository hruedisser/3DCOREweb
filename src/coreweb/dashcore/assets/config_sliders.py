from dash import dcc, register_page, html
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

'''
Configuration for the sliders
'''

############################################################
# Geometrical Models

modelslidervars = [{'var_name': 'Longitude (HEEQ)',
                    'min': 0., 
                 'max': 360., 
                 'step': 0.01, 
                 'def': 0.,
                 'doubl_def': [0., 360],
                 'unit': '[deg.]',
                 'marks': {i: str(i) for i in range(-180, 361, 90)},
                 'variablename': 'longit',
                 'variablename_double': 'longit_double',
               },
                {'var_name':'Latitude (HEEQ)',
                'min': -90., 
                 'max': 90., 
                 'step': 0.01, 
                 'def': 0.,
                 'doubl_def': [-90, 90],
                 'unit': '[deg.]',
                 'marks': {i: str(i) for i in range(-90, 91, 45)},
                 'variablename': 'latitu',
                 'variablename_double': 'latitu_double',
               },
                {'var_name':'Inclination', 
                'min': 0., 
                 'max': 360., 
                 'step': 1., 
                 'def': 0.,
                 'doubl_def': [0., 360],
                 'unit': '[deg.]',
                 'marks': {i: str(i) for i in range(0, 361, 90)},
                 'variablename': 'inc',
                 'variablename_double': 'inc_double'
                },
                {'var_name':'Diameter 1 AU', 
                'min': 0.05, 
                 'max': 0.35, 
                 'step': 0.01, 
                 'def': 0.2,
                 'doubl_def': [0.05, 0.35],
                 'unit': '[AU]',
                 'marks': {0.05: '0.05', 0.15: '0.15',0.25: '0.25',0.35: '0.35'},
                 'variablename': 'dia',
                 'variablename_double': 'dia_double'
                },
                {'var_name':'Aspect Ratio', 
                'min': 1., 
                 'max': 6., 
                 'step': 0.1, 
                 'def':3.,
                 'doubl_def': [1., 6],
                 'unit': '',
                 'marks': {i: str(i) for i in range(0,7, 1)},
                 'variablename': 'asp',
                 'variablename_double': 'asp_double'
                },
                {'var_name':'Launch Radius', 
                'min' : 5., 
                 'max': 100. , 
                 'step': 1., 
                 'def':20.,
                 'doubl_def': [15, 25],
                 'unit': '[R_Sun]',
                 'marks': {5: '5', 25: '25',50: '50',75: '75',100: '100'},
                 'variablename': 'l_rad',
                 'variablename_double': 'l_rad_double'
                },
                {'var_name':'Launch Velocity', 
                'min': 400., 
                 'max': 3000., 
                 'step': 10., 
                 'def':800.,
                 'doubl_def': [400., 1200],
                 'unit': '[km/s]',
                 'marks': {i: str(i) for i in [400, 1000, 1500, 2000, 2500, 3000]},
                 'variablename': 'l_vel',
                 'variablename_double': 'l_vel_double'
                },
                {'var_name':'Expansion Rate', 
                'min': 0.3 , 
                 'max': 2., 
                 'step':0.01 , 
                 'def':1.14,
                 'doubl_def': [1.14, 1.14],
                 'unit': '',
                 'marks': {0.3: '0.3', 1.14: '1.14',2: '2'},
                 'variablename': 'exp_rat',
                 'variablename_double': 'exp_rat_double',
                },
                {'var_name':'Background Drag', 
                'min': 0.2, 
                 'max': 3., 
                 'step': 0.01, 
                 'def':1.,
                 'doubl_def': [0.1, 3.],
                 'unit': '',
                 'marks': {0.2: '0.2', 1: '1',2: '2',3: '3'},
                 'variablename': 'b_drag',
                 'variablename_double': 'b_drag_double',
                },
                {'var_name':'Background Velocity', 
                'min': 100.,
                 'max': 700.,
                 'step': 10., 
                 'def':500.,
                 'doubl_def': [100., 700],
                 'unit': '[km/s]',
                 'marks': {i: str(i) for i in range(100, 701, 100)},
                 'variablename': 'bg_vel',
                 'variablename_double': 'bg_vel_double'
                },
]

magslidervars = [{'var_name': 'T_Factor',
                    'min': -250., 
                 'max': 250., 
                 'step': 1., 
                 'def': 100.,
                 'doubl_def': [-250., 250],
                 'unit': '',
                 'marks': {i: str(i) for i in range(-250, 251, 50)},
                    'variablename': 't_fac',
                 'variablename_double': 't_fac_double'
                },
                   {'var_name': 'Magnetic Decay Rate',
                    'min': 1., 
                 'max': 2., 
                 'step': 0.01, 
                 'def': 1.64,
                 'doubl_def': [1.64, 1.64],
                 'unit': '',
                 'marks': {1: '1', 1.64: '1.64',2: '2'},
                 'variablename': 'mag_dec',
                'variablename_double': 'mag_dec_double'
                           },
                   {'var_name': 'Magnetic Field Strength 1 AU',
                    'min': 5., 
                 'max': 150., 
                 'step': 1., 
                 'def': 25.,
                 'doubl_def': [5., 100.],
                 'unit':  '[nT]',
                 'marks': {i: str(i) for i in [5, 25, 50, 75, 100, 125, 150]},
                 'variablename': 'mag_strength',
                    'variablename_double': 'mag_strength_double'
                           },
                  
                  ]
fittingstate = ["launch-label", "spacecrafttable"] + [item['variablename_double'] for item in modelslidervars + magslidervars] + ["particle-slider", "reference_frame", "fitter-radio", 'n_jobs', "multiprocesscheck", 'n_iter',"ensemble-slider"]

modelstate = [item['variablename'] for item in modelslidervars + magslidervars] 


dataarchive =html.Div([
    html.Br(),
        html.Hr(),
        html.Br(),
        dbc.Row([
            
            dbc.Col(
                [dcc.Link(
                    dmc.Group(
                        [
                            dmc.ThemeIcon(
                                DashIconify(icon='ph:folder-bold', width=18, style={"color": "black"}),
                                size=40,
                                radius=40,
                                variant="light",
                                style={"backgroundColor": "#eaeaea", "marginRight": "12px"}
                            ),
                            dmc.Text('Data Archive', size="l", color="gray", weight=500),
                        ],
                        style={"display": "flex", 
                               "alignItems": "center", 
                               "justifyContent": "start"
                              },
                    ),
                    href="https://doi.org/10.6084/m9.figshare.11973693.v23",
                    target="_blank",
                    style={"textDecoration": "none",
                          },
                ),
                ], width = 3),
            dbc.Col([
                dmc.Text(
                    """ 
                    Consider downloading the full data archive to avoid the need for automatic data retrieval during analysis of an event. Place the files in 3DCOREweb/src/coreweb/dashcore/data/archive.
                    """
                , size="l", color="black", weight=345)], width=9),
        ]),
    html.Br(),
])