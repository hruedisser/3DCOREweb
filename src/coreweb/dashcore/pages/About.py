from dash import dcc, register_page, html
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

from PIL import Image
import os

image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "3dcore.png"))

pil_image = Image.open(image_path)




register_page(__name__, icon="gis:satellite", order=3)

layout = dbc.Container(
    [
        html.Br(),
        html.H2("About this application:", 
                className="display-10"),
        html.Br(),   
        html.Hr(),
        html.Br(),
        dcc.Markdown(
            """ 
            You have several options to analyze the chosen event: \n
            
            **Fitting:** \n
            >_Fit the model numerically or scroll through the results of a fitting run._\n
            
            **Insitu:** \n
            >_Look at the insitu data measured by a specific observer or generate synthetic insitu data._ \n
            
            **Positions:** \n
            >_Take a look at planet and spacecraft positions and model the 3D shape of the CME._ \n
            """
        ),
        
        html.Br(),
        html.Hr(),
        html.Br(),
        dbc.Row([
            dbc.Col(
                [dcc.Markdown(
                 """
                 _3DCOREweb_  is an open-source software package based on 3DCORE, that can be used to
                 reconstruct the 3D structure of Coronal Mass Ejections (CMEs) and create synthetic
                 insitu signatures. It can be fitted to insitu data from several spacecraft using an
                 Approximate Bayesian Computation Sequential Monte Carlo (ABC-SMC) algorithm, model
                 their kinematics and compare to remote-sensing observations.
                 The 3DCORE model assumes an empirically motivated torus-like flux rope structure that
                 expands self-similarly within the heliosphere, is influenced by a simplified interaction
                 with the solar wind environment, and carries along an embedded analytical magnetic
                 field. The tool also implements remote-sensing observations from multiple viewpoints
                 such as the SOlar and Heliospheric Observatory (SOHO) and Solar Terrestrial Relations
                 Observatory (STEREO).
                 """
             ),
             ], width = 6),
         dbc.Col([html.Div(
             html.Img(
                 src=pil_image,
                 style={
                     'max-width': '80%',
                     'max-height': '80%',
                     'display': 'block',
                     'margin-left': 'auto',
                     'margin-right': 'auto',
                     'margin-top': 'auto',
                     'margin-bottom': 'auto'
                 }
             ),
             style={
                 'height': '100%',
                 'display': 'flex',
                 'justify-content': 'center',
                 'align-items': 'center'
             }
         )
                 ], width=6)
     ]),
     html.Hr(),
     html.Br(),
     dbc.Row([
         dbc.Col(
             [dcc.Link(
         dmc.Group(
             [
                 dmc.ThemeIcon(
                     DashIconify(icon='devicon:github', width=18),
                     size=40,
                     radius=40,
                     variant="light",
                     style={"backgroundColor": "#eaeaea", "marginRight": "12px"}
                 ),
                 dmc.Text('Github', size="l", color="gray", weight=500),
                 dmc.Text('Find the latest version here', size="l", color="black", weight=345)
             ],
             style={"display": "flex", 
                    "alignItems": "center", 
                    "justifyContent": "start"
                   },
         ),
         href="https://github.com/hruedisser/3DCOREweb",
         target="_blank",
         style={"textDecoration": "none",
               },
     ),], width = 6),
      dbc.Col([dcc.Link(
         dmc.Group(
             [dmc.ThemeIcon(
                 DashIconify(icon='simple-icons:arxiv', width=18),
                 size=40,
                 radius=40,
                 variant="light",
                 style={"backgroundColor": "#eaeaea", "color": "black", "marginRight": "12px"}
             ),
              dmc.Text('ArXiv', size="l", color="gray", weight=500),
              dmc.Text('Find associated papers here', size="l", color="black", weight=345)
             ],
             style={"display": "flex", 
                    "alignItems": "center", 
                    "justifyContent": "start", 
                   },
         ),
         href="https://arxiv.org/search/?query=3dcore&searchtype=all&source=header",
         target="_blank",
         style={"textDecoration": "none",
               },
     )], width=6)
     ]),
    ]
)
