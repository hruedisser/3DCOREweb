from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly_express as px
import plotly.figure_factory as ff

import numpy as np 

def plot_insitu(names, t_data, b_data,view_legend_insitu):
    
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scatter(
            x=t_data,
            y=b_data[:, 0],
            name=names[0],
            line_color='red',
            line_width = 1,
            showlegend=view_legend_insitu
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=t_data,
            y=b_data[:, 1],
            name=names[1],
            line_color='green',
            line_width = 1,
            showlegend=view_legend_insitu
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=t_data,
            y=b_data[:, 2],
            name=names[2],
            line_color='blue',
            line_width = 1,
            showlegend=view_legend_insitu
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=t_data,
            y=np.sqrt(np.sum(b_data**2, axis=1)),
            name='Btot',
            line_color='black',
            line_width = 1,
            showlegend=view_legend_insitu
        ),
        row=1, col=1
    )

    fig.update_yaxes(title_text='B [nT]', row=1, col=1)
    fig.update_yaxes(showgrid=True, zeroline=False, showticklabels=True,
                     showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid', spikethickness=1)
    fig.update_xaxes(showgrid=True, zeroline=False, showticklabels=True, rangeslider_visible=False,
                     showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid', spikethickness=1)

    
    return fig