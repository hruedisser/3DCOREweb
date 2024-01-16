from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly_express as px
import plotly.figure_factory as ff

import coreweb

from coreweb.dashcore.utils.utils import cart2sphere, sphere2cart, round_to_hour_or_half, get_iparams_live, process_coordinates, plot_body3d
import coreweb.dashcore.utils.heliocats as hc

import numpy as np 

import datetime

def plot_insitu(names, t_data, b_data,view_legend_insitu):
    
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scatter(
            x=t_data,
            y=b_data[:, 0],
            name=names[0],
            line_color='red',
            line_width = 1,
            showlegend=view_legend_insitu,
            legendgroup = '1'
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
            showlegend=view_legend_insitu,
            legendgroup = '1'
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
            showlegend=view_legend_insitu,
            legendgroup = '1'
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
            showlegend=view_legend_insitu,
            legendgroup = '1'
        ),
        row=1, col=1
    )

    fig.update_yaxes(title_text='B [nT]', row=1, col=1)
    fig.update_yaxes(showgrid=True, zeroline=False, showticklabels=True,
                     showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid', spikethickness=1)
    fig.update_xaxes(showgrid=True, zeroline=False, showticklabels=True, rangeslider_visible=False,
                     showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid', spikethickness=1)

    
    return fig

import math

def angular_speed_in_degrees_per_hour(semi_major_axis_AU):
    # Constants
    G = 6.674 * 10**(-11)  # Gravitational constant in m^3 kg^(-1) s^(-2)
    M = 1.989 * 10**(30)   # Mass of the Sun in kg

    # Convert semi-major axis from AU to meters
    semi_major_axis_meters = semi_major_axis_AU * 1.496 * 10**(11)

    # Calculate orbital period in seconds
    orbital_period_seconds = math.sqrt((4 * math.pi**2 * semi_major_axis_meters**3) / (G * M))

    # Calculate angular speed in radians per second
    angular_speed_radians_per_second = 2 * math.pi / orbital_period_seconds

    # Convert angular speed to degrees per hour
    angular_speed_degrees_per_hour = angular_speed_radians_per_second * (360 / (2 * math.pi)) * 3600

    return angular_speed_degrees_per_hour

def get_longmove_array(longmove, rinput, lonput, latput, graph):
    
    desired_length = len(graph['t_data'])
    pos_array = np.empty((desired_length, 3))

    if longmove == 0:
        pos_array[:, 0], pos_array[:, 1], pos_array[:, 2] = sphere2cart(float(rinput), np.deg2rad(-float(latput)+90), np.deg2rad(float(lonput)))

    else:
        if longmove == True:
            anglestep = angular_speed_in_degrees_per_hour(rinput)
            longmove = anglestep/60
            print('spacecraft position changes by ' + str(anglestep) + '° per hour')


        else:
            print('spacecraft position changes by ' + str(longmove) + '° per timestep')
        pos_array_n = np.empty((desired_length, 3))
        pos_array_n[:, 0], pos_array_n[:, 1], pos_array_n[0, 2] = rinput, latput, lonput
        pos_array[0, 0], pos_array[0, 1], pos_array[0, 2] = sphere2cart(float(pos_array_n[0, 0]), np.deg2rad(-float(pos_array_n[0, 1])+90), np.deg2rad(float(pos_array_n[0, 2])))
        for i in range(1, desired_length):
            pos_array_n[i,2] = pos_array_n[i-1,2]+longmove
            #print(pos_array_n[i,2])
            pos_array[i, 0], pos_array[i, 1], pos_array[i, 2] = sphere2cart(float(pos_array_n[i, 0]), np.deg2rad(-float(pos_array_n[i, 1])+90), np.deg2rad(float(pos_array_n[i, 2])))
            #print(pos_array[i, 0], pos_array[i, 1], pos_array[i, 2])

    return pos_array



def check_animation(pos_array, results, plottheme, graph, reference_frame, rinput, lonput, latput, timeslider, infodata, launchlabel, plotoptions, spacecraftoptions, bodyoptions, insitu, positions, view_legend_insitu, camera, posstore, *modelstatevars):
    template = "none"
    bg_color = 'rgba(0, 0,0, 0)'
    line_color = 'white'
    line_colors = ['red','green','blue','black']

    sc = "SYN"
                
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
                

    if launchlabel == None:
        roundedlaunch = datetime.datetime(2012,12,21,6)

    else:
        datetime_format = "Launch Time: %Y-%m-%d %H:%M"
        #(launchlabel)
        try:
            t_launch = datetime.datetime.strptime(launchlabel, datetime_format)
        except:
            t_launch = datetime.datetime.strptime(launchlabel[0], datetime_format)

        roundedlaunch = round_to_hour_or_half(t_launch) 
        

    if "Synthetic Event" in plotoptions:
        iparams = get_iparams_live(*modelstatevars)
        model_obj = coreweb.ToroidalModel(roundedlaunch, **iparams) # model gets initialized
        model_obj.generator()  
        
    posrow = 1
    
    t_data = graph['t_data']

    if reference_frame == "HEEQ":
        b_data = graph['b_data_HEEQ']
        names = ['Bx', 'By', 'Bz']
    else:
        b_data = graph['b_data_RTN']
        names = ['Br', 'Bt', 'Bn']

    
    if insitu and positions:
        insiturow = 3
        dim = 2
        
    elif insitu or positions:
        dim = 1
        insiturow = 1
    else:
        return {}
    
    specs=[]
    subtitles = []
    
    
    if positions:
        
        ndims = dim + 1
        specs.append([{"type": "scene", "rowspan": 2}])
        specs.append([None])
        subtitles.append(str(roundedlaunch + datetime.timedelta(hours=timeslider)))
    else:
        ndims = dim
        
    if insitu:
        specs.append([{"type": "xy"}])
        subtitles.append(str(infodata['id'][0]+'_'+reference_frame))
        
        
    if "Title" in plotoptions:
        fig = make_subplots(rows=ndims, cols=1, specs = specs, subplot_titles = subtitles)
    else:
        fig = make_subplots(rows=ndims, cols=1, specs = specs)    
    
    
    
    
    ############### POLAR THING ###################
    
    if positions:
        if plottheme == 'dark' or plottheme == 'dark-simple':
            template = "plotly_dark"
            bg_color = 'rgba(255, 255, 255, 0)'
            line_color = 'white'
            line_colors = ['orange','lime','deepskyblue','white']
            eventshade = "white"
            framecolor = 'rgba(100, 100, 100, 0.8)'
            if plottheme == 'dark-simple':
                cmecolor = 'rgba(100, 100, 100, 0.8)'
            else:
                cmecolor = 'orange'
        
        else:
            template = "none"  
            bg_color = 'rgba(0,0,0,0)'
            line_color = 'black'
            line_colors = ['red','green','blue','black']
            eventshade = "LightSalmon"
            framecolor = 'rgba(100, 100, 100, 0.8)'
            if plottheme == 'light-simple':
                cmecolor = 'rgba(100, 100, 100, 0.8)'
            else:
                cmecolor = 'red'
            
            
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
                           line=dict(width=1, color=cmecolor),
                           showlegend=False), row=posrow, col=1)

            # Transpose the wf_array to extract wireframe points along the other direction
            x_wire = wf_array[:,:,0].T.flatten()
            y_wire = wf_array[:,:,1].T.flatten()
            z_wire = wf_array[:,:,2].T.flatten()

            # Create another 3D wireframe plot using plotly
            fig.add_trace(go.Scatter3d(x=x_wire, y=y_wire, z=z_wire, mode='lines',
                           line=dict(width=1, color=cmecolor),
                           showlegend=False), row=posrow, col=1)
            
            
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
                name='Sun',
                legendgroup = '1'
            )

            fig.add_trace(sun_trace, row=posrow, col=1)

        if "Earth" in bodyoptions:

            # Create data for the Earth
            #earth_trace = go.Scatter3d(
            #    x=[1], y=[0], z=[0],
            #    mode='markers',
            #    marker=dict(size=4, color='mediumseagreen'),
            #    name='Earth',
            #    legendgroup = '1'
            #)
            try:
                fig.add_trace(plot_body3d(graph['bodydata']['Earth']['data'], roundedlaunch + datetime.timedelta(hours=timeslider), 'mediumseagreen', 'Earth')[0])
            except Exception as e:
                print('Data for Earth not found: ', e)

            #fig.add_trace(earth_trace, row=posrow, col=1)
                        
        if "Mercury" in bodyoptions:
            try:
                fig.add_trace(plot_body3d(graph['bodydata']['Mercury']['data'], roundedlaunch + datetime.timedelta(hours=timeslider), 'slategrey', 'Mercury',legendgroup = '1')[0], row=posrow, col=1)
            except Exception as e:
                print('Data for Mercury not found: ', e)
            
            
        if "Venus" in bodyoptions:
            try:
                fig.add_trace(plot_body3d(graph['bodydata']['Venus']['data'], roundedlaunch + datetime.timedelta(hours=timeslider), 'darkgoldenrod', 'Venus',legendgroup = '1')[0], row=posrow, col=1)
            except Exception as e:
                print('Data for Venus not found: ', e)
            
        if "Mars" in bodyoptions:
            try:
                fig.add_trace(plot_body3d(graph['bodydata']['Mars']['data'], roundedlaunch + datetime.timedelta(hours=timeslider), 'red', 'Mars',legendgroup = '1')[0], row=posrow, col=1)
            except Exception as e:
                print('Data for Mars not found: ', e)
            
        if spacecraftoptions is not None:
            for scopt in spacecraftoptions:
                #try:
                if scopt == "SYN":
                    #try:
                    #print(timeslider)
                    x,y,z = pos_array[60*int(timeslider)] #sphere2cart(float(rinput), np.deg2rad(-float(latput)+90), np.deg2rad(float(lonput)))
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
                            legendgroup = '1',
                            hovertemplate="<b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                         + scopt + "</extra>"
                        ), row=posrow, col=1)
                    

                    fig.add_trace(
                        go.Scatter3d(
                            x=pos_array[:,0], y=pos_array[:,1], z=pos_array[:,2],
                            mode='lines', 
                            name="SYN",
                            #customdata=np.vstack((rinput, latput, lonput)).T,
                            showlegend=False,
                            #legendgroup = '1',
                            #hovertemplate="<b>(x, y, z):</b> (%{x:.2f} AU, %{y:.2f} AU, %{z:.2f} AU)<br><b>(r, lon, lat):</b> (%{customdata[0]:.2f} AU, %{customdata[2]:.2f}°, %{customdata[1]:.2f}°)<extra>" 
                         #+ scopt + "</extra>"
                        ), row=posrow, col=1)
                    #except:
                     #   pass

                else:                    
                    traces = process_coordinates(posstore[scopt]['data']['data'], roundedlaunch, roundedlaunch + datetime.timedelta(hours=timeslider), posstore[scopt]['data']['color'], scopt, legendgroup='1')
                    if "Trajectories" in plotoptions:
                        fig.add_trace(traces[0], row=posrow, col=1)
                        fig.add_trace(traces[1], row=posrow, col=1)

                    fig.add_trace(traces[2], row=posrow, col=1)
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
                fig.add_trace(circle_trace, row=posrow, col=1)

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
                fig.add_trace(label_trace, row=posrow, col=1)

            
            
            
            
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
                fig.add_trace(au_line, row=posrow, col=1)

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
                fig.add_trace(label_trace, row=posrow, col=1)
                
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
                    line=dict(color=framecolor),
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(circle_trace, row=posrow, col=1)

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
                fig.add_trace(label_trace, row=posrow, col=1)

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
                    line=dict(color=framecolor),
                    name=f'{angle_degrees}°',
                    showlegend=False,
                    hovertemplate = None, 
                    hoverinfo = "skip", 
                )
                fig.add_trace(au_line, row=posrow, col=1)

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
                fig.add_trace(label_trace, row=posrow, col=1)

        
            
        if "Timer" in plotoptions:
            if insitu:
                fig.add_annotation(text=f"t_launch + {timeslider} h", xref="paper", yref="paper", x=0.5, y=1.05, showarrow=False) #, row=posrow, col=1)
            else:
                fig.add_annotation(text=f"t_launch + {timeslider} h", xref="paper", yref="paper", x=0.5, y=1.1, showarrow=False) #, row=posrow, col=1)
        
        if "Title" in plotoptions:
            if insitu:
                fig.add_annotation(text="HEEQ", xref="paper", yref="paper", x=1., y=0.51, showarrow=False) #, row=row, col=1)
            else:
                fig.add_annotation(text="HEEQ", xref="paper", yref="paper", x=1., y=0.01, showarrow=False) #, row=row, col=1)
        
        

        # Set the layout
        fig.update_layout(
            template=template, 
            plot_bgcolor=bg_color,  # Background color for the entire figure
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title = '',showspikes=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title = '', showspikes=False),
                zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, title = '', showspikes=False, range=[0, 0]),  # Adjust the range as needed
                aspectmode='cube',
                
            bgcolor=bg_color,
            )#, row=posrow, col=1
        )
    
    
    
    
    ############## INSITU THING ###################
    
    if results is not None:
        
        
    
        if reference_frame == 'HEEQ':
            ed = results['ensemble_HEEQ']
        else:
            ed = results['ensemble_RTN']

        #print(len(ed[0][3][0]))
        #print(len(graph['t_data']))
        
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
                    x=t_data,
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
                    x=t_data,
                    y=y2,
                    fill='tonexty',
                    mode='lines',
                    line_color=line_color,
                    line_width=0,
                    fillcolor=fill_color,
                    showlegend=False
                )
            )

    if insitu:
        row = ndims

        if np.all(np.isnan(b_data)) == False:

            fig.add_trace(
                go.Scatter(
                    x=t_data,
                    y=b_data[:, 0],
                    name=names[0],
                    line_color=line_colors[0],
                    line_width = 1,
                    showlegend=view_legend_insitu,
                    legendgroup = '2',
                ),
                row=row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=t_data,
                    y=b_data[:, 1],
                    name=names[1],
                    line_color=line_colors[1],
                    line_width = 1,
                    showlegend=view_legend_insitu,
                    legendgroup = '2',
                ),
                row=row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=t_data,
                    y=b_data[:, 2],
                    name=names[2],
                    line_color=line_colors[2],
                    line_width = 1,
                    showlegend=view_legend_insitu,
                    legendgroup = '2',
                ),
                row=row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=t_data,
                    y=np.sqrt(np.sum(b_data**2, axis=1)),
                    name='Btot',
                    line_color=line_colors[3],
                    line_width = 1,
                    showlegend=view_legend_insitu,
                    legendgroup = '2',
                ),
                row=row, col=1
            )
        
        if "Catalog Event" in plotoptions:
            # Define the shape for the vertical rectangle
            rect_shape = {
                'type': 'rect',
                'x0': begin,
                'x1': end,
                'y0': -100,  # Adjust the y0 and y1 values as needed
                'y1': 100,
                'fillcolor': eventshade,
                'opacity': opac,
                'layer': "below",
                'line_width': 0,
                'xref': 'x',  # Position relative to the x-axis
                'yref': 'y2'  # Position relative to the second subplot
            }

            # Add the shape to the data of the figure
            fig.add_shape(rect_shape,row=row, col=1)
            
            
            
            
        if "Synthetic Event" in plotoptions:

            
            # Create ndarray with dtype=object to handle ragged nested sequences
            if sc == "SYN":
                try:
                    outa = np.array(model_obj.simulator(graph['t_data'], pos_array), dtype=object)
                    
                    
                
                except Exception as e:
                    print(e)
                    outa = np.array(model_obj.simulator(graph['t_data'], graph['pos_data']), dtype=object)
            else:
                outa = np.array(model_obj.simulator(graph['t_data'], graph['pos_data']), dtype=object)
                #print(graph['t_data'])
                
                #print(graph['pos_data'])
            
            outa = np.squeeze(outa[0])
            
            if sc == "SYN":
                if reference_frame == "RTN":
                    rtn_bx, rtn_by, rtn_bz = hc.convert_HEEQ_to_RTN_mag(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], outa[:, 0],outa[:, 1],outa[:, 2])
                    outa[:, 0],outa[:, 1],outa[:, 2] = rtn_bx, rtn_by, rtn_bz
            else:
                if reference_frame == "RTN":
                    x,y,z = hc.separate_components(graph['pos_data'])
                    #print(x,y,z)
                    rtn_bx, rtn_by, rtn_bz = hc.convert_HEEQ_to_RTN_mag(x,y,z, outa[:, 0],outa[:, 1],outa[:, 2])
                    outa[:, 0],outa[:, 1],outa[:, 2] = rtn_bx, rtn_by, rtn_bz
                        
            
            outa[outa==0] = np.nan

            names = graph['names']

            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=outa[:, 0],
                    line=dict(color=line_colors[0], width=3, dash='dot'),
                    name=names[0]+'_synth',
                    legendgroup = '2',
                ),
                row=row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=outa[:, 1],
                    line=dict(color=line_colors[1], width=3, dash='dot'),
                    name=names[1]+'_synth',
                    legendgroup = '2',
                ),
                row=row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=outa[:, 2],
                    line=dict(color=line_colors[2], width=3, dash='dot'),
                    name=names[2]+'_synth',
                    legendgroup = '2',
                ),
                row=row, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=graph['t_data'],
                    y=np.sqrt(np.sum(outa**2, axis=1)),
                    line=dict(color=line_colors[3], width=3, dash='dot'),
                    name='Btot_synth',
                    legendgroup = '2',
                ),
                row=row, col=1
            )
            
            
            rect_shape = {
                'type': 'rect',
                'x0': roundedlaunch + datetime.timedelta(hours=timeslider),
                'x1': roundedlaunch + datetime.timedelta(hours=timeslider),
                'y0': -100,  # Adjust the y0 and y1 values as needed
                'y1': 100,
                'fillcolor': "Red",
                'line': dict(color="Red", width=2), #, width=3, dash='dot'),
                'type' : 'line',
                #'opacity': opac,
                'layer': "above",
                #name="Current Time",
                'xref': 'x',  # Position relative to the x-axis
                'yref': 'y2'  # Position relative to the second subplot
            }
            
            # Add the shape to the data of the figure
            fig.add_shape(rect_shape,row=row, col=1)
            
        # Calculate the y-axis limits for the second subplot
        min_b_data = np.min(b_data)
        max_b_data = np.max(b_data)
        y_range_padding = 10  # Adjust this value as needed

        if sc == "SYN":
            # Sample a subset of time steps for tick labels
            max_ticks = 10  # Adjust the number of ticks as needed
            sampled_ticks = graph['t_data'][::len(graph['t_data']) // max_ticks]

            # Update x-axis tick labels
            tick_labels = [("+ " + str(int((i - roundedlaunch).total_seconds()/3600)) + " h") for i in sampled_ticks]

            fig.update_xaxes(tickvals=sampled_ticks, ticktext=tick_labels, row=row, col=1)



        fig.update_yaxes(title_text='B [nT]', row=row, col=1, range=[min_b_data - y_range_padding, max_b_data + y_range_padding])
        fig.update_yaxes(showgrid=True, zeroline=False, showticklabels=True,
                         showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
                         spikethickness=1, row=row, col=1)
        fig.update_xaxes(showgrid=True, zeroline=False, showticklabels=True, rangeslider_visible=False,
                         showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
                         spikethickness=1, row=row, col=1)

    
    height = 0
    if positions:
        height = height + 700
    if insitu:
        height = height + 350
    
    gapwidth = 550 + 5 * len(bodyoptions) + 5 * len(spacecraftoptions)
    
    if camera != 'auto':
        
        zeropoint = [0,206.62,90]
        for i in range(3):
            camera[i] = camera[i] + zeropoint[i]
        xcam, ycam, zcam = sphere2cart(float(camera[0]),np.deg2rad(-float(camera[1]))+90,np.deg2rad(float(camera[2])))
        camera = dict(
            eye=dict(x=xcam, y=ycam, z=zcam)
        )
        #print(xcam, ycam, zcam)
        
        fig.update_layout(height=height, width = 1000, showlegend=view_legend_insitu, scene_camera = camera, legend_tracegroupgap = gapwidth)
    else:
        fig.update_layout(height=height, width = 1000, showlegend=view_legend_insitu, legend_tracegroupgap = gapwidth)
    
    #fig.show()
    return fig





def check_fittingpoints(graph, reference_frame, infodata, view_legend_insitu, showtitle = True, t_fit = None, t_s = None, t_e= None):
    
    
    sc = infodata['sc'][0]
    if t_s == None:
        begin = infodata['begin'][0]
        end = infodata['end'][0]
    else:
        begin = t_s
        end = t_e
    
    template = "none"
    bg_color = 'rgba(0, 0,0, 0)'
    line_color = 'black'
    line_colors = ['red','green','blue','black']
    eventshade = "LightSalmon"

    if infodata['id'][0] == 'I':
        opac = 0
    else:
        opac = 0.5

    dateFormat = "%Y-%m-%dT%H:%M:%S%z"
    dateFormat2 = "%Y-%m-%d %H:%M:%S"
    dateFormat3 = "%Y-%m-%dT%H:%M:%S"

    try:
        begin = datetime.datetime.strptime(begin, dateFormat2)
    except:
        try:
            begin = datetime.datetime.strptime(begin, dateFormat)
        except:
            try:
                begin = datetime.datetime.strptime(begin, dateFormat3)
            except:
                pass

    try:
        end = datetime.datetime.strptime(end, dateFormat2)
    except:
        try:
            end = datetime.datetime.strptime(end, dateFormat)
        except:
            try:
                end = datetime.datetime.strptime(end, dateFormat3)
            except:
                pass
            
    t_data = graph['t_data']
    if reference_frame == "HEEQ":
        b_data = graph['b_data_HEEQ']
        names = ['Bx', 'By', 'Bz']
    else:
        b_data = graph['b_data_RTN']
        names = ['Br', 'Bt', 'Bn']

    
    if showtitle:
        fig = make_subplots(rows=1, cols=1, specs = [[{"type": "xy"}]], subplot_titles = [str(infodata['id'][0]+'_'+reference_frame)])
    else:
        fig = make_subplots(rows=1, cols=1, specs = [[{"type": "xy"}]])    
    
    
    
    

    
    ############## INSITU THING ###################
    
    # Calculate the new time range for the plot
    new_begin = begin - datetime.timedelta(hours=12)
    new_end = end + datetime.timedelta(hours=12)

    # Filter the data to the new time range
    mask = (t_data >= new_begin.replace(tzinfo=None)) & (t_data <= new_end.replace(tzinfo=None))
    t_data = t_data[mask]
    b_data = b_data[mask]
    
    fig.add_trace(
        go.Scatter(
            x=t_data,
            y=b_data[:, 0],
            name=names[0],
            line_color=line_colors[0],
            line_width = 1,
            showlegend=view_legend_insitu,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=t_data,
            y=b_data[:, 1],
            name=names[1],
            line_color=line_colors[1],
            line_width = 1,
            showlegend=view_legend_insitu,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=t_data,
            y=b_data[:, 2],
            name=names[2],
            line_color=line_colors[2],
            line_width = 1,
            showlegend=view_legend_insitu,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=t_data,
            y=np.sqrt(np.sum(b_data**2, axis=1)),
            name='Btot',
            line_color=line_colors[3],
            line_width = 1,
            showlegend=view_legend_insitu,
        ),
        row=1, col=1
    )
        
    # Define the shape for the vertical rectangle
    rect_shape = {
        'type': 'rect',
        'x0': begin,
        'x1': end,
        'y0': -100,  # Adjust the y0 and y1 values as needed
        'y1': 100,
        'fillcolor': eventshade,
        'opacity': opac,
        'layer': "below",
        'line_width': 0,
        'xref': 'x',  # Position relative to the x-axis
        'yref': 'y2'  # Position relative to the second subplot
    }

    # Add the shape to the data of the figure
    fig.add_shape(rect_shape,row=1, col=1)
    
    
    if t_s == None:
        t_s = begin
        
    if t_e == None:
        t_e = end
        
    if t_fit == None or t_fit == []:
        time_difference = (t_e - t_s)

        # Calculate the interval between each of the four times
        interval = time_difference / 5
        
        # Calculate and round t_1, t_2, t_3, and t_4
        t_1 = t_s + interval
        t_2 = t_1 + interval
        t_3 = t_2 + interval
        t_4 = t_3 + interval

        # Round the times to the nearest 30-minute precision
        t_1 = t_1 - datetime.timedelta(minutes=t_1.minute % 30, seconds=t_1.second, microseconds=t_1.microsecond)
        t_2 = t_2 - datetime.timedelta(minutes=t_2.minute % 30, seconds=t_2.second, microseconds=t_2.microsecond)
        t_3 = t_3 - datetime.timedelta(minutes=t_3.minute % 30, seconds=t_3.second, microseconds=t_3.microsecond)
        t_4 = t_4 - datetime.timedelta(minutes=t_4.minute % 30, seconds=t_4.second, microseconds=t_4.microsecond)
        
        t_fit = [t_1, t_2, t_3, t_4]
    
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
            
            
            
        
            
    # Calculate the y-axis limits for the second subplot
    min_b_data = np.nanmin(b_data)
    max_b_data = np.nanmax(b_data)
    y_range_padding = 10  # Adjust this value as needed

    fig.update_yaxes(title_text='B [nT]', row=1, col=1, range=[min_b_data - y_range_padding, max_b_data + y_range_padding], showgrid=True, zeroline=False, showticklabels=True,
                     showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
                     spikethickness=1)
    fig.update_xaxes(showgrid=True, zeroline=False, showticklabels=True, rangeslider_visible=False,
                     showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
                     spikethickness=1, row=1, col=1)

    
    height = 350
    
   
    
    fig.update_layout(height=height, width = 1000, showlegend=view_legend_insitu)
    
    #fig.show()
    return fig, t_s, t_e, t_fit



def plot_simpleres(pos_array, results, plottheme, graph, reference_frame, infodata, view_legend_insitu, showtitle = True, t_s = None, t_e= None):
    
    
    sc = infodata['sc'][0]
    if t_s == None:
        begin = infodata['begin'][0]
        end = infodata['end'][0]
    else:
        begin = t_s
        end = t_e
    
    template = "none"
    bg_color = 'rgba(0, 0,0, 0)'
    line_color = 'black'
    line_colors = ['red','green','blue','black']
    eventshade = "LightSalmon"

    if infodata['id'][0] == 'I':
        opac = 0
    else:
        opac = 0.5

    dateFormat = "%Y-%m-%dT%H:%M:%S%z"
    dateFormat2 = "%Y-%m-%d %H:%M:%S"
    dateFormat3 = "%Y-%m-%dT%H:%M:%S"

    try:
        begin = datetime.datetime.strptime(begin, dateFormat2)
    except:
        try:
            begin = datetime.datetime.strptime(begin, dateFormat)
        except:
            try:
                begin = datetime.datetime.strptime(begin, dateFormat3)
            except:
                pass

    try:
        end = datetime.datetime.strptime(end, dateFormat2)
    except:
        try:
            end = datetime.datetime.strptime(end, dateFormat)
        except:
            try:
                end = datetime.datetime.strptime(end, dateFormat3)
            except:
                pass
            
    t_data = graph['t_data']
    if reference_frame == "HEEQ":
        b_data = graph['b_data_HEEQ']
        names = ['Bx', 'By', 'Bz']
    else:
        b_data = graph['b_data_RTN']
        names = ['Br', 'Bt', 'Bn']

    
    if showtitle:
        fig = make_subplots(rows=1, cols=1, specs = [[{"type": "xy"}]], subplot_titles = [str(infodata['id'][0]+'_'+reference_frame)])
    else:
        fig = make_subplots(rows=1, cols=1, specs = [[{"type": "xy"}]])    
    
    
    
    

    
    ############## INSITU THING ###################
    
    # Calculate the new time range for the plot
    new_begin = begin - datetime.timedelta(hours=12)
    new_end = end + datetime.timedelta(hours=12)

    # Filter the data to the new time range
    mask = (t_data >= new_begin.replace(tzinfo=None)) & (t_data <= new_end.replace(tzinfo=None))
    t_data = t_data[mask]
    b_data = b_data[mask]
    
    fig.add_trace(
        go.Scatter(
            x=t_data,
            y=b_data[:, 0],
            name=names[0],
            line_color=line_colors[0],
            line_width = 1,
            showlegend=view_legend_insitu,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=t_data,
            y=b_data[:, 1],
            name=names[1],
            line_color=line_colors[1],
            line_width = 1,
            showlegend=view_legend_insitu,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=t_data,
            y=b_data[:, 2],
            name=names[2],
            line_color=line_colors[2],
            line_width = 1,
            showlegend=view_legend_insitu,
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=t_data,
            y=np.sqrt(np.sum(b_data**2, axis=1)),
            name='Btot',
            line_color=line_colors[3],
            line_width = 1,
            showlegend=view_legend_insitu,
        ),
        row=1, col=1
    )
        
    # Define the shape for the vertical rectangle
    rect_shape = {
        'type': 'rect',
        'x0': begin,
        'x1': end,
        'y0': -100,  # Adjust the y0 and y1 values as needed
        'y1': 100,
        'fillcolor': eventshade,
        'opacity': opac,
        'layer': "below",
        'line_width': 0,
        'xref': 'x',  # Position relative to the x-axis
        'yref': 'y2'  # Position relative to the second subplot
    }

    # Add the shape to the data of the figure
    fig.add_shape(rect_shape,row=1, col=1)
    
    
    if t_s == None:
        t_s = begin
        
    if t_e == None:
        t_e = end
        
    if t_fit == None or t_fit == []:
        time_difference = (t_e - t_s)

        # Calculate the interval between each of the four times
        interval = time_difference / 5
        
        # Calculate and round t_1, t_2, t_3, and t_4
        t_1 = t_s + interval
        t_2 = t_1 + interval
        t_3 = t_2 + interval
        t_4 = t_3 + interval

        # Round the times to the nearest 30-minute precision
        t_1 = t_1 - datetime.timedelta(minutes=t_1.minute % 30, seconds=t_1.second, microseconds=t_1.microsecond)
        t_2 = t_2 - datetime.timedelta(minutes=t_2.minute % 30, seconds=t_2.second, microseconds=t_2.microsecond)
        t_3 = t_3 - datetime.timedelta(minutes=t_3.minute % 30, seconds=t_3.second, microseconds=t_3.microsecond)
        t_4 = t_4 - datetime.timedelta(minutes=t_4.minute % 30, seconds=t_4.second, microseconds=t_4.microsecond)
        
        t_fit = [t_1, t_2, t_3, t_4]
    
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
            
            
            
        
            
    # Calculate the y-axis limits for the second subplot
    min_b_data = np.nanmin(b_data)
    max_b_data = np.nanmax(b_data)
    y_range_padding = 10  # Adjust this value as needed

    fig.update_yaxes(title_text='B [nT]', row=1, col=1, range=[min_b_data - y_range_padding, max_b_data + y_range_padding], showgrid=True, zeroline=False, showticklabels=True,
                     showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
                     spikethickness=1)
    fig.update_xaxes(showgrid=True, zeroline=False, showticklabels=True, rangeslider_visible=False,
                     showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='solid',
                     spikethickness=1, row=1, col=1)

    
    height = 350
    
   
    
    fig.update_layout(height=height, width = 1000, showlegend=view_legend_insitu)
    
    #fig.show()
    return fig, t_s, t_e, t_fit