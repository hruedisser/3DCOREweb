import plotly.graph_objs as go
import plotly.express as px
import numpy as np 
from plotly.subplots import make_subplots


def direction_from_pts(x1,x2,North_to_south,left_handed):
    if((North_to_south and left_handed) or (not North_to_south and  not left_handed)):
        #prev to next 
        direction = np.array([
                    x1[0]-x2[0],
                    x1[1]-x2[1],
                    x1[2]-x2[2],
        ])
    elif((North_to_south and not left_handed)  or (not North_to_south and  left_handed)):
         #next to prev 
        direction = np.array([
                    x2[0]-x1[0],
                    x2[1]-x1[1],
                    x2[2]-x1[2],
        ])
    direction = direction / np.linalg.norm(direction)
    return direction

def create_twist(fig,row,col,North_to_south = False,left_handed= True,resolution = 1000,turns = 10,line_thickness_red=20.0,red_arrow_thickness=0.6,black_arrow_thickness=0.55):
    
    if(left_handed):
        start = 0
    else:
        start = 0.5*np.pi



    if(not left_handed):
        if(turns%2==0):
            if((turns * 270 + 90.0)%360==90.0):
                theta  = np.linspace(start,  np.deg2rad(turns * 270 + 90.0 ), resolution)
            else:
                theta  = np.linspace(start,  np.deg2rad(turns * 270 - 90.0 ), resolution)
        else:
            theta  = np.linspace(start,  np.deg2rad(turns *360 +  90.0 ), resolution)
    else:
        if(turns%2==0):
            # print("good 4: ",(4 * 270.0)//180.0,"bad 6: ",(6 * 270.0)//180.0,"good 8: ",(8 * 270.0)//180.0,"bad 10: ",(10 * 270.0)//180.0)
            if(((turns * 270.0)//180.0)%2==0):
                theta  = np.linspace(start,  np.deg2rad(turns * 270  ), resolution)
            else:
                theta  = np.linspace(start,  np.deg2rad(turns * 270 - 180 ), resolution)
        else:
            theta  = np.linspace(start,  np.deg2rad(turns *360), resolution)


    Z = np.linspace(0,turns, resolution) #here
    if(left_handed):
        X =  2.0*np.sin(theta)
        Y =  2.0*np.cos(theta)
    else:
        X =  2.0*np.cos(theta)
        Y =  2.0*np.sin(theta)

    previous = 0
    added = False
    for i in range(1,len(Z)):
        if(Y[i]<0 and Y[i-1]>=0):
            if(added==False):
                fig.add_trace(go.Scatter3d(x=X[previous:i], 
                                y=Y[previous:i],
                                z=Z[previous:i],
                                mode='lines',
                                line=dict(color='rgba(255 ,0 ,0 ,1.0)',width=line_thickness_red)
                                ),
                                row=row, col=col,
                        )
                added = True
            else:
                fig.add_trace(go.Scatter3d(x=X[previous:i], 
                                y=Y[previous:i],
                                z=Z[previous:i],
                                mode='lines',
                                line=dict(color='rgba(255 ,0 ,0 ,1.0)',width=line_thickness_red)
                                ),
                                row=row, col=col,
                )
            previous = i
        elif(Y[i]>=0 and Y[i-1]<0):
            if(added==False):
                fig.add_trace(go.Scatter3d(x=X[previous:i], 
                                y=Y[previous:i],
                                z=Z[previous:i],
                                mode='lines',
                                line=dict(color='rgba(255 ,0 ,0 ,0.6)',width=line_thickness_red,dash='dash')
                                ),
                                row=row, col=col,
                        )
            else:
                fig.add_trace(go.Scatter3d(x=X[previous:i], 
                                y=Y[previous:i],
                                z=Z[previous:i],
                                mode='lines',
                                line=dict(color='rgba(255 ,0 ,0 ,0.6)',width=line_thickness_red,dash='dash')
                                ),
                                row=row, col=col,
                )
            previous = i


    if(Y[-1]>=0):
        fig.add_trace(go.Scatter3d(x=X[previous:i], 
                                y=Y[previous:i],
                                z=Z[previous:i],
                                mode='lines',
                                line=dict(color='rgba(255 ,0 ,0 ,1.0)',width=line_thickness_red)
                                ),
                                row=row, col=col,
                )       

    else:
        fig.add_trace(go.Scatter3d(x=X[previous:i], 
                        y=Y[previous:i],
                        z=Z[previous:i],
                        mode='lines',
                        line=dict(color='rgba(255 ,0 ,0 ,0.6)',width=line_thickness_red,dash='dash')
                        ),
                        row=row, col=col,
        )



    ########FIND BETTER WAY TO PUT THE ARROWS MID WAY 
        

    start = 20
    for a in range(0,turns):
        for t in range(start,Y.shape[0]):
            if(np.abs(Y[t]- Y.max()) <0.0003):
                dir0= direction_from_pts([X[t+1:t+2],Y[t+1:t+2],Z[t+1:t+2]],
                            [X[t:t+1],Y[t:t+1],Z[t:t+1]],North_to_south,left_handed)
        
                fig.add_trace( go.Cone(x=X[t:t+1], 
                                    y=Y[t:t+1],
                                    z=Z[t:t+1], 
                                    u=dir0[0], 
                                    v=dir0[1], 
                                    w=dir0[2],
                                    sizemode="absolute",
                                    sizeref=red_arrow_thickness,
                                    colorscale=[[0, 'red'],[1, 'red']],
                                    showscale=False),
                                    row=row, col=col,
                                )
                
                start = t+1
        



    if ((North_to_south and left_handed) or (not North_to_south and not left_handed)):
        direction = direction_from_pts([X[-1],Y[-1],Z[-1]],[X[-2],Y[-2],Z[-2]],North_to_south,left_handed)
        fig.add_trace( go.Cone(x=X[-1:], y=Y[-1:],z=Z[-1:], u=[direction[0]], v=[direction[1]], w=[direction[2]],sizemode="absolute",sizeref=red_arrow_thickness,colorscale=[[0, 'red'],[1, 'red']],showscale=False),
                                row=row, col=col)
        
    else:
        direction = direction_from_pts([X[1],Y[1],Z[1]],[X[0],Y[0],Z[0]],North_to_south,left_handed)
        fig.add_trace( go.Cone(x=[X[0]], y=[Y[0]],z=[Z[0]], u=[direction[0]], v=[direction[1]], w=[direction[2]],sizemode="absolute",sizeref=red_arrow_thickness,colorscale=[[0, 'red'],[1, 'red']],showscale=False),
                                row=row, col=col,)


    if((North_to_south and not left_handed) or (not North_to_south and left_handed)):
        fig.add_trace( go.Scatter3d(x=[0,0], 
                    y=[0,0],
                    z=[-2,turns+2],
                    mode='lines',
                    line=dict(color='black',width=15)
                    ),
                                row=row, col=col,
                )
        fig.add_trace( go.Cone(x=[0], y=[0],z=[-2], u=[0],v=[0], w=[-1],sizemode="absolute",sizeref=black_arrow_thickness,colorscale=[[0, 'black'],[1, 'black']],showscale=False),
                                row=row, col=col,)
        

    else:
   
        fig.add_trace( go.Scatter3d(x=[0,0], 
                    y=[0,0],
                    z=[-1,turns+2],
                    mode='lines',
                    line=dict(color='black',width=15)
                    ),
                    row=row, col=col,
                )
        fig.add_trace( go.Cone(x=[0], y=[0],z=[turns+2], u=[0],v=[0], w=[1],sizemode="absolute",sizeref=black_arrow_thickness,colorscale=[[0, 'black'],[1, 'black']],showscale=False),
                                row=row, col=col,)


    # fig2 =
    camera = dict(
        up=dict(x=1, y=0, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=1.25, z=0)
    )
    # fig.update_layout(scene_camera=camera, title="prout",scene = dict(  xaxis = dict(showgrid = False,showticklabels = False,backgroundcolor ="white"),
    #                                                                     yaxis = dict(showgrid = False,showticklabels = False,backgroundcolor ="white"),
    #                                                                     zaxis = dict(showgrid = False,showticklabels = False,backgroundcolor ="white")
    #                                                                     )
    #             )
    # fig.update_xaxes(showgrid = False,showticklabels = False,backgroundcolor ="white", row=row, col=row)
    # fig.update_yaxes(showgrid = False,showticklabels = False,backgroundcolor ="white", row=row, col=row)
    # fig.update_zaxes(showgrid = False,showticklabels = False,backgroundcolor ="white", row=row, col=row)
    fig.get_subplot(row=row,col=col).camera = camera
    fig.get_subplot(row=row,col=col).xaxis.showgrid = False
    fig.get_subplot(row=row,col=col).xaxis.showticklabels = False
    fig.get_subplot(row=row,col=col).xaxis.backgroundcolor = 'white'
    fig.get_subplot(row=row,col=col).xaxis.title = ''

    fig.get_subplot(row=row,col=col).yaxis.showgrid = False
    fig.get_subplot(row=row,col=col).yaxis.showticklabels = False
    fig.get_subplot(row=row,col=col).yaxis.backgroundcolor = 'white'
    fig.get_subplot(row=row,col=col).yaxis.title = ''


    fig.get_subplot(row=row,col=col).zaxis.showgrid = False
    fig.get_subplot(row=row,col=col).zaxis.showticklabels = False
    fig.get_subplot(row=row,col=col).zaxis.backgroundcolor = 'white'
    fig.get_subplot(row=row,col=col).zaxis.title = ''


    return fig



if __name__ == '__main__':
    fig = make_subplots(4,1,specs =[[{'type': 'scene'}],[{'type': 'scene'}],[{'type': 'scene'}],[{'type': 'scene'}]])
    fig = create_twist(fig,row=1,col=1,North_to_south = True,left_handed= False,resolution = 1000,turns = 10,line_thickness_red=20.0,red_arrow_thickness=1.3,black_arrow_thickness=1.3)
    fig = create_twist(fig,row=2,col=1,North_to_south = True,left_handed= True,resolution = 1000,turns = 10,line_thickness_red=20.0,red_arrow_thickness=1.3,black_arrow_thickness=1.3)
    fig = create_twist(fig,row=3,col=1,North_to_south = False,left_handed= True,resolution = 1000,turns = 10,line_thickness_red=20.0,red_arrow_thickness=1.3,black_arrow_thickness=1.3)
    fig = create_twist(fig,row=4,col=1,North_to_south = False,left_handed= False,resolution = 1000,turns = 10,line_thickness_red=20.0,red_arrow_thickness=1.3,black_arrow_thickness=1.3)
    fig.show() 