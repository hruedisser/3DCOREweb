import numpy as np
import copy
    
def convert_RTN_to_HEEQ_mag(x, y, z, bx, by, bz):
    #print('conversion RTN to HEEQ') 
    
    heeq_bx = np.zeros(len(x))
    heeq_by = np.zeros(len(x))
    heeq_bz = np.zeros(len(x))
    
    ########## first RTN to HEEQ 
    
    # go through all data points
    for i in range(len(x)):
        # HEEQ vectors
        X_heeq = [1, 0, 0]
        Y_heeq = [0, 1, 0]
        Z_heeq = [0, 0, 1]

        # normalized X RTN vector
        Xrtn = [x[i], y[i], z[i]] / np.linalg.norm([x[i], y[i], z[i]])
        # solar rotation axis at 0, 0, 1 in HEEQ
        Yrtn = np.cross(Z_heeq, Xrtn) / np.linalg.norm(np.cross(Z_heeq, Xrtn))
        Zrtn = np.cross(Xrtn, Yrtn) / np.linalg.norm(np.cross(Xrtn, Yrtn))
        
        # project into new system
        heeq_bx[i] = np.dot(np.dot(bx[i], Xrtn) + np.dot(by[i], Yrtn) + np.dot(bz[i], Zrtn), X_heeq)
        heeq_by[i] = np.dot(np.dot(bx[i], Xrtn) + np.dot(by[i], Yrtn) + np.dot(bz[i], Zrtn), Y_heeq)
        heeq_bz[i] = np.dot(np.dot(bx[i], Xrtn) + np.dot(by[i], Yrtn) + np.dot(bz[i], Zrtn), Z_heeq)
        
    return heeq_bx, heeq_by, heeq_bz


def convert_HEEQ_to_RTN_mag(x,y,z,bx,by,bz, printer = True):
    '''
    for all spacecraft
    '''

    #if printer == True:
        #print('conversion HEEQ to RTN')            
    
    rtn_bx = np.zeros(len(x))
    rtn_by = np.zeros(len(x))
    rtn_bz = np.zeros(len(x))

    #unit vectors of HEEQ basis
    heeq_x=[1,0,0]
    heeq_y=[0,1,0]
    heeq_z=[0,0,1]

    #project into new system RTN
    for i in np.arange(0,len(x)):

        #make unit vectors of RTN in basis of HEEQ
        rtn_r=[x[i],y[i],z[i]]/np.linalg.norm([x[i],y[i],z[i]])
        rtn_t=np.cross(heeq_z,rtn_r) / np.linalg.norm(np.cross(heeq_z,rtn_r))
        rtn_n=np.cross(rtn_r,rtn_t) / np.linalg.norm(np.cross(rtn_r,rtn_t))

        rtn_bx[i]=bx[i]*np.dot(heeq_x,rtn_r)+by[i]*np.dot(heeq_y,rtn_r)+bz[i]*np.dot(heeq_z,rtn_r)
        rtn_by[i]=bx[i]*np.dot(heeq_x,rtn_t)+by[i]*np.dot(heeq_y,rtn_t)+bz[i]*np.dot(heeq_z,rtn_t)
        rtn_bz[i]=bx[i]*np.dot(heeq_x,rtn_n)+by[i]*np.dot(heeq_y,rtn_n)+bz[i]*np.dot(heeq_z,rtn_n)                               

    return rtn_bx, rtn_by, rtn_bz


def separate_components(input_list):
    first_components = []
    second_components = []
    third_components = []

    for sublist in input_list:
        if len(sublist) >= 3:
            first_components.append(sublist[0])
            second_components.append(sublist[1])
            third_components.append(sublist[2])
    
    return first_components, second_components, third_components


def combine_components(first_components, second_components, third_components):
    combined_list = []

    # Determine the minimum length among the three input lists
    min_length = min(len(first_components), len(second_components), len(third_components))

    # Combine the components into sublists
    for i in range(min_length):
        combined_list.append([first_components[i], second_components[i], third_components[i]])

    return combined_list