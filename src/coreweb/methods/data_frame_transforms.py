import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import itertools

'''
Functions by E.E. Davies, modified for 3DCORE
'''


#input datetime to return T1, T2 and T3 based on Hapgood 1992
#http://www.igpp.ucla.edu/public/vassilis/ESS261/Lecture03/Hapgood_sdarticle.pdf
def get_geocentric_transformation_matrices(time):
    #format dates correctly, calculate MJD, T0, UT 
    ts = pd.Timestamp(time)
    jd=ts.to_julian_date()
    mjd=float(int(jd-2400000.5)) #use modified julian date    
    T0=(mjd-51544.5)/36525.0
    UT=ts.hour + ts.minute / 60. + ts.second / 3600. #time in UT in hours    
    #define position of geomagnetic pole in GEO coordinates
    pgeo=78.8+4.283*((mjd-46066)/365.25)*0.01 #in degrees
    lgeo=289.1-1.413*((mjd-46066)/365.25)*0.01 #in degrees
    #GEO vector
    Qg=[np.cos(pgeo*np.pi/180)*np.cos(lgeo*np.pi/180), np.cos(pgeo*np.pi/180)*np.sin(lgeo*np.pi/180), np.sin(pgeo*np.pi/180)]
    #now move to equation at the end of the section, which goes back to equations 2 and 4:
    #CREATE T1; T0, UT is known from above
    zeta=(100.461+36000.770*T0+15.04107*UT)*np.pi/180
    ################### theta und z
    T1=np.matrix([[np.cos(zeta), np.sin(zeta),  0], [-np.sin(zeta) , np.cos(zeta) , 0], [0,  0,  1]]) #angle for transpose
    LAMBDA=280.460+36000.772*T0+0.04107*UT
    M=357.528+35999.050*T0+0.04107*UT
    lt2=(LAMBDA+(1.915-0.0048*T0)*np.sin(M*np.pi/180)+0.020*np.sin(2*M*np.pi/180))*np.pi/180 #lamda sun
    #CREATE T2, LAMBDA, M, lt2 known from above
    ##################### lamdbda und Z
    t2z=np.matrix([[np.cos(lt2), np.sin(lt2),  0], [-np.sin(lt2) , np.cos(lt2) , 0], [0,  0,  1]])
    et2=(23.439-0.013*T0)*np.pi/180
    ###################### epsilon und x
    t2x=np.matrix([[1,0,0],[0,np.cos(et2), np.sin(et2)], [0, -np.sin(et2), np.cos(et2)]])
    T2=np.dot(t2z,t2x)  #equation 4 in Hapgood 1992
    #matrix multiplications   
    T2T1t=np.dot(T2,np.matrix.transpose(T1))
    ################
    Qe=np.dot(T2T1t,Qg) #Q=T2*T1^-1*Qq
    psigsm=np.arctan(Qe.item(1)/Qe.item(2)) #arctan(ye/ze) in between -pi/2 to +pi/2
    T3=np.matrix([[1,0,0],[0,np.cos(-psigsm), np.sin(-psigsm)], [0, -np.sin(-psigsm), np.cos(-psigsm)]])
    return T1, T2, T3


def get_heliocentric_transformation_matrices(time):
    #format dates correctly, calculate MJD, T0, UT 
    ts = pd.Timestamp(time)
    jd=ts.to_julian_date()
    mjd=float(int(jd-2400000.5)) #use modified julian date    
    T0=(mjd-51544.5)/36525.0
    UT=ts.hour + ts.minute / 60. + ts.second / 3600. #time in UT in hours
    #equation 12
    LAMBDA=280.460+36000.772*T0+0.04107*UT
    M=357.528+35999.050*T0+0.04107*UT
    #lamda sun in radians
    lt2=(LAMBDA+(1.915-0.0048*T0)*np.sin(M*np.pi/180)+0.020*np.sin(2*M*np.pi/180))*np.pi/180
    #S1 matrix
    S1=np.matrix([[np.cos(lt2+np.pi), np.sin(lt2+np.pi),  0], [-np.sin(lt2+np.pi) , np.cos(lt2+np.pi) , 0], [0,  0,  1]])
    #equation 13
    #create S2 matrix with angles with reversed sign for transformation HEEQ to HAE
    iota=7.25*np.pi/180
    omega=(73.6667+0.013958*((mjd+3242)/365.25))*np.pi/180 #in rad         
    theta=np.arctan(np.cos(iota)*np.tan(lt2-omega))
    #quadrant of theta must be opposite lt2 - omega; Hapgood 1992 end of section 5   
    #get lambda-omega angle in degree mod 360   
    lambda_omega_deg=np.mod(lt2-omega,2*np.pi)*180/np.pi
    x = np.cos(np.deg2rad(lambda_omega_deg))
    y = np.sin(np.deg2rad(lambda_omega_deg))
    #get theta_node in deg
    x_theta = np.cos(theta)
    y_theta = np.sin(theta)
    #if in same quadrant, then theta_node = theta_node +pi  
    if (x>=0 and y>=0):
        if (x_theta>=0 and y_theta>=0): theta = theta - np.pi
        elif (x_theta<=0 and y_theta<=0): theta = theta
        elif (x_theta>=0 and y_theta<=0): theta = theta - np.pi/2
        elif (x_theta<=0 and y_theta>=0): theta = np.pi+(theta-np.pi/2)
        
    elif (x<=0 and y<=0):
        if (x_theta>=0 and y_theta>=0): theta = theta
        elif (x_theta<=0 and y_theta<=0): theta = theta + np.pi
        elif (x_theta>=0 and y_theta<=0): theta = theta + np.pi/2
        elif (x_theta<=0 and y_theta>=0): theta = theta-np.pi/2
        
    elif (x>=0 and y<=0):
        if (x_theta>=0 and y_theta>=0): theta = theta + np.pi/2
        elif (x_theta<=0 and y_theta<=0): theta = np.pi+(theta-np.pi/2) 
        elif (x_theta>=0 and y_theta<=0): theta = theta + np.pi
        elif (x_theta<=0 and y_theta>=0): theta = theta

    elif (x<0 and y>0):
        if (x_theta>=0 and y_theta>=0): theta = theta - np.pi/2
        elif (x_theta<=0 and y_theta<=0): theta = theta + np.pi/2
        elif (x_theta>=0 and y_theta<=0): theta = theta
        elif (x_theta<=0 and y_theta>=0): theta = theta -np.pi   

    s2_theta = np.matrix([[np.cos(theta), np.sin(theta),  0], [-np.sin(theta) , np.cos(theta) , 0], [0,  0,  1]])
    s2_iota = np.matrix([[1,  0,  0], [0, np.cos(iota), np.sin(iota)], [0, -np.sin(iota) , np.cos(iota)]])
    s2_omega = np.matrix([[np.cos(omega), np.sin(omega),  0], [-np.sin(omega) , np.cos(omega) , 0], [0,  0,  1]])
    S2 = np.dot(np.dot(s2_theta,s2_iota),s2_omega)

    return S1, S2


def GSE_to_GSM(bx, by, bz, time):
    B_GSM = []
    for i in range(len(bx)):
        T1, T2, T3 = get_geocentric_transformation_matrices(time[i])
        B_GSE_i = np.array([[bx[i]], [by[i]], [bz[i]]]) 
        B_GSM_i = np.dot(T3, B_GSE_i)
        B_GSM_i_list = B_GSM_i.tolist()
        flat_B_GSM_i = list(itertools.chain(*B_GSM_i_list))
        B_GSM.append(flat_B_GSM_i)

    B_GSM = np.array(B_GSM)

    return B_GSM[:,0], B_GSM[:,1], B_GSM[:,2]


def GSM_to_GSE(bx, by, bz, time):
    B_GSE = []
    for i in range(len(bx)):
        T1, T2, T3 = get_geocentric_transformation_matrices(time[i])
        T3_inv = np.linalg.inv(T3)
        B_GSM_i = np.array([[bx[i]], [by[i]], [bz[i]]]) 
        B_GSE_i = np.dot(T3_inv, B_GSM_i)
        B_GSE_i_list = B_GSE_i.tolist()
        flat_B_GSE_i = list(itertools.chain(*B_GSE_i_list))
        B_GSE.append(flat_B_GSE_i)

    B_GSE = np.array(B_GSE)

    return B_GSE[:,0], B_GSE[:,1], B_GSE[:,2]


def GSE_to_RTN_approx(bx, by, bz):
    rt_approx_bx = -bx
    rt_approx_by = -by
    rt_approx_bz = bz

    return rt_approx_bx, rt_approx_by, rt_approx_bz

    
def GSM_to_RTN_approx(x, y, z, bx, by, bz, time):
    gse_bx, gse_by, gse_bz = GSM_to_GSE(bx, by, bz, time)
    rt_approx_bx, rt_approx_by, rt_approx_bz = GSE_to_RTN_approx(gse_bx, gse_by, gse_bz)
    
    return rt_approx_bx, rt_approx_by, rt_approx_bz

def GSE_to_HEE(bx, by, bz):
    hee_bx = -bx
    hee_by = -by
    hee_bz = bz

    return hee_bx, hee_by, hee_bz

def HEE_to_GSE(bx, by, bz):
    gse_bx = -bx
    gse_by = -by
    gse_bz = bz

    return gse_bx, gse_by, gse_bz


def HEE_to_HAE(bx, by, bz, time):
    B_HAE = []
    for i in range(len(bx)):
        S1, S2 = get_heliocentric_transformation_matrices(time[i])
        S1_inv = np.linalg.inv(S1)
        B_HEE_i = np.array([[bx[i]], [by[i]], [bz[i]]]) 
        B_HAE_i = np.dot(S1_inv, B_HEE_i)
        B_HAE_i_list = B_HAE_i.tolist()
        flat_B_HAE_i = list(itertools.chain(*B_HAE_i_list))
        B_HAE.append(flat_B_HAE_i)

    B_HAE = np.array(B_HAE)
    bt = np.linalg.norm(B_HAE, axis=1)

    return B_HAE[:,0], B_HAE[:,1], B_HAE[:,2]


def HAE_to_HEE(bx, by, bz, time):
    B_HEE = []
    for i in range(len(bx)):
        S1, S2 = get_heliocentric_transformation_matrices(time[i])
        B_HAE_i = np.array([[bx[i]], [by[i]], [bz[i]]]) 
        B_HEE_i = np.dot(S1, B_HAE_i)
        B_HEE_i_list = B_HEE_i.tolist()
        flat_B_HEE_i = list(itertools.chain(*B_HEE_i_list))
        B_HEE.append(flat_B_HEE_i)

    B_HEE = np.array(B_HEE)
    bt = np.linalg.norm(B_HEE, axis=1)

    return B_HEE[:,0], B_HEE[:,1], B_HEE[:,2]


def HAE_to_HEEQ(bx, by, bz, time):
    B_HEEQ = []
    for i in range(len(bx)):
        S1, S2 = get_heliocentric_transformation_matrices(time[i])
        B_HAE_i = np.array([[bx[i]], [by[i]], [bz[i]]]) 
        B_HEEQ_i = np.dot(S2, B_HAE_i)
        B_HEEQ_i_list = B_HEEQ_i.tolist()
        flat_B_HEEQ_i = list(itertools.chain(*B_HEEQ_i_list))
        B_HEEQ.append(flat_B_HEEQ_i)

    B_HEEQ = np.array(B_HEEQ)
    bt = np.linalg.norm(B_HEEQ, axis=1)

    return B_HEEQ[:,0], B_HEEQ[:,1], B_HEEQ[:,2]


def HEEQ_to_HAE(bx, by, bz, time):
    B_HAE = []
    for i in range(len(bx)):
        S1, S2 = get_heliocentric_transformation_matrices(time[i])
        S2_inv = np.linalg.inv(S2)
        B_HEEQ_i = np.array([[bx[i]], [by[i]], [bz[i]]]) 
        B_HEA_i = np.dot(S2_inv, B_HEEQ_i)
        B_HAE_i_list = B_HEA_i.tolist()
        flat_B_HAE_i = list(itertools.chain(*B_HAE_i_list))
        B_HAE.append(flat_B_HAE_i)

    B_HAE = np.array(B_HAE)
    bt = np.linalg.norm(B_HAE, axis=1)

    return B_HAE[:,0], B_HAE[:,1], B_HAE[:,2]


def HEE_to_HEEQ(bx, by, bz, time):
    hae_bx, hae_by, hae_bz = HEE_to_HAE(bx, by, bz, time)
    heeq_bx, heeq_by, heeq_bz = HAE_to_HEEQ(hae_bx, hae_by, hae_bz, time)
    
    return heeq_bx, heeq_by, heeq_bz


def HEEQ_to_HEE(bx, by, bz, time):
    hae_bx, hae_by, hae_bz = HEEQ_to_HAE(bx, by, bz, time)
    hee_bx, hee_by, hee_bz = HAE_to_HEE(hae_bx, hae_by, hae_bz, time)
    
    return hee_bx, hee_by, hee_bz

def HEEQ_to_RTN(x, y, z, bx, by, bz):
    # Unit vectors of HEEQ basis
    heeq_x = np.array([1, 0, 0])
    heeq_y = np.array([0, 1, 0])
    heeq_z = np.array([0, 0, 1])
    
    rtn_bx = []
    rtn_by = []
    rtn_bz = []

    for i in range(len(x)):
        # Make unit vectors of RTN in basis of HEEQ
        rtn_r = np.array([x[i], y[i], z[i]]) / np.linalg.norm([x[i], y[i], z[i]])
        rtn_t = np.cross(heeq_z, rtn_r)
        rtn_n = np.cross(rtn_r, rtn_t)

        # Calculate components of B in RTN coordinates
        br_i = bx[i] * np.dot(heeq_x, rtn_r) + by[i] * np.dot(heeq_y, rtn_r) + bz[i] * np.dot(heeq_z, rtn_r)
        bt_i = bx[i] * np.dot(heeq_x, rtn_t) + by[i] * np.dot(heeq_y, rtn_t) + bz[i] * np.dot(heeq_z, rtn_t)
        bn_i = bx[i] * np.dot(heeq_x, rtn_n) + by[i] * np.dot(heeq_y, rtn_n) + bz[i] * np.dot(heeq_z, rtn_n)
        
        rtn_bx.append(br_i)
        rtn_by.append(bt_i)
        rtn_bz.append(bn_i)

    return rtn_bx, rtn_by, rtn_bz

def RTN_to_HEEQ(x, y, z, bx, by, bz):
    # HEEQ unit vectors (same as spacecraft xyz position)
    heeq_x = np.array([1, 0, 0])
    heeq_y = np.array([0, 1, 0])
    heeq_z = np.array([0, 0, 1])
    
    B_HEEQ = []
    for i in range(len(x)):
        # Make normalized RTN unit vectors from spacecraft position in HEEQ basis
        rtn_x = np.array([x[i], y[i], z[i]]) / np.linalg.norm([x[i], y[i], z[i]])
        rtn_y = np.cross(heeq_z, rtn_x) / np.linalg.norm(np.cross(heeq_z, rtn_x))
        rtn_z = np.cross(rtn_x, rtn_y) / np.linalg.norm(np.cross(rtn_x, rtn_y))
        
        # Project into new system (HEEQ)
        bx_i = np.dot(np.dot(bx[i], rtn_x) + np.dot(by[i], rtn_y) + np.dot(bz[i], rtn_z), heeq_x)
        by_i = np.dot(np.dot(bx[i], rtn_x) + np.dot(by[i], rtn_y) + np.dot(bz[i], rtn_z), heeq_y)
        bz_i = np.dot(np.dot(bx[i], rtn_x) + np.dot(by[i], rtn_y) + np.dot(bz[i], rtn_z), heeq_z)
        
        B_HEEQ.append([bx_i, by_i, bz_i])
    
    B_HEEQ = np.array(B_HEEQ)
    heeq_bx = B_HEEQ[:, 0]
    heeq_by = B_HEEQ[:, 1]
    heeq_bz = B_HEEQ[:, 2]
    
    return heeq_bx, heeq_by, heeq_bz


def RTN_to_HAE(x, y, z, bx, by, bz, time):
    heeq_bx, heeq_by, heeq_bz = RTN_to_HEEQ(x, y, z, bx, by, bz)
    hae_bx, hae_by, hae_bz = HEEQ_to_HAE(heeq_bx, heeq_by, heeq_bz, time)
    
    return hae_bx, hae_by, hae_bz

def RTN_to_HEE(x, y, z, bx, by, bz, time):
    hae_bx, hae_by, hae_bz = RTN_to_HAE(x, y, z, bx, by, bz, time)
    hee_bx, hee_by, hee_bz = HAE_to_HEE(hae_bx, hae_by, hae_bz, time)
    
    return hee_bx, hee_by, hee_bz

def RTN_to_GSE(x, y, z, bx, by, bz, time):
    hee_bx, hee_by, hee_bz = RTN_to_HEE(x, y, z, bx, by, bz, time)
    gse_bx, gse_by, gse_bz = HEE_to_GSE(hee_bx, hee_by, hee_bz)
    
    return gse_bx, gse_by, gse_bz

def RTN_to_GSM(x, y, z, bx, by, bz, time):
    gse_bx, gse_by, gse_bz = RTN_to_GSE(x, y, z, bx, by, bz, time)
    gsm_bx, gsm_by, gsm_bz = GSE_to_GSM(gse_bx, gse_by, gse_bz, time)
    
    return gsm_bx, gsm_by, gsm_bz