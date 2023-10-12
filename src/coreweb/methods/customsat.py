# -*- coding: utf-8 -*-

from typing import Any, List, Optional, Sequence, Tuple, Union

import heliosat
import numba
import numpy as np
import os
import glob

from scipy.signal import detrend, welch
######
import matplotlib.dates as mdates
from matplotlib.dates import  DateFormatter
import datetime
from datetime import timedelta

from sunpy.coordinates import frames, get_horizons_coord, get_body_heliographic_stonyhurst
from sunpy.time import parse_time

import urllib.request
import json
import pandas as pds
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time

import itertools

import cdflib
import pickle

import logging

logger = logging.getLogger(__name__)


############## CUSTOM DATA

#input datetime to return T1, T2 and T3 based on Hapgood 1992
#http://www.igpp.ucla.edu/public/vassilis/ESS261/Lecture03/Hapgood_sdarticle.pdf
def get_transformation_matrices(timestamp):
    #format dates correctly, calculate MJD, T0, UT 
    ts = pds.Timestamp(timestamp)
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
    lt2=(LAMBDA+(1.915-0.0048*T0)*np.sin(M*np.pi/180)+0.020*np.sin(2*M*np.pi/180))*np.pi/180
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

def delete_old_files(folder_path='src/py3dcore/realtimedata/', days_threshold=5):
    # Get a list of all files in the folder matching the pattern
    file_pattern = os.path.join(folder_path, 'dscvrrealtime_*.p')
    files = glob.glob(file_pattern)
    # Get the current date
    current_date = datetime.datetime.now().date()
    
    # Check if any file exists
    if not files:
        logger.info("No files found. Downloading realtime data.")
        data = loadrealtime()
        return data
    
    # Iterate over the files
    for file in files:
        # Extract the date from the file name
        file_name = os.path.basename(file)
        date_str = file_name.split('_')[1].split('.')[0]  # Extracting date from the file name
        file_date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
        
        # Check if the file is older than the threshold
        if (current_date - file_date).days > days_threshold:
            # Delete the file
            os.remove(file)
            logger.info(f"Deleted old file: {file}")
            logger.info("Downloading realtime data")
            data = loadrealtime()
        else:
            # Load the file
            with open(file, 'rb') as f:
                data = pickle.load(f)
                # Process the loaded data as needed
                logger.info(f"Loaded file: {file}")
   
    return data
    
def getrealtime(dt,
                reference_frame,
                **kwargs):
    
    """
    Load and preprocess realtime data
    """
    
    logger.info("Checking for up-to-date realtime data")
    
    data = delete_old_files()
    
    sampling_freq = kwargs.pop("sampling_freq", 60)

    if kwargs.pop("as_endpoints", False):
        _ = np.linspace(dt[0].timestamp(), dt[-1].timestamp(), int((dt[-1].timestamp() - dt[0].timestamp()) // sampling_freq))  
        dt = [datetime.datetime.fromtimestamp(_, datetime.timezone.utc) for _ in _]

    dat = []
    tt = [x.replace(tzinfo=None,second=0, microsecond=0) for x in dt]

    ii = []
    for x in tt:
        idx = np.argmin(np.abs(data["time"] - x))
        ii.append(idx)
    
    if reference_frame == "HEEQ":
        for t in ii:
            res = [data[com][t] for com in ['bx_heeq','by_heeq','bz_heeq']]
            dat.append((res))
            
    if reference_frame == "GSE":
        for t in ii:
            res = [data[com][t] for com in ['bx_gse','by_gse','bz_gse']]
            dat.append((res))
            
    if reference_frame == "GSM":
        for t in ii:
            res = [data[com][t] for com in ['bx_gsm','by_gsm','bz_gsm']]
            dat.append((res))

    return np.array(dt), np.array(dat)


def getsunpyposition(dt,
                     reference_frame,
                     observer):
    """
    Get Position in case HelioSat fails"""
    
    try:
        if observer == "WIND":
            coord = get_horizons_coord('EM-L1', dt)
        
        else:
            coord = get_horizons_coord(observer,dt)
    except:
        coord = getapproximate(observer, dt, reference_frame)
        return coord
        
    if reference_frame == 'HEEQ':
        heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        r = heeq.radius.value
        lon = heeq.lon.value
        lat = heeq.lat.value
        
        x,y,z = sphere2cart(r,lon,lat)
        
        combined = np.column_stack((x,y,z))
        
        logger.info("Using sunpy to obtain trajectory")
        
        return combined
    else:
        raise NotImplementedError('The frame was not recognized as a known reference frame.')
    
    return trajectory

def getapproximate(observer, dtp, reference_frame):
    """
    get approximate position
    """
    logger.info("Using approximate spacecraft position")
    
    
    raise NotImplementedError('Approximate position not yet implemented')
    return position

def to_datetime(date):
    """
    Converts a numpy datetime64 object to a python datetime object 
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.utcfromtimestamp(timestamp)

def loadrealtime():
    """
    Download and transform realtime data
    """

    request_mag=urllib.request.urlopen('https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json')
    file_mag = request_mag.read()
    data_mag = json.loads(file_mag)
    noaa_mag_gsm = pds.DataFrame(data_mag[1:], columns=['timestamp', 'b_x', 'b_y', 'b_z', 'lon_gsm', 'lat_gsm', 'b_tot'])

    # Convert timestamp to datetime and magnetic field components to float
    noaa_mag_gsm['timestamp'] = pds.to_datetime(noaa_mag_gsm['timestamp'].values)
    noaa_mag_gsm['b_x'] = noaa_mag_gsm['b_x'].astype('float')
    noaa_mag_gsm['b_y'] = noaa_mag_gsm['b_y'].astype('float')
    noaa_mag_gsm['b_z'] = noaa_mag_gsm['b_z'].astype('float')
    noaa_mag_gsm['b_tot'] = noaa_mag_gsm['b_tot'].astype('float')
    
    noaa_mag = pds.DataFrame()
    noaa_mag['time'] = noaa_mag_gsm['timestamp']
    noaa_mag['bx_gsm'] = noaa_mag_gsm['b_x']
    noaa_mag['by_gsm'] = noaa_mag_gsm['b_y']
    noaa_mag['bz_gsm'] = noaa_mag_gsm['b_z']

    # Transform magnetic field components from GSM to GSE
    noaa_mag = add_GSE(noaa_mag)
    
    # Transform magnetic field components from GSM to GSE
    noaa_mag = add_HEEQ(noaa_mag)
    
    print(noaa_mag)
    
    beginstr = pds.to_datetime(str(noaa_mag['time'].values[0])).strftime('%Y%m%d')
    endstr = pds.to_datetime(str(noaa_mag['time'].values[-1])).strftime('%Y%m%d')
    
    filename= 'dscvrrealtime_'+endstr+'.p'

    logger.info("Created pickle file from realtime data: %s", filename)
    pickle.dump(noaa_mag, open('src/py3dcore/realtimedata/' + filename, "wb"))
    
    
    return noaa_mag

def add_GSE(df):
    
    """
    Add transformed mag.
    """
    logger.info("Transforming data to GSE")
    
    B_GSE = []
    for i in range(df.shape[0]):
        T1, T2, T3 = get_transformation_matrices(df['time'].iloc[i])
        T3_inv = np.linalg.inv(T3)
        B_GSM_i = np.matrix([[df['bx_gsm'].iloc[i]],[df['by_gsm'].iloc[i]],[df['bz_gsm'].iloc[i]]]) 
        B_GSE_i = np.dot(T3_inv,B_GSM_i)
        B_GSE_i_list = B_GSE_i.tolist()
        flat_B_GSE_i = list(itertools.chain(*B_GSE_i_list))
        B_GSE.append(flat_B_GSE_i)
    df_transformed = pds.DataFrame(B_GSE, columns=['b_x', 'b_y', 'b_z'])
    df['bx_gse'] = df_transformed['b_x']
    df['by_gse'] = df_transformed['b_y']
    df['bz_gse'] = df_transformed['b_z']
    
    return df
    
    
def add_HEEQ(df):
    
    """
    Add transformed mag.
    """
    logger.info("Transforming data to HEEQ")
    
    jd = np.zeros(len(df))
    mjd = np.zeros(len(df))
    
    for i in range(len(df)):
        jd[i] = Time(df['time'][i]).jd
        mjd[i] = float(int(jd[i] - 2400000.5))  # use modified Julian date    

        # GSE to HEE
        #Hapgood 1992 rotation by 180 degrees, or simply change sign in bx by 
        b_hee = [-df['bx_gse'][i], -df['by_gse'][i], df['bz_gse'][i]]
        
        #HEE to HAE        
        
        # HEE to HAE
        T00 = (mjd[i] - 51544.5) / 36525.0
        dobj = df['time'][i].to_pydatetime()
        UT = dobj.hour + dobj.minute / 60. + dobj.second / 3600.  # time in UT in hours   
        
        M = np.radians(357.528 + 35999.050 * T00 + 0.04107 * UT)
        LAMBDA = 280.460 + 36000.772 * T00 + 0.04107 * UT
        lambda_sun = np.radians((LAMBDA + (1.915 - 0.0048 * T00) * np.sin(M) + 0.020 * np.sin(2 * M)))
        
        c, s = np.cos(-(lambda_sun + np.radians(180))), np.sin(-(lambda_sun + np.radians(180)))
        Sm1 = np.array(((c, s, 0), (-s, c, 0), (0, 0, 1)))
        b_hae = np.dot(Sm1, b_hee)

        # HAE to HEEQ
        iota = np.radians(7.25)
        omega = np.radians((73.6667 + 0.013958 * ((mjd[i] + 3242) / 365.25)))
        theta = np.arctan(np.cos(iota) * np.tan(lambda_sun - omega))

        lambda_omega_deg = np.mod(np.degrees(lambda_sun) - np.degrees(omega), 360)
        theta_node_deg = np.degrees(theta)

        if np.logical_or(abs(lambda_omega_deg - theta_node_deg) < 1, abs(lambda_omega_deg - 360 - theta_node_deg) < 1):
            theta = theta + np.pi

        c, s = np.cos(theta), np.sin(theta)
        S2_1 = np.array(((c, s, 0), (-s, c, 0), (0, 0, 1)))

        iota = np.radians(7.25)
        c, s = np.cos(iota), np.sin(iota)
        S2_2 = np.array(((1, 0, 0), (0, c, s), (0, -s, c)))

        c, s = np.cos(omega), np.sin(omega)
        S2_3 = np.array(((c, s, 0), (-s, c, 0), (0, 0, 1)))

        [bx_heeq, by_heeq, bz_heeq] = np.dot(np.dot(np.dot(S2_1, S2_2), S2_3), b_hae)

        df.loc[i, 'bx_heeq'] = bx_heeq
        df.loc[i, 'by_heeq'] = by_heeq
        df.loc[i, 'bz_heeq'] = bz_heeq
        
    return df


def sphere2cart(r,lon,lat):
        
    x = r * np.cos(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))
    y = r * np.sin(np.deg2rad(lon)) * np.cos(np.deg2rad(lat))
    z = r * np.sin(np.deg2rad(lat))
    
    return x,y,z