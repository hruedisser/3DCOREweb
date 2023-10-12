
import time
import datetime
import coreweb
import os
from ...methods.method import BaseMethod


def extract_fitvars(fittingstate_values):
    
    t_launch = datetime.datetime.strptime(fittingstate_values[0], 'Launch Time: %Y-%m-%d %H:%M')
    model_kwargs = get_modelkwargs(fittingstate_values)
    
    try:
        fitobserver, fit_coord_system = get_fitobserver(fittingstate_values[16], fittingstate_values[1][0]['spacecraft'])
    except:
        print('Observer not known!')
        return
    
    iter_i = 0 # keeps track of iterations
    hist_eps = [] # keeps track of epsilon values
    hist_time = [] # keeps track of time
    
    t_s, t_e, t_fit = extract_t(fittingstate_values[1][0])

    if fittingstate_values[17] == 'abc-smc':
        base_fitter = BaseMethod()
        base_fitter.initialize(t_launch, coreweb.ToroidalModel, model_kwargs)
        base_fitter.add_observer(fitobserver, t_fit, t_s, t_e)
        
    njobs = fittingstate_values[18]
    
    if fittingstate_values[19] == ['multiprocessing']:
        multiprocessing = True
    else:
        multiprocessing = False
        
    itermin = fittingstate_values[20][0]
    itermax = fittingstate_values[20][1]
    
    if fittingstate_values[15] == 0:
        n_particles = 265
    elif fittingstate_values[15] == 1:
        n_particles = 512
    elif fittingstate_values[15] == 2:
        n_particles = 1024
    elif fittingstate_values[15] == 3:
        n_particles = 2048
    
    outputfile = fittingstate_values[22]['id'][0]+'_HEEQ'#+ fittingstate_values[16]#[0]
    
    #current_time = str(int(time.time()))
    
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Format the datetime as a string
    current_time = current_datetime.strftime("%Y%m%d%H%M")
    outputpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output/"))
    
    if isinstance(outputfile, str):
        outputfilecode = outputpath +'/' + outputfile + "_" + current_time + "/"
    elif isinstance(outputfile, list) and len(outputfile) == 1 and isinstance(outputfile[0], str):
        outputfilecode = outputpath + outputfile[0] + "_" + current_time + "/"
    
    #print(outputfilecode)
    
    return base_fitter, fit_coord_system, multiprocessing, itermin, itermax, n_particles, outputfilecode, njobs, model_kwargs, t_launch

def extract_t(row_data):
    refa = datetime.datetime.strptime(row_data['ref_a'], '%Y-%m-%d %H:%M')
    refb = datetime.datetime.strptime(row_data['ref_b'], '%Y-%m-%d %H:%M')
    
    keys = list(row_data.keys())
    
    t_fit = [datetime.datetime.strptime(row_data[t], '%Y-%m-%d %H:%M') for t in keys[3:] if row_data[t] != '']
    
    return refa, refb, t_fit 

def get_fitobserver(mag_coord_system, sc):
    
    if mag_coord_system == 'HEEQ':
        reference_frame = 'HEEQ'
        
    if sc == 'BepiColombo':
        if mag_coord_system == 'RTN':
            reference_frame = 'GSE'
        observer = 'Bepi'
    elif (sc == 'DSCOVR'):
        if mag_coord_system == 'RTN':
            reference_frame = 'GSE'
        observer = 'DSCOVR'
    elif sc == 'MESSENGER':
        if mag_coord_system == 'RTN':
            reference_frame = 'MSGR_RTN'
        observer = 'Mes'
    elif sc == 'PSP':
        if mag_coord_system == 'RTN':
            reference_frame = 'SPP_RTN'
        observer = 'PSP'
    elif (sc == 'SolarOrbiter') or (sc == 'SOLO'):
        if mag_coord_system == 'RTN':
            reference_frame = 'SOLO_SUN_RTN'
        observer = 'SOLO'
    elif (sc == 'STEREO-A') or (sc == "STEREO-A_beacon"):
        if mag_coord_system == 'RTN':
            reference_frame = 'STAHGRTN'
        observer = 'STA'
    elif sc == 'VEX-A':
        if mag_coord_system == 'RTN':
            reference_frame = 'VSO'
        observer = 'VEX'
    elif (sc == 'Wind'):
        if mag_coord_system == 'RTN':
            reference_frame = 'GSE'
        observer = 'Wind'
        
    elif (sc == "NOAA_RTSW"):
        if mag_coord_system == 'RTN':
            reference_frame = 'GSE'
        observer = 'NOAA_RTSW'
        
        
    return observer, reference_frame

def get_modelkwargs(fittingstate_values):
    
    if fittingstate_values[21] == 16:
        ensemble_size = int(2**16)
    elif fittingstate_values[21] == 17:
        ensemble_size = int(2**17)
    elif fittingstate_values[21] == 18:
        ensemble_size = int(2**18)
        
    model_kwargs = {
        "ensemble_size": ensemble_size, #2**17
        "iparams": {
            "cme_longitude": {
                "maximum": fittingstate_values[2][1],
                "minimum": fittingstate_values[2][0]
            },
            "cme_latitude": {
                "maximum": fittingstate_values[3][1],
                "minimum": fittingstate_values[3][0]
            },
            "cme_inclination": {
                "distribution": "uniform",
                "maximum": fittingstate_values[4][1],
                "minimum": fittingstate_values[4][0]
            },
            "cme_diameter_1au": {
                "maximum": fittingstate_values[5][1],
                "minimum": fittingstate_values[5][0]
            },
            "cme_aspect_ratio": {
                "maximum": fittingstate_values[6][1],
                "minimum": fittingstate_values[6][0]
            },
            "cme_launch_radius": {
                "distribution": "uniform",
                "maximum": fittingstate_values[7][1],
                "minimum": fittingstate_values[7][0]
            },
            "cme_launch_velocity": {
                "maximum": fittingstate_values[8][1],
                "minimum": fittingstate_values[8][0]
            },
            "t_factor": {
                "maximum": fittingstate_values[12][1],
                "minimum": fittingstate_values[12][0],
            },
            "cme_expansion_rate": {
                "distribution": "uniform",
                "maximum": fittingstate_values[9][1],
                "minimum": fittingstate_values[9][0],
            },
            "magnetic_decay_rate": {
                "distribution": "uniform",
                "maximum": fittingstate_values[13][1],
                "minimum": fittingstate_values[13][0],
            },
            "magnetic_field_strength_1au": {
                "maximum": fittingstate_values[14][1],
                "minimum": fittingstate_values[14][0],
            },
            "background_drag": {
                "distribution": "uniform",
                "maximum": fittingstate_values[10][1],
                "minimum": fittingstate_values[10][0],
            },
            "background_velocity": {
                "distribution": "uniform",
                "maximum": fittingstate_values[11][1],
                "minimum": fittingstate_values[11][0],
            }
        }
    }
    
    for param, values in model_kwargs["iparams"].items():
        if values["maximum"] == values["minimum"]:
            values["distribution"] = "fixed"
            values["default_value"] = values["minimum"]
            del values["maximum"]
            del values["minimum"]
    
    return model_kwargs