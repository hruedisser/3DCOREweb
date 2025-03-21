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
    # Format dates correctly, calculate MJD, T0, UT 
    ts = pd.Timestamp(time)
    jd = ts.to_julian_date()
    mjd = float(int(jd - 2400000.5))  # Use modified Julian date    
    T0 = (mjd - 51544.5) / 36525.0
    UT = ts.hour + ts.minute / 60.0 + ts.second / 3600.0  # Time in UT in hours    

    # Define position of geomagnetic pole in GEO coordinates
    pgeo = np.deg2rad(78.8 + 4.283 * ((mjd - 46066) / 365.25) * 0.01)  # Convert to radians
    lgeo = np.deg2rad(289.1 - 1.413 * ((mjd - 46066) / 365.25) * 0.01)  # Convert to radians

    # GEO vector
    Qg = np.array([np.cos(pgeo) * np.cos(lgeo), np.cos(pgeo) * np.sin(lgeo), np.sin(pgeo)])

    # CREATE T1
    zeta = np.deg2rad(100.461 + 36000.770 * T0 + 15.04107 * UT)
    cos_z, sin_z = np.cos(zeta), np.sin(zeta)
    T1 = np.array([[cos_z, sin_z, 0], [-sin_z, cos_z, 0], [0, 0, 1]])

    # CREATE T2
    LAMBDA = 280.460 + 36000.772 * T0 + 0.04107 * UT
    M = 357.528 + 35999.050 * T0 + 0.04107 * UT
    M_rad = np.deg2rad(M)

    lt2 = np.deg2rad(LAMBDA + (1.915 - 0.0048 * T0) * np.sin(M_rad) + 0.020 * np.sin(2 * M_rad))
    cos_lt2, sin_lt2 = np.cos(lt2), np.sin(lt2)
    t2z = np.array([[cos_lt2, sin_lt2, 0], [-sin_lt2, cos_lt2, 0], [0, 0, 1]])

    et2 = np.deg2rad(23.439 - 0.013 * T0)
    cos_e, sin_e = np.cos(et2), np.sin(et2)
    t2x = np.array([[1, 0, 0], [0, cos_e, sin_e], [0, -sin_e, cos_e]])

    T2 = t2z @ t2x  # Matrix multiplication

    # Compute Qe
    T2T1t = T2 @ T1.T
    Qe = T2T1t @ Qg
    psigsm = np.arctan2(Qe[1], Qe[2])  # Use arctan2 for better numerical stability

    # CREATE T3
    cos_psigsm, sin_psigsm = np.cos(-psigsm), np.sin(-psigsm)
    T3 = np.array([[1, 0, 0], [0, cos_psigsm, sin_psigsm], [0, -sin_psigsm, cos_psigsm]])

    return T1, T2, T3


def get_heliocentric_transformation_matrices(time):
    # Convert timestamp and compute Julian & Modified Julian Date
    ts = pd.Timestamp(time)
    jd = ts.to_julian_date()
    mjd = int(jd - 2400000.5)  # Modified Julian Date
    T0 = (mjd - 51544.5) / 36525.0
    UT = ts.hour + ts.minute / 60.0 + ts.second / 3600.0  # UT in hours

    # Precompute constants and use numpy operations for efficiency
    deg_to_rad = np.pi / 180
    LAMBDA = 280.460 + 36000.772 * T0 + 0.04107 * UT
    M = 357.528 + 35999.050 * T0 + 0.04107 * UT

    # Compute λ_sun in radians directly
    M_rad = M * deg_to_rad
    lt2 = (LAMBDA + (1.915 - 0.0048 * T0) * np.sin(M_rad) + 0.020 * np.sin(2 * M_rad)) * deg_to_rad

    # Compute S1 transformation matrix using direct numpy operations
    lt2_pi = lt2 + np.pi
    cos_lt2, sin_lt2 = np.cos(lt2_pi), np.sin(lt2_pi)
    S1 = np.array([[cos_lt2, sin_lt2, 0], 
                   [-sin_lt2, cos_lt2, 0], 
                   [0, 0, 1]])
    # Equation 13 calculations
    iota = 7.25 * deg_to_rad
    omega = (73.6667 + 0.013958 * ((mjd + 3242) / 365.25)) * deg_to_rad  # in radians
    lambda_omega = lt2 - omega
    theta = np.arctan(np.cos(iota) * np.tan(lambda_omega))

    # Compute the quadrant of theta using vectorized numpy calculations
    lambda_omega_deg = np.mod(lambda_omega, 2 * np.pi) * (180 / np.pi)
    x, y = np.cos(np.radians(lambda_omega_deg)), np.sin(np.radians(lambda_omega_deg))
    x_theta, y_theta = np.cos(theta), np.sin(theta)

    #get theta_node in deg
    x_theta = np.cos(theta)
    y_theta = np.sin(theta)

    if (x>=0 and y>=0):
        if (x_theta>=0 and y_theta>=0): theta = theta - np.pi
        elif (x_theta>=0 and y_theta<=0): theta = theta - np.pi/2
        elif (x_theta<=0 and y_theta>=0): theta = theta + np.pi/2
        
    elif (x<=0 and y<=0):
        if (x_theta<=0 and y_theta<=0): theta = theta - np.pi
        elif (x_theta>=0 and y_theta<=0): theta = theta + np.pi/2
        elif (x_theta<=0 and y_theta>=0): theta = theta - np.pi/2
        
    elif (x>=0 and y<=0):
        if (x_theta>=0 and y_theta>=0): theta = theta + np.pi/2
        elif (x_theta<=0 and y_theta<=0): theta = theta-np.pi/2 
        elif (x_theta>=0 and y_theta<=0): theta = theta - np.pi

    elif (x<0 and y>0):
        if (x_theta>=0 and y_theta>=0): theta = theta - np.pi/2
        elif (x_theta<=0 and y_theta<=0): theta = theta + np.pi/2
        elif (x_theta<=0 and y_theta>=0): theta = theta - np.pi   
    
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    cosiota = np.cos(iota)
    siniota = np.sin(iota)
    cosomega = np.cos(omega)
    sinomega = np.sin(omega)

    s2_theta = np.array([[costheta, sintheta,  0], [-sintheta , costheta, 0], [0,  0,  1]])
    s2_iota = np.array([[1,  0,  0], [0, cosiota, siniota], [0, -siniota , cosiota]])
    s2_omega = np.array([[cosomega, sinomega,  0], [-sinomega , cosomega, 0], [0,  0,  1]])
    S2 = s2_theta @ s2_iota @ s2_omega

    return S1, S2



def GSE_to_GSM(bx, by, bz, time):
    #print(f"Converting GSE to GSM for {len(bx)} data points")

    # Convert bx, by, bz into a single 2D NumPy array (shape: N x 3)
    B_GSE = np.vstack((bx, by, bz)).T  

    # Precompute all transformation matrices T3 (shape: N x 3 x 3)
    transformation_matrices = np.array([get_geocentric_transformation_matrices(t)[2] for t in time])

    # Apply the transformation in a vectorized way
    B_GSM = np.einsum('ijk,ik->ij', transformation_matrices, B_GSE)

    return B_GSM[:, 0], B_GSM[:, 1], B_GSM[:, 2]



def GSM_to_GSE(bx, by, bz, time):
    """
    Convert GSM to GSE coordinates using geocentric transformation matrices.

    Parameters:
    bx, by, bz : array-like
        Components of the magnetic field in GSM coordinates.
    time : array-like
        Corresponding timestamps for transformation.

    Returns:
    bx_gse, by_gse, bz_gse : array-like
        Components of the magnetic field in GSE coordinates.
    """
    #print(f"Converting GSM to GSE for {len(bx)} data points")

    B_GSM = np.vstack((bx, by, bz)).T  # Shape (N, 3)

    # Get all transformation matrices in one go
    T3_matrices = np.array([get_geocentric_transformation_matrices(t)[2] for t in time])
    
    # Compute the inverse transformation matrices
    T3_inv_matrices = np.linalg.inv(T3_matrices)

    # Apply the transformation using batch matrix-vector multiplication
    B_GSE = np.einsum('ijk,ik->ij', T3_inv_matrices, B_GSM)

    return B_GSE[:, 0], B_GSE[:, 1], B_GSE[:, 2]


def GSE_to_RTN_approx(bx, by, bz):

    #print(f"Converting GSE to RTN approx for {len(bx)} data points")

    rt_approx_bx = -bx
    rt_approx_by = -by
    rt_approx_bz = bz

    return rt_approx_bx, rt_approx_by, rt_approx_bz


def GSM_to_RTN_approx(x, y, z, bx, by, bz, time):

    #print(f"Converting GSM to RTN approx for {len(bx)} data points")

    gse_bx, gse_by, gse_bz = GSM_to_GSE(bx, by, bz, time)
    rt_approx_bx, rt_approx_by, rt_approx_bz = GSE_to_RTN_approx(gse_bx, gse_by, gse_bz)

    return rt_approx_bx, rt_approx_by, rt_approx_bz


def GSE_to_HEE(bx, by, bz):

    #print(f"Converting GSE to HEE for {len(bx)} data points")

    hee_bx = -bx
    hee_by = -by
    hee_bz = bz

    return hee_bx, hee_by, hee_bz


def HEE_to_GSE(bx, by, bz):

    #print(f"Converting HEE to GSE for {len(bx)} data points")

    gse_bx = -bx
    gse_by = -by
    gse_bz = bz

    return gse_bx, gse_by, gse_bz


def HEE_to_HAE(bx, by, bz, time):
    #print(f"Converting HEE to HAE for {len(bx)} data points")

    # Convert bx, by, bz into a single 2D NumPy array (shape: N x 3)
    B_HEE = np.vstack((bx, by, bz)).T

    # Precompute all inverse transformation matrices
    transformation_matrices = np.array([np.linalg.inv(get_heliocentric_transformation_matrices(t)[0]) for t in time])

    # Apply the transformation using vectorized operations
    B_HAE = np.einsum('ijk,ik->ij', transformation_matrices, B_HEE)

    return B_HAE[:, 0], B_HAE[:, 1], B_HAE[:, 2]


def HAE_to_HEE(bx, by, bz, time):

    #print(f"Converting HAE to HEE for {len(bx)} data points")

    # Convert bx, by, bz into a single 2D NumPy array (shape: N x 3)
    B_HAE = np.vstack((bx, by, bz)).T  

    # Precompute all transformation matrices S1 (shape: N x 3 x 3)
    transformation_matrices = np.array([get_heliocentric_transformation_matrices(t)[0] for t in time])

    # Apply the transformation in a vectorized way
    B_HEE = np.einsum('ijk,ik->ij', transformation_matrices, B_HAE)

    return B_HEE[:, 0], B_HEE[:, 1], B_HEE[:, 2]


def HAE_to_HEEQ(bx, by, bz, time):
    #print(f"Converting HAE to HEEQ for {len(bx)} data points")

    # Convert bx, by, bz into a single 2D NumPy array
    B_HAE = np.vstack((bx, by, bz)).T  # Shape: (N, 3)

    # Precompute all transformation matrices
    transformation_matrices = np.array([get_heliocentric_transformation_matrices(t)[1] for t in time])

    # Apply transformation in a vectorized way
    B_HEEQ = np.einsum('ijk,ik->ij', transformation_matrices, B_HAE)

    # Compute total magnetic field strength
    bt = np.linalg.norm(B_HEEQ, axis=1)

    return B_HEEQ[:, 0], B_HEEQ[:, 1], B_HEEQ[:, 2]


def HEEQ_to_HAE(bx, by, bz, time):

    #print(f"Converting HEEQ to HAE for {len(bx)} data points")

    # Convert bx, by, bz into a single 2D NumPy array (shape: N x 3)
    B_HEEQ = np.vstack((bx, by, bz)).T  

    # Precompute all inverse transformation matrices S2^-1 (shape: N x 3 x 3)
    transformation_matrices = np.array([np.linalg.inv(get_heliocentric_transformation_matrices(t)[1]) for t in time])

    # Apply the transformation in a vectorized way
    B_HAE = np.einsum('ijk,ik->ij', transformation_matrices, B_HEEQ)

    return B_HAE[:, 0], B_HAE[:, 1], B_HAE[:, 2]


def HEE_to_HEEQ(bx, by, bz, time):
    hae_bx, hae_by, hae_bz = HEE_to_HAE(bx, by, bz, time)
    heeq_bx, heeq_by, heeq_bz = HAE_to_HEEQ(hae_bx, hae_by, hae_bz, time)

    return heeq_bx, heeq_by, heeq_bz


def HEEQ_to_HEE(bx, by, bz, time):
    hae_bx, hae_by, hae_bz = HEEQ_to_HAE(bx, by, bz, time)
    hee_bx, hee_by, hee_bz = HAE_to_HEE(hae_bx, hae_by, hae_bz, time)

    return hee_bx, hee_by, hee_bz



def HEEQ_to_RTN(x, y, z, bx, by, bz):
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    bx = np.asarray(bx)
    by = np.asarray(by)
    bz = np.asarray(bz)

    # Stack position and magnetic field vectors
    r_vec = np.stack([x, y, z], axis=-1)
    b_vec = np.stack([bx, by, bz], axis=-1)

    # Normalize R (radial unit vector)
    r_hat = r_vec / np.linalg.norm(r_vec, axis=1)[:, np.newaxis]

    # Define constant z-axis of HEEQ
    z_hat = np.array([0, 0, 1])

    # Calculate T (tangential unit vector): T = Z × R
    t_hat = np.cross(np.tile(z_hat, (len(r_hat), 1)), r_hat)

    # Normalize T (to be safe)
    t_hat /= np.linalg.norm(t_hat, axis=1)[:, np.newaxis]

    # Calculate N (normal unit vector): N = R × T
    n_hat = np.cross(r_hat, t_hat)

    # Project B onto RTN basis
    rtn_bx = np.einsum('ij,ij->i', b_vec, r_hat)
    rtn_by = np.einsum('ij,ij->i', b_vec, t_hat)
    rtn_bz = np.einsum('ij,ij->i', b_vec, n_hat)

    return rtn_bx, rtn_by, rtn_bz


def RTN_to_HEEQ(x, y, z, bx, by, bz):
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    bx = np.asarray(bx)
    by = np.asarray(by)
    bz = np.asarray(bz)

    # Stack position and magnetic field vectors
    r_vec = np.stack([x, y, z], axis=-1)
    b_rtn = np.stack([bx, by, bz], axis=-1)

    # Normalize R (radial unit vector)
    r_hat = r_vec / np.linalg.norm(r_vec, axis=1)[:, np.newaxis]

    # HEEQ z-axis
    z_hat = np.array([0, 0, 1])

    # T = Z × R
    t_hat = np.cross(np.tile(z_hat, (len(r_hat), 1)), r_hat)
    t_hat /= np.linalg.norm(t_hat, axis=1)[:, np.newaxis]

    # N = R × T
    n_hat = np.cross(r_hat, t_hat)
    n_hat /= np.linalg.norm(n_hat, axis=1)[:, np.newaxis]

    # Construct the RTN → HEEQ rotation matrix for each sample
    # The RTN basis is [r_hat, t_hat, n_hat]
    # So, the transformation is B_HEEQ = R.T @ B_RTN
    # where R = [r_hat, t_hat, n_hat].T per sample
    # Use einsum for efficient batched dot product

    R = np.stack([r_hat, t_hat, n_hat], axis=-1)  # shape: (N, 3, 3)
    b_heeq = np.einsum('nij,ni->nj', R, b_rtn)

    heeq_bx, heeq_by, heeq_bz = b_heeq[:, 0], b_heeq[:, 1], b_heeq[:, 2]
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


def GSM_to_HEEQ(bx, by, bz, time):
    gse_bx, gse_by, gse_bz = GSM_to_GSE(bx, by, bz, time)
    hee_bx, hee_by, hee_bz = GSE_to_HEE(gse_bx, gse_by, gse_bz)
    heeq_bx, heeq_by, heeq_bz = HEE_to_HEEQ(hee_bx, hee_by, hee_bz, time)

    return heeq_bx, heeq_by, heeq_bz
