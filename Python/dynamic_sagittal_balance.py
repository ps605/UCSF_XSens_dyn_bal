from scipy.spatial.transform import Rotation as R
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import glob, os
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf as acf
import stumpy as sp
from lsqEllipse import *

# SETUP
plt.ioff()
flag_seperateXYZ    = False
flag_makeGIF        = False
flag_neckPevlis     = False
flag_useAngle       = True
flag_ACFandMP       = False # Compute Autocorreletion and Matrix Profile

# Filtering and Smoothing Parameters
# For IMU           
f_order_imu = 8#2
f_cutoff_imu = 14#0.25
f_sampling_imu = 60
f_nyquist_imu = f_cutoff_imu/(f_sampling_imu/2)
b_imu, a_imu = signal.butter(f_order_imu, f_nyquist_imu, btype='lowpass')

# For HPE           
f_order = 2
f_cutoff = 0.25
f_sampling = 60
f_nyquist = f_cutoff/(f_sampling/2)
b, a = signal.butter(f_order, f_nyquist, btype='lowpass')

# Smoothing
f_order_sm = 8#2
f_cutoff_sm = 14#0.25
f_sampling_sm = 60
f_nyquist_sm = f_cutoff_sm/(f_sampling_sm/2)
b_sm, a_sm = signal.butter(f_order_sm, f_nyquist_sm, btype='lowpass')

# This is a script to develop metrics to estimate dynamic saggital balance (DSB) of patients with markerless mocap methods

# Where to read data from
data_path = '../In/Box_data/'
post_op_ptnts = [2, 6, 8, 9, 16, 17, 18, 19, 22, 24, 27, \
                 31, 33, 34, 36, 38, 39, 44, 45, 46, 48, \
                 51, 52, 55, 60, 62, 64, 67, 70, 71, 79, \
                 83, 99, 102, 112, 127, 146]

ang_T8inGlob = pd.DataFrame(np.arange(100),columns=['initialise'])

# List .files in directory, loop through them and check for .csv
csv_files = glob.glob(data_path + '*pos.csv')

# params = np.zeros([len(csv_files),16])
params = pd.DataFrame(columns=['Patient_ID','BLN_Dx', 'BLN_Dy', 'BLN_X_c', 'BLN_Y_c', 'BLN_semi_major', 'BLN_semi_minor','BLN_ecc', 'BLN_X_rot', 'BLN_area', \
                                '6WK_Dx', '6WK_Dy', '6WK_X_c', '6WK_Y_c', '6WK_semi_major', '6WK_semi_minor', '6WK_ecc', '6WK_X_rot', '6WK_area'])

for i_csv in range(len(csv_files)):

    ## --- Get base trial name ---
    csv_path = csv_files.__getitem__(i_csv)    
    csv_file = os.path.basename(csv_path)
    trial_name = data_path + csv_file[:-8] #'IMU_Segment_pos_xyz'
    id_int = int(csv_file[0:3])
    
    if not id_int in post_op_ptnts:
        continue
    
    ## --- Get joint position data ---
    # Load in tracked joint data from 3D pose and pass to array (XYZ)   
    data_xyz = pd.read_csv(trial_name + '_pos.csv')
    data_xyz = data_xyz.drop(columns='Frame')
    # Load in tracked Ergonomic Joint angles    
    data_ang = pd.read_csv(trial_name + '_ang.csv')  
    data_ang = data_ang.drop(columns='Frame')
    # Load in tracked 3D IMU Orientation and pass to array (quaternions)    
    data_q0123 = pd.read_csv(trial_name + '_qua.csv')  
    data_q0123 = data_q0123.drop(columns='Frame')
    # Load in tracked 3D IMU Orientation and pass to array (euler)    
    data_eul = pd.read_csv(trial_name[:-3] + 'sen_eul.csv')  
    data_eul = data_eul.drop(columns='Frame')
    # Load in tracked 3D IMU Free Accelerations and pass to array (quaternions)    
    data_acc = pd.read_csv(trial_name + '_acc.csv')  
    data_acc = data_acc.drop(columns='Frame')

    ## --- Get position info ---
    # Get indeces of Pelvis and L/R Shoulders (Upper Arm segment) for position. Divide by three for split data
    idx_pelvis_p = int(data_xyz.columns.get_loc('Pelvis x')/3)
    idx_shld_r_p = int(data_xyz.columns.get_loc('Right Upper Arm x')/3)
    idx_shld_l_p = int(data_xyz.columns.get_loc('Left Upper Arm x')/3) 
    idx_neck_p  =  int(data_xyz.columns.get_loc('Neck x')/3)
    idx_foot_r_p = int(data_xyz.columns.get_loc('Right Foot x')/3)
    idx_foot_l_p = int(data_xyz.columns.get_loc('Left Foot x')/3)

    # To np.array and into mm
    pose_xyz = np.array(data_xyz, dtype='float')*1000

    # split data into X, Y, Z
    pose_x = pose_xyz[:,0::3]
    pose_y = pose_xyz[:,1::3]
    pose_z = pose_xyz[:,2::3]
    
    ## --- Get angle info ---
    idx_pel_a = int(data_ang.columns.get_loc('Vertical_Pelvis Lateral Bending')/3)
    idx_T8_a = int(data_ang.columns.get_loc('Vertical_T8 Lateral Bending')/3)
    idx_T8inPel_a = int(data_ang.columns.get_loc('Pelvis_T8 Lateral Bending')/3)

    # To np.array and into mm
    joint_angle = np.array(data_ang, dtype='float')

    # split data into X, Y, Z
    angle_x = joint_angle[:,0::3]
    angle_y = joint_angle[:,1::3]
    angle_z = joint_angle[:,2::3] 

    ## --- Get quaternion info ---
    # Get indeces of Pelvis.
    idx_pelvis_q = int(data_q0123.columns.get_loc('Pelvis q0')/4)

    # To np.array quaternion
    ori_quat = np.array(data_q0123, dtype='float')

    # split data into q0, q1, q2, q3
    # MVN uses Scalar First for quaternion and SciPy uses Scalar Last for quaternion. Divide by 4 for split data
    ori_q0 = ori_quat[:,0::4]                       
    ori_q1 = ori_quat[:,1::4] 
    ori_q2 = ori_quat[:,2::4]
    ori_q3 = ori_quat[:,3::4]

    ## --- Get Euler info ---
    # Get indeces of L/R Foot.
    idx_footR_eul = int(data_eul.columns.get_loc('Right Foot x'))
    idx_footL_eul = int(data_eul.columns.get_loc('Left Foot x'))

    # To np.array
    eul = np.array(data_eul, dtype='float')

    # Split to x, y, z and filter
    eul_x = signal.filtfilt(b_imu,a_imu,eul[:,0::3], axis=0)
    eul_y = signal.filtfilt(b_imu,a_imu,eul[:,1::3], axis=0)
    eul_z = signal.filtfilt(b_imu,a_imu,eul[:,2::3], axis=0)

    # Get gyration and negate (literature uses the negative gyration about y-axis)
    gyr_footR = -eul_y[:,idx_footR_eul//3]
    gyr_footL = -eul_y[:,idx_footL_eul//3]
    
    # Get max and min
    max_gyr_R,_ = signal.find_peaks(gyr_footR, distance=50, prominence=10)
    max_gyr_L,_ = signal.find_peaks(gyr_footL, distance=50, prominence=10)

    min_gyr_R,_ = signal.find_peaks(-gyr_footR, distance=50, prominence=10)
    min_gyr_L,_ = signal.find_peaks(-gyr_footL, distance=50, prominence=10)

    ## --- Get Accelaration data ---
    # Get indeces of L/R Foot.
    idx_footR_acc = int(data_acc.columns.get_loc('Right Foot x'))
    idx_footL_acc = int(data_acc.columns.get_loc('Left Foot x'))

    # To np.array
    acc = np.array(data_acc, dtype='float')

    # Split to x, y, z and filter
    acc_x = signal.filtfilt(b_imu,a_imu,acc[:,0::3], axis=0)
    acc_y = signal.filtfilt(b_imu,a_imu,acc[:,1::3], axis=0)
    acc_z = signal.filtfilt(b_imu,a_imu,acc[:,2::3], axis=0)

    # Calculate Jerk norm
    jer_footR = np.sqrt((np.diff(acc_x[:,idx_footR_acc//3])*f_sampling_imu)**2 + (np.diff(acc_y[:,idx_footR_acc//3])*f_sampling_imu)**2 + (np.diff(acc_z[:,idx_footR_acc//3])*f_sampling_imu)**2)
    jer_footL = np.sqrt((np.diff(acc_x[:,idx_footL_acc//3])*f_sampling_imu)**2 + (np.diff(acc_y[:,idx_footL_acc//3])*f_sampling_imu)**2 + (np.diff(acc_z[:,idx_footL_acc//3])*f_sampling_imu)**2)
    
    # Smoothing
    # jer_footR_sm = signal.filtfilt(b_sm, a_sm, jer_footR)
    # jer_footL_sm = signal.filtfilt(b_sm, a_sm, jer_footL)

    # Get max
    max_jer_R,_ = signal.find_peaks(jer_footR, distance=50, height=150, prominence=20)
    max_jer_L,_ = signal.find_peaks(jer_footL, distance=50, height=150, prominence=20)

    ## --- Compute signal analysis ---
    if flag_ACFandMP:
        # Compute Autocorrelation Functions with 600 frame lags (10 seconds)
        acf_jer_R = acf(jer_footR, nlags=600)
        acf_jer_L = acf(jer_footL, nlags=600)
        acf_gyr_R = acf(gyr_footR, nlags=600)
        acf_gyr_L = acf(gyr_footL, nlags=600)

        # Matrix Profile
        window = 100
        jer_L_mp = sp.stump(jer_footL, window)
        idx_motif = np.argsort(jer_L_mp[:, 0])[0]
        idx_nearN = jer_L_mp[idx_motif,1]

        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
        axs[0].plot(jer_footL)
        axs[1].plot(jer_L_mp[:,0])
        axs[0].axvline(x=idx_motif, linestyle='dashed', color='r')
        axs[0].axvline(x=idx_nearN, linestyle='dashed', color='g')
        rect = Rectangle((idx_motif, 0), window, 800, facecolor='lightgrey')
        axs[0].add_patch(rect)
        axs[1].axvline(x=idx_motif, linestyle='dashed', color='r')
        axs[1].axvline(x=idx_nearN, linestyle='dashed', color='g')
        rect = Rectangle((idx_nearN, 0), window, 800, facecolor='lightgrey')
        axs[0].add_patch(rect)

        plt.figure()
        plt.plot(acf_jer_L)
        plt.plot(acf_jer_R)
        plt.figure()
        plt.plot(acf_gyr_L)
        plt.plot(acf_gyr_R)
   
    # Check both position and quat have the same frames        
    n_frames_pos, n_cols_pos = np.shape(data_xyz)
    n_frames_quat, n_cols_quat = np.shape(data_q0123)
    n_frames_acc, n_cols_acc = np.shape(data_acc)

    if n_frames_pos == n_frames_quat and n_frames_pos == n_frames_acc:
        n_frames = n_frames_pos
    else:
        n_frames = np.min([n_frames_pos, n_frames_quat, n_frames_acc])

   
    ## --- Calculate delta(mid shoulder, pelvis) ---
    if flag_neckPevlis == True:
        # ! should find better way of indexing data with header info from .mvn
        pos_pelvis = np.transpose(np.array([pose_x[:,idx_pelvis_p], pose_y[:,idx_pelvis_p], pose_z[:,idx_pelvis_p]]))
        # pos_shldr_r = [pose_x[:,idx_shld_r_p], pose_y[:,idx_shld_r_p], pose_z[:,idx_shld_r_p]]
        # pos_shldr_l = [pose_x[:,idx_shld_l_p], pose_y[:,idx_shld_l_p], pose_z[:,idx_shld_l_p]]
        pos_neck = np.transpose(np.array([pose_x[:,idx_neck_p], pose_y[:,idx_neck_p], pose_z[:,idx_neck_p]]))
        # Calculate mid shoulders (in global frame)
        # pos_midShldr = np.transpose(np.mean(np.array([pos_shldr_r, pos_shldr_l]), axis=0))
        # Calculate mid shoulder to pelvis (in global frame)
        d_neckPelvis = pos_neck - pos_pelvis

        # Get Transformed (?) Pelvis Angle
        v_angle = np.zeros([n_frames,1])
        # Loop through and transform to pelvis ref system
        outcome_var = np.zeros([n_frames, 3])
        for i_frame in range(n_frames):
            # Get rotation matrix at each frame
            rm_pelvis = R.from_quat([ori_q1[i_frame, idx_pelvis_q], ori_q2[i_frame, idx_pelvis_q], ori_q3[i_frame, idx_pelvis_q], ori_q0[i_frame, idx_pelvis_q]])
            rmi_pelvis = rm_pelvis.inv()
            rmi_pelvis.as_matrix()
            outcome_var[i_frame,:] = rmi_pelvis.apply(d_neckPelvis[i_frame,:])
            v_angle[i_frame] = np.linalg.norm(rmi_pelvis.as_rotvec(degrees=True))
    
    # TEMP - replace position with angle
    if flag_neckPevlis == True and flag_useAngle == True:
        outcome_var = np.transpose(np.array([angle_x[:,idx_pel_a], angle_y[:,idx_pel_a], angle_z[:,idx_pel_a]]))


    v_angle_filt = signal.filtfilt(b,a,v_angle,0)

    if v_angle_filt[0]>75:
        v_angle_filt = -v_angle_filt

    # Identify 180 deg U-turn
    v_angle_d1 = np.diff(v_angle_filt,1,0)*f_sampling
    v_angle_d2 = np.diff(v_angle_d1,1,0)*f_sampling
    # idx_turn = np.where(v_angle_d1 >15)[0][0]
    idx_pos_max = np.argmax(pos_pelvis[:,0] - pos_pelvis[0,0])
    idx_pos_min = np.argmin(pos_pelvis[:,0] - pos_pelvis[0,0])

    if np.abs(pos_pelvis[idx_pos_max,0]- pos_pelvis[0,0]) > np.abs(pos_pelvis[idx_pos_min,0]- pos_pelvis[0,0]):
        idx_turn = idx_pos_max
    else:
        idx_turn = idx_pos_min

    ## Identify Gait events (Heel Strike HS; Toe Off TO)
    # Find closest previous toe off (gyration min) to turn
    near_L = np.where((idx_turn - min_gyr_L)>0)[0]
    near_R = np.where((idx_turn - min_gyr_R)>0)[0]

    HS_L = max_gyr_L[near_L]
    HS_R = max_gyr_R[near_R]
    TO_L = min_gyr_L[near_L]
    TO_R = min_gyr_R[near_R]

    # Tidy up gait event so HS is first (removes toe off from standing)
    if TO_L[0] < HS_L[0]:
        TO_L = np.delete(TO_L,0)

    if TO_R[0] < HS_R[0]:
        TO_R = np.delete(TO_R,0)

    max_TO_L = np.max(min_gyr_L[near_L])
    max_TO_R = np.max(min_gyr_R[near_R])
    max_TO = np.maximum(max_TO_L, max_TO_R)

    # Find first HS
    min_HS = np.minimum(np.min(HS_L), np.min(HS_R))

    # Get Foot contact ranges
    n_TO_L = TO_L.shape[0]
    n_TO_R = TO_R.shape[0]

    foot_contact_L = np.vstack((HS_L[:n_TO_L], TO_L)).T
    foot_contact_R = np.vstack((HS_R[:n_TO_R], TO_R)).T

    if min_HS in HS_L:
        first_HS = 'L'
    elif min_HS in HS_R:
        first_HS = 'R'


    # Get Foot contact vectors
    foot_contact_L_v = []
    for j in range(foot_contact_L.shape[0]):
        frames_add = np.arange(foot_contact_L[j,0], foot_contact_L[j,1], 1)
        foot_contact_L_v.extend(frames_add)

    foot_contact_R_v = []
    for j in range(foot_contact_R.shape[0]):
        frames_add = np.arange(foot_contact_R[j,0], foot_contact_R[j,1], 1)
        foot_contact_R_v.extend(frames_add)

    quiv_col = []
    for i in range(min_HS, max_TO):
        if  i in foot_contact_R_v and i in foot_contact_L_v:
            quiv_col.append('r')
        elif i in foot_contact_L_v:
            quiv_col.append('b')
        elif i in foot_contact_R_v:
            quiv_col.append('g')
        else:
            quiv_col.append('y')

    # plt.plot(v_angle)
    plt.figure()
    plt.plot(v_angle_filt)
    plt.plot(v_angle_d1)
    plt.plot((max_TO, max_TO), (100,-30), 'r')
    plt.title(csv_file[0:-12])       
    plt.savefig(data_path + 'Figures/' + csv_file[0:-12] + 'cutoff.png')
    plt.close()
    # plt.plot(v_angle_d2)
    
    x_min = np.min(outcome_var[:max_TO,0])
    x_max = np.max(outcome_var[:max_TO,0])
    y_min = np.min(outcome_var[:max_TO,1])
    y_max = np.max(outcome_var[:max_TO,1])

    x_dist = x_max - x_min
    y_dist = y_max - y_min

    # Fit Ellipse
    if flag_useAngle == False:

        x_min = np.min(outcome_var[:max_TO,0])
        x_max = np.max(outcome_var[:max_TO,0])
        y_min = np.min(outcome_var[:max_TO,1])
        y_max = np.max(outcome_var[:max_TO,1])

        x_dist = x_max - x_min
        y_dist = y_max - y_min

        ellipse_fit_cart = fit_ellipse(outcome_var[min_HS:max_TO,0], outcome_var[min_HS:max_TO,1])
        ellipse_fit_polr = cart_to_pol(ellipse_fit_cart)
        x_e, y_e = get_ellipse_pts(ellipse_fit_polr)
    else:
        # Flexion/Extension
        x_min = np.min(outcome_var[:max_TO,2])
        x_max = np.max(outcome_var[:max_TO,2])
        # Lateral Bending
        y_min = np.min(outcome_var[:max_TO,0])
        y_max = np.max(outcome_var[:max_TO,0])

        x_dist = x_max - x_min
        y_dist = y_max - y_min

        ellipse_fit_cart = fit_ellipse(outcome_var[min_HS:max_TO,2], outcome_var[min_HS:max_TO,0])
        ellipse_fit_polr = cart_to_pol(ellipse_fit_cart)
        x_e, y_e = get_ellipse_pts(ellipse_fit_polr)

    # Save data to appropriate condition
    if csv_file.__contains__('BLN') and id_int not in params['Patient_ID'].values:
        rows, cols = params.shape
        params.loc[rows, 'Patient_ID'] = id_int
        params.loc[rows, 'BLN_Dx'] = x_dist
        params.loc[rows, 'BLN_Dy'] = y_dist
        params.loc[rows, 'BLN_X_c'] = ellipse_fit_polr[0]
        params.loc[rows, 'BLN_Y_c'] = ellipse_fit_polr[1]
        params.loc[rows, 'BLN_semi_major'] = ellipse_fit_polr[2]
        params.loc[rows, 'BLN_semi_minor'] = ellipse_fit_polr[3]
        params.loc[rows, 'BLN_ecc'] = ellipse_fit_polr[4]
        params.loc[rows, 'BLN_X_rot'] = ellipse_fit_polr[5]
        params.loc[rows, 'BLN_area'] = np.pi * ellipse_fit_polr[3] * ellipse_fit_polr[2]
      
    elif csv_file.__contains__('BLN') and id_int in params['Patient_ID'].values:
        idx = params[params['Patient_ID'] == id_int].index
        rows, cols = params.shape
        params.loc[idx, 'BLN_Dx'] = x_dist
        params.loc[idx, 'BLN_Dy'] = y_dist
        params.loc[idx, 'BLN_X_c'] = ellipse_fit_polr[0]
        params.loc[idx, 'BLN_Y_c'] = ellipse_fit_polr[1]
        params.loc[idx, 'BLN_semi_major'] = ellipse_fit_polr[2]
        params.loc[idx, 'BLN_semi_minor'] = ellipse_fit_polr[3]
        params.loc[idx, 'BLN_ecc'] = ellipse_fit_polr[4]
        params.loc[idx, 'BLN_X_rot'] = ellipse_fit_polr[5]
        params.loc[idx, 'BLN_area'] = np.pi * ellipse_fit_polr[3] * ellipse_fit_polr[2]
    
    elif csv_file.__contains__('6WK') and id_int not in params['Patient_ID'].values:
        rows, cols = params.shape
        params.loc[rows, 'Patient_ID'] = id_int
        params.loc[rows, '6WK_Dx'] = x_dist
        params.loc[rows, '6WK_Dy'] = y_dist
        params.loc[rows, '6WK_X_c'] = ellipse_fit_polr[0]
        params.loc[rows, '6WK_Y_c'] = ellipse_fit_polr[1]
        params.loc[rows, '6WK_semi_major'] = ellipse_fit_polr[2]
        params.loc[rows, '6WK_semi_minor'] = ellipse_fit_polr[3]
        params.loc[rows, '6WK_ecc'] = ellipse_fit_polr[4]
        params.loc[rows, '6WK_X_rot'] = ellipse_fit_polr[5]
        params.loc[rows, '6WK_area'] = np.pi * ellipse_fit_polr[3] * ellipse_fit_polr[2]
       
    elif csv_file.__contains__('6WK') and id_int in params['Patient_ID'].values:
        idx = params[params['Patient_ID'] == id_int].index
        rows, cols = params.shape
        params.loc[idx, '6WK_Dx'] = x_dist
        params.loc[idx, '6WK_Dy'] = y_dist
        params.loc[idx, '6WK_X_c'] = ellipse_fit_polr[0]
        params.loc[idx, '6WK_Y_c'] = ellipse_fit_polr[1]
        params.loc[idx, '6WK_semi_major'] = ellipse_fit_polr[2]
        params.loc[idx, '6WK_semi_minor'] = ellipse_fit_polr[3]
        params.loc[idx, '6WK_ecc'] = ellipse_fit_polr[4]
        params.loc[idx, '6WK_X_rot'] = ellipse_fit_polr[5]
        params.loc[idx, '6WK_area'] = np.pi * ellipse_fit_polr[3] * ellipse_fit_polr[2]

    # # Compute Foot Contact

    frames_v = range(n_frames)
    norm_frames = 100*(np.arange(frames_v[min_HS], frames_v[max_TO]) - frames_v[min_HS])*100/(frames_v[max_TO]-frames_v[min_HS+1])
    
    # Check Plots max jerk and gyration
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    axs[0].plot(jer_footL[:max_TO], color='blue')
    axs[0].plot(max_jer_L[max_jer_L<=max_TO], jer_footL[max_jer_L[max_jer_L<=max_TO]],'o', color='blue')   
    axs[1].plot(gyr_footL[:max_TO], color='blue', linestyle='-')
    axs[1].plot(min_gyr_L[min_gyr_L<=max_TO], gyr_footL[min_gyr_L[min_gyr_L<=max_TO]],'x', color='blue')
    axs[1].plot(max_gyr_L[max_gyr_L<=max_TO], gyr_footL[max_gyr_L[max_gyr_L<=max_TO]],'x', color='blue')
    for i_stance in range(foot_contact_L.shape[0]):
        rect = Rectangle((foot_contact_L[i_stance,0],-50), foot_contact_L[i_stance,1] - foot_contact_L[i_stance,0], 800, facecolor='lightblue', edgecolor='blue', alpha= 1)
        axs[0].add_patch(rect)
        rect = Rectangle((foot_contact_L[i_stance,0],-50), foot_contact_L[i_stance,1] - foot_contact_L[i_stance,0], 80, facecolor='lightblue', edgecolor='blue', alpha= 1)
        axs[1].add_patch(rect)

    axs[0].plot(jer_footR[:max_TO], color='green')
    axs[0].plot(max_jer_R[max_jer_R<=max_TO], jer_footR[max_jer_R[max_jer_R<=max_TO]],'o', color='green')
    axs[1].plot(gyr_footR[:max_TO], color='green', linestyle='-')
    axs[1].plot(min_gyr_R[min_gyr_R<=max_TO], gyr_footR[min_gyr_R[min_gyr_R<=max_TO]],'x', color='green')
    axs[1].plot(max_gyr_R[max_gyr_R<=max_TO], gyr_footR[max_gyr_R[max_gyr_R<=max_TO]],'x', color='green')
    for i_stance in range(TO_R.__len__()):
        rect = Rectangle((foot_contact_R[i_stance,0],-50), foot_contact_R[i_stance,1] - foot_contact_R[i_stance,0], 800, facecolor='lightgreen', edgecolor='green', alpha= 0.5)
        axs[0].add_patch(rect)
        rect = Rectangle((foot_contact_R[i_stance,0],-50), foot_contact_R[i_stance,1] - foot_contact_R[i_stance,0], 80, facecolor='lightgreen', edgecolor='green', alpha= 0.5)
        axs[1].add_patch(rect)

    axs[0].set_ylabel('Jerk norm (m/s^3)')
    axs[1].set_ylabel('Gyrarion y-axis - Euler (deg/s)')
    axs[1].set_xlabel('Frame Number')
    plt.savefig(data_path + 'Figures/' + csv_file[0:-12] + '_d_NeckinP_gaitEvents.png')
    plt.close()

    # Plot in transverse plane vs frames
    if flag_useAngle == False:
        plt.figure()
        plt.scatter(outcome_var[min_HS:max_TO,0], outcome_var[min_HS:max_TO,1])
        plt.plot(x_e, y_e)
        clb = plt.colorbar()
        clb.ax.set_title('Frames')
        plt.grid()
        plt.xlim(-150, 150)
        plt.ylim(-150, 150)
        plt.xlabel('Anterior(+) / Posterior (-) displacement (mm)')
        plt.ylabel('Left (+) / Right (-) displacement (mm)') 
        plt.title(csv_file[0:-12])       
        plt.savefig(data_path + 'Figures/' + csv_file[0:-12] + '_d_NeckinP_footstrike.png')
        plt.close()
    else:
        # Set colourmap
        cm = plt.colormaps.get('RdYlBu')
        plt.figure()
        plt.axvline(c='grey', zorder=0)
        plt.axhline(c='grey', zorder=0)
        plt.scatter(outcome_var[min_HS:max_TO,2], outcome_var[min_HS:max_TO,0], c = norm_frames)
        plt.plot(x_e, y_e, c = 'black')
        plt.scatter(ellipse_fit_polr[0],ellipse_fit_polr[1],c='black')
        clb = plt.colorbar(cmap=norm_frames)
        clb.ax.set_title('% Walk')
        # plt.grid()
        plt.xlim(-20, 40)
        plt.ylim(-20, 20)        
        plt.xlabel('Flexion (+) / Extension (-) (\N{DEGREE SIGN})')
        plt.ylabel('Left (+) / Right (-) Lateral Bending (\N{DEGREE SIGN})') 
        # plt.title(csv_file[0:-12])       
        plt.savefig(data_path + 'Figures/' + csv_file[0:-12] + '_angleT8Global_pcWalk.png')
        plt.close()

        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
        axs[0].plot(outcome_var[min_HS:max_TO,2],c='blue')
        axs[1].plot(outcome_var[min_HS:max_TO,0],c='green')
        axs[0].set_ylabel('Flex. (+) / Ext. (-) (\N{DEGREE SIGN})')
        axs[1].set_ylabel('L (+) / R (-) Bending (\N{DEGREE SIGN})')
        axs[1].set_xlabel('Frame Number')


    # # Plot quiver plot
    # if flag_useAngle == False:
    #     plt.figure()
    #     plt.plot(pos_pelvis[min_HS:max_TO,0], pos_pelvis[min_HS:max_TO,1], c='orange')
    #     plt.quiver(pos_pelvis[min_HS:max_TO,0], pos_pelvis[min_HS:max_TO,1], d_neckPelvis[min_HS:max_TO,0], d_neckPelvis[min_HS:max_TO,1],
    #                 angles='xy', scale_units='xy', scale=0.5, color=quiv_col, headwidth=0, headlength=0, headaxislength=0)
    #     plt.xlabel('Anterior displacement (mm)')
    #     plt.ylabel('Lateral displacement (mm)') 
    #     ax = plt.gca()
    #     ax.set_aspect('equal', adjustable='box')
    #     plt.savefig(data_path + 'Figures/' + csv_file[0:-12] + '_d_MSPinP_sway.png')
    #     plt.close()
    # else:
    #     plt.figure()
    #     plt.plot(pos_pelvis[min_HS:max_TO,0], pos_pelvis[min_HS:max_TO,1], c='orange')
    #     plt.quiver(pos_pelvis[min_HS:max_TO,2], pos_pelvis[min_HS:max_TO,0], d_neckPelvis[min_HS:max_TO,2], d_neckPelvis[min_HS:max_TO,0],
    #                 angles='xy', scale_units='xy', scale=0.5, color=quiv_col, headwidth=0, headlength=0, headaxislength=0)
    #     plt.xlabel('Anterior displacement (mm)')
    #     plt.ylabel('Lateral displacement (mm)') 
    #     ax = plt.gca()
    #     ax.set_aspect('equal', adjustable='box')
    #     plt.savefig(data_path + 'Figures/' + csv_file[0:-12] + '_angle_sway.png')
    #     plt.close()

    print(data_path + 'Figures/' + csv_file[0:-12] + ' saved')
    angLB_df = pd.DataFrame(outcome_var[min_HS:max_TO,0], columns=[csv_file[0:-12] + '_LB'])
    angAR_df = pd.DataFrame(outcome_var[min_HS:max_TO,1], columns=[csv_file[0:-12] + '_AR'])
    angFE_df = pd.DataFrame(outcome_var[min_HS:max_TO,2], columns=[csv_file[0:-12] + '_FE'])
    
    ang_T8inGlob = pd.concat([ang_T8inGlob, angLB_df], ignore_index=False, axis=1)
    ang_T8inGlob = pd.concat([ang_T8inGlob, angAR_df], ignore_index=False, axis=1)
    ang_T8inGlob = pd.concat([ang_T8inGlob, angFE_df], ignore_index=False, axis=1)


ang_T8inGlob.to_csv(data_path + 'pre_and_6WK_pelinGlob.csv')

if flag_useAngle == False:
    xy_csv_df = pd.DataFrame(data = params, index = csv_files)
    params.to_csv(data_path + 'params_pos.csv')
else:
    xy_csv_df = pd.DataFrame(data = params, index = csv_files)
    params.to_csv(data_path + 'params_ang_T8inPel.csv')