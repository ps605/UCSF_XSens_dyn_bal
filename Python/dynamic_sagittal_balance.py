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
flag_neckPevlis     = True

# Filtering            
f_order = 14#2
f_cutoff = 8#0.25
f_sampling = 60
f_nyquist = f_cutoff/(f_sampling/2)
b, a = signal.butter(f_order, f_nyquist, btype='lowpass')

# Smoothing
f_order_sm = 6#2
f_cutoff_sm = 4#0.25
f_sampling_sm = 60
f_nyquist_sm = f_cutoff_sm/(f_sampling_sm/2)
b_sm, a_sm = signal.butter(f_order_sm, f_nyquist_sm, btype='lowpass')

# This is a script to develop metrics to estimate dynamic saggital balance (DSB) of patients with markerless mocap methods

# Where to read data from
data_path = '../In/Box_data/'

# List .files in directory, loop through them and check for .csv
csv_files = glob.glob(data_path + '*pos.csv')

# params = np.zeros([len(csv_files),16])
params = pd.DataFrame(columns=['Patient_ID','BLN_Dx', 'BLN_Dy', 'BLN_X_c', 'BLN_Y_c', 'BLN_semi_major', 'BLN_semi_minor','BLN_ecc', 'BLN_X_rot', 'BLN_area', \
                                '6WK_Dx', '6WK_Dy', '6WK_X_c', '6WK_Y_c', '6WK_semi_major', '6WK_semi_minor', '6WK_ecc', '6WK_X_rot', '6WK_area'])

for i_csv in range(len(csv_files)):

    # Get base trial name
    csv_path = csv_files.__getitem__(i_csv)    
    csv_file = os.path.basename(csv_path)
    trial_name = data_path + csv_file[:-8] #'IMU_Segment_pos_xyz'
    id_int = int(csv_file[0:3])

    # Get joint position data
    # Load in tracked joint data from 3D pose and pass to array (XYZ)   
    data_xyz = pd.read_csv(trial_name + '_pos.csv')
    data_xyz = data_xyz.drop(columns='Frame')
    # Load in tracked 3D IMU Orientation and pass to array (quaternions)    
    data_q0123 = pd.read_csv(trial_name + '_qua.csv')  
    data_q0123 = data_q0123.drop(columns='Frame')
    # Load in tracked 3D IMU Orientation and pass to array (quaternions)    
    data_eul = pd.read_csv(trial_name + '_eul.csv')  
    data_eul = data_eul.drop(columns='Frame')
    # Load in tracked 3D IMU Free Accelerations and pass to array (quaternions)    
    data_acc = pd.read_csv(trial_name + '_acc.csv')  
    data_acc = data_acc.drop(columns='Frame')

    ## Get position info

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
        
    ## Get quaternion info

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

    ## Get Euler info
    # Get indeces of L/R Foot.
    idx_footR_eul = int(data_eul.columns.get_loc('Right Foot x'))
    idx_footL_eul = int(data_eul.columns.get_loc('Left Foot x'))

    # To np.array
    eul = np.array(data_eul, dtype='float')

    # Split to x, y, z and filter
    eul_x = signal.filtfilt(b,a,eul[:,0::3], axis=0)
    eul_y = signal.filtfilt(b,a,eul[:,1::3], axis=0)
    eul_z = signal.filtfilt(b,a,eul[:,2::3], axis=0)

    ## Get Accelaration data

    # Get indeces of L/R Foot.
    idx_footR_acc = int(data_acc.columns.get_loc('Right Foot x'))
    idx_footL_acc = int(data_acc.columns.get_loc('Left Foot x'))

    # To np.array
    acc = np.array(data_acc, dtype='float')

    # Split to x, y, z and filter
    acc_x = signal.filtfilt(b,a,acc[:,0::3], axis=0)
    acc_y = signal.filtfilt(b,a,acc[:,1::3], axis=0)
    acc_z = signal.filtfilt(b,a,acc[:,2::3], axis=0)

    # Calculate Jerk norm
    jer_footR = np.sqrt((np.diff(acc_x[:,idx_footR_acc//3])*f_sampling)**2 + (np.diff(acc_y[:,idx_footR_acc//3])*f_sampling)**2 + (np.diff(acc_z[:,idx_footR_acc//3])*f_sampling)**2)
    jer_footL = np.sqrt((np.diff(acc_x[:,idx_footL_acc//3])*f_sampling)**2 + (np.diff(acc_y[:,idx_footL_acc//3])*f_sampling)**2 + (np.diff(acc_z[:,idx_footL_acc//3])*f_sampling)**2)
    
    # Smoothing
    jer_footR_sm = signal.filtfilt(b_sm, a_sm, jer_footR)
    jer_footL_sm = signal.filtfilt(b_sm, a_sm, jer_footL)

    # Get max
    max_jer_R,_ = signal.find_peaks(jer_footR_sm, distance=70, height=100)
    max_jer_L,_ = signal.find_peaks(jer_footL_sm, distance=70, height=100)

    # Get gyration and invert
    gyr_footR = -eul_y[:,idx_footR_eul//3]
    gyr_footL = -eul_y[:,idx_footL_eul//3]
    
    # Get max
    max_gyr_R,_ = signal.find_peaks(gyr_footR, distance=70, height=100)
    max_gyr_L,_ = signal.find_peaks(gyr_footL, distance=70, height=100)

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

    # Check Plots
    plt.figure()
    plt.plot(jer_footL)
    plt.plot(max_gyr_L,jer_footL_sm[max_gyr_L],'x')


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

   
    ## Calculate delta(mid shoulder, pelvis)
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

        v_angle = np.zeros([n_frames,1])
        # Loop through and transform to pelvis ref system
        pos_neck_inPelvis = np.zeros([n_frames, 3])
        for i_frame in range(n_frames):
            # Get rotation matrix at each frame
            rm_pelvis = R.from_quat([ori_q1[i_frame, idx_pelvis_q], ori_q2[i_frame, idx_pelvis_q], ori_q3[i_frame, idx_pelvis_q], ori_q0[i_frame, idx_pelvis_q]])
            rmi_pelvis = rm_pelvis.inv()
            rmi_pelvis.as_matrix()
            pos_neck_inPelvis[i_frame,:] = rmi_pelvis.apply(d_neckPelvis[i_frame,:])
            v_angle[i_frame] = np.linalg.norm(rmi_pelvis.as_rotvec(degrees=True))

    v_angle_filt = signal.filtfilt(b,a,v_angle,0)

    if v_angle_filt[0]>75:
        v_angle_filt = -v_angle_filt

    # v_angle_filt = gaussian_filter1d(v_angle,10)
    v_angle_d1 = np.diff(v_angle_filt,1,0)*f_sampling
    v_angle_d2 = np.diff(v_angle_d1,1,0)*f_sampling
    frame_cutoff = np.where(v_angle_d1 >15)[0][0]

    # pol = np.polyfit(range(0,n_frames), v_angle_filt,30)
    # p2 = np.poly1d(np.squeeze(pol))

    # plt.plot(v_angle)
    plt.plot(v_angle_filt)
    plt.plot(v_angle_d1)
    plt.plot((frame_cutoff, frame_cutoff), (100,-30), 'r')
    plt.title(csv_file[0:-12])       
    plt.savefig(data_path + 'Figures/' + csv_file[0:-12] + 'cutoff.png')
    plt.close()
    # plt.plot(v_angle_d2)
    

    x_min = np.min(pos_neck_inPelvis[:frame_cutoff,0])
    x_max = np.max(pos_neck_inPelvis[:frame_cutoff,0])
    y_min = np.min(pos_neck_inPelvis[:frame_cutoff,1])
    y_max = np.max(pos_neck_inPelvis[:frame_cutoff,1])

    x_dist = x_max - x_min
    y_dist = y_max - y_min

    # Fit Ellipse
    ellipse_fit_cart = fit_ellipse(pos_neck_inPelvis[0:frame_cutoff,0], pos_neck_inPelvis[0:frame_cutoff,1])
    ellipse_fit_polr = cart_to_pol(ellipse_fit_cart)
    x_e, y_e = get_ellipse_pts(ellipse_fit_polr)

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
       
    # If baseline has already been recorded
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
    # pos_foot_r_x = pose_x[:,idx_foot_r_p]
    # vel_foot_r_x = np.diff(pos_foot_r_x)*f_cap
    # # True = Foot contact
    # thld_r_x = np.abs(vel_foot_r_x)<500

    # pos_foot_l_x = pose_x[:,idx_foot_l_p]
    # vel_foot_l_x = np.diff(pos_foot_l_x)*f_cap
    # # True = Foot contact
    # thld_l_x = np.abs(vel_foot_l_x)<500

    # #quiv_col = np.where(thld_l_x, 'g', 'r')

    # n_frames, n_cols = np.shape(pose_xyz)
    frames_v = range(n_frames)
    

    # quiv_col = []
    # for i in frames_v:
    #     if thld_r_x[i]==True and thld_l_x[i]==False:
    #         quiv_col.append('g')
    #     elif thld_r_x[i]==False and thld_l_x[i]==True:
    #         quiv_col.append('r')
    #     elif thld_r_x[i]==True and thld_l_x[i]==True:
    #         quiv_col.append('b')
    #     elif thld_r_x[i]==False and thld_l_x[i]==False:
    #         quiv_col.append('y')

    # Plot in transverse plave vs frames
    plt.figure()
    plt.scatter(pos_neck_inPelvis[0:frame_cutoff,0], pos_neck_inPelvis[0:frame_cutoff,1], c=range(0,frame_cutoff), cmap='jet')
    plt.plot(x_e, y_e)
    clb = plt.colorbar()
    clb.ax.set_title('Frames')
    plt.grid()
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    plt.xlabel('Anterior(+) / Posterior (-) displacement (mm)')
    plt.ylabel('Left (+) / Right (-) displacement (mm)') 
    plt.title(csv_file[0:-12])       
    plt.savefig(data_path + 'Figures/' + csv_file[0:-12] + '_d_NeckinP.png')
    plt.close()

    # # Plot sagittal and coronal against frames
    # plt.figure()
    # plt.plot(frames_v[0:plot_to], pos_neck_inPelvis[0:plot_to,0], c='r')
    # plt.plot(frames_v[0:plot_to], pos_neck_inPelvis[0:plot_to,1], c='b')
    # plt.ylim(-100, 100)
    # plt.xlabel('Frame Number')
    # plt.ylabel('Mid-Shoulder Position in Pelvis Reference System') 
    # plt.legend(['Red - Sagittal (A/P)', 'Blue - Coronal (L/R)'], loc='upper right')
    # plt.savefig(data_path + 'Figures/' + csv_file[0:-12] + '_d_MSPinP_sep.png')
    # plt.close()

    # # Plot quiver plot
    # plt.figure()
    # plt.plot(pos_pelvis[0:plot_to,0], pos_pelvis[0:plot_to,1], c='orange')
    # plt.quiver(pos_pelvis[0:plot_to,0], pos_pelvis[0:plot_to,1], d_neckPelvis[0:plot_to,0], d_neckPelvis[0:plot_to,1], angles='xy', scale_units='xy', scale=0.5, color=quiv_col, headwidth=0, headlength=0, headaxislength=0)
    # plt.xlabel('Anterior displacement (mm)')
    # plt.ylabel('Lateral displacement (mm)') 
    # ax = plt.gca()
    # ax.set_aspect('equal', adjustable='box')
    # plt.savefig(data_path + 'Figures/' + csv_file[0:-12] + '_d_MSPinP_sway.png')
    # plt.close()

    print(data_path + 'Figures/' + csv_file[0:-12] + '_d_MSPinP.png saved')

# xy_csv_df = pd.DataFrame(data = params, index = csv_files)
params.to_csv(data_path + 'params.csv')