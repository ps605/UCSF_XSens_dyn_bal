from scipy.spatial.transform import Rotation as R
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import math
import glob, os
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf as acf
import stumpy as sp
from lsqEllipse import *

# Peiod function
def calculate_period(signal, sampling_rate):
    """
    Calculates the period of a signal.

    Args:
        signal (array_like): The signal data.
        sampling_rate (float): The sampling rate of the signal (in Hz).

    Returns:
        float: The period of the signal (in seconds), or None if it cannot be determined.
    """
    N = len(signal)
    if N <= 1:
        return None

    yf = fft(signal)
    xf = np.fft.fftfreq(N, 1 / sampling_rate)
    yf_abs = np.abs(yf)

    peaks, _ = find_peaks(yf_abs[1:N//2], prominence=0.1 * np.max(yf_abs))

    if not peaks.size:
        return None
    
    dominant_frequency_index = peaks[np.argmax(yf_abs[peaks+1])]

    period = 1 / xf[dominant_frequency_index+1] 
    return period

def check_fft(signal, sampling_rate):
    N = signal.__len__()
    yf = rfft(signal - np.average(signal))
    xf = rfftfreq(N, 1 / sampling_rate)

    plt.figure()
    plt.plot(xf[2:], np.abs(yf[2:]), c='r')
    plt.show() 

def id_turn(data_eul, data_acc, angle_T8, flag_ACFandMP):
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
   
       ## Identify Gait events (Heel Strike HS; Toe Off TO)
    
    # Find turn
    idx_turn = np.min(np.diff(angle_T8)>45)
    
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

# SETUP
plt.ioff()
flag_plot = True
flag_ACFandMP = False

# Filtering and Smoothing Parameters
# For IMU           
f_order_imu = 2#8
f_cutoff_imu = 5#14
f_sampling_imu = 60
f_nyquist_imu = f_cutoff_imu/(f_sampling_imu/2)
b_imu, a_imu = signal.butter(f_order_imu, f_nyquist_imu, btype='lowpass')

# This is a script to develop metrics to estimate dynamic trunk sway

# Where to read data from
data_path = '../In/Box_data/'
post_op_ptnts = [2, 6, 8, 9, 16, 17, 18, 19, 22, 24, 27, \
                 31, 33, 34, 36, 38, 39, 44, 45, 46, 48, \
                 51, 52, 55, 60, 62, 64, 67, 70, 71, 79, \
                 83, 102, 112] # 99, 127, 146

conditions = ['BLN', '6WK']
data_type = ['seg', 'sen']

ang_T8inGlob = pd.DataFrame(np.arange(100),columns=['initialise'])

# Loop patients
for i_ptnt in post_op_ptnts:
     i_ptnt = f"{i_ptnt:03d}"

    # Loop conditions (baseline, 6 week)
     for i_cond in conditions:
        if flag_plot == True:
            fig, (ax1, ax2, ax3) = plt.subplots(1,3)
            plt.ioff()
            plt.figure()

        # Loop data type (sensor, segment)
        for i_data_type in data_type:

            ## --- Get base trial name ---
            csv_path = data_path + i_ptnt + '_' + i_cond + '_' + i_data_type + '_eul.csv'
            csv_file = os.path.basename(csv_path)
            trial_name = data_path + csv_file[:-4] #'IMU_Segment_pos_xyz'
            # id_int = int(csv_file[0:3])
            
            # if not id_int in post_op_ptnts:
            #     continue
            
            ## --- Get IMU data ---

            # Load in tracked 3D IMU Orientation and pass to array (quaternions)    
            data_q0123 = pd.read_csv(trial_name[:-3] + 'qua.csv')  
            data_q0123 = data_q0123.drop(columns='Frame')
            # Load in tracked 3D IMU Orientation and pass to array (euler)    
            data_eul = pd.read_csv(trial_name[:-3] + 'eul.csv')  
            data_eul = data_eul.drop(columns='Frame')
            # Load in tracked 3D IMU Free Accelerations and pass to array 
            # data_acc = pd.read_csv(trial_name + '_acc.csv')  
            # data_acc = data_acc.drop(columns='Frame')

            ## --- Get quaternion info ---
            # Get indeces of Pelvis (unused as of 20250328) and T8.
            # Quaternion
            idx_pelvis_q = int(data_q0123.columns.get_loc('Pelvis q0'))
            idx_T8_q = int(data_q0123.columns.get_loc('T8 q0'))
            # Euler
            idx_pelvis_e = int(data_eul.columns.get_loc('Pelvis x'))
            idx_T8_e = int(data_eul.columns.get_loc('T8 x'))

            # To np.array quaternion
            ori_quat = np.array(data_q0123, dtype='float')

            # split data into q0, q1, q2, q3
            # MVN uses Scalar First for quaternion and SciPy uses Scalar Last for quaternion. 
            ori_q0_T8 = ori_quat[:,idx_T8_q]      # Scalar                
            ori_q1_T8 = ori_quat[:,idx_T8_q+1] 
            ori_q2_T8 = ori_quat[:,idx_T8_q+2]
            ori_q3_T8 = ori_quat[:,idx_T8_q+3]

            r = R.from_quat([ori_q1_T8[1], ori_q2_T8[1], ori_q3_T8[1], ori_q0_T8[1]])
            eul_recon = np.array([0.000]*ori_q0_T8.__len__()*3).reshape(ori_q0_T8.__len__(),3)

            roll=np.zeros(ori_q0_T8.__len__())
            pitch=np.zeros(ori_q0_T8.__len__())
            yaw=np.zeros(ori_q0_T8.__len__())
            for i_frame in range(ori_q0_T8.__len__()):
                roll[i_frame] = math.atan2(2*(ori_q0_T8[i_frame]*ori_q1_T8[i_frame] + ori_q2_T8[i_frame]*ori_q3_T8[i_frame]),1-2*(ori_q1_T8[i_frame]*ori_q1_T8[i_frame] + ori_q2_T8[i_frame]*ori_q2_T8[i_frame]))
                pitch[i_frame] = math.asin(2*(ori_q0_T8[i_frame]*ori_q2_T8[i_frame] - ori_q3_T8[i_frame]*ori_q1_T8[i_frame]))
                yaw[i_frame] = math.atan2(2*(ori_q0_T8[i_frame]*ori_q3_T8[i_frame] + ori_q1_T8[i_frame]*ori_q2_T8[i_frame]),1-2*(ori_q2_T8[i_frame]*ori_q2_T8[i_frame] + ori_q3_T8[i_frame]*ori_q3_T8[i_frame]))
                
                r = R.from_quat([ori_q1_T8[i_frame], ori_q2_T8[i_frame], ori_q3_T8[i_frame], ori_q0_T8[i_frame]])
                eul_recon[i_frame,:] = r.as_euler('ZYX',degrees=False)

            ## --- Get Euler info ---
            ori_eul = np.array(data_eul, dtype='float')

            # Split to x, y, z 
            eul_x_T8 = ori_eul[:,idx_T8_e]
            eul_y_T8 = ori_eul[:,idx_T8_e+1]
            eul_z_T8 = ori_eul[:,idx_T8_e+2]

        # Initialise unwrap array
            eul_recon_uwr = np.array([0.000]*ori_q0_T8.__len__()*3).reshape(ori_q0_T8.__len__(),3)
            # Unwrap 
            eul_recon_uwr[:,0] = np.unwrap(eul_recon[:,0]) 
            eul_recon_uwr[:,1] = np.unwrap(eul_recon[:,1])
            eul_recon_uwr[:,2] = np.unwrap(eul_recon[:,2])

            eul_x_T8[:] = np.unwrap(eul_x_T8[:]) 
            eul_y_T8[:] = np.unwrap(eul_y_T8[:]) 
            eul_z_T8[:] = np.unwrap(eul_z_T8[:]) 

            # Remove offset
            eul_recon_uwr[:,0] = eul_recon_uwr[:,0] - eul_recon_uwr[0,0]
            eul_recon_uwr[:,1] = eul_recon_uwr[:,1] - eul_recon_uwr[0,1]
            eul_recon_uwr[:,2] = eul_recon_uwr[:,2] - eul_recon_uwr[0,2]

            eul_x_T8[:] = eul_x_T8[:] - eul_x_T8[0]
            eul_y_T8[:] = eul_y_T8[:] - eul_y_T8[0]
            eul_z_T8[:] = eul_z_T8[:] - eul_z_T8[0]

            eul_recon = np.rad2deg(eul_recon)
            
            # Filter
            eul_x_T8 = signal.filtfilt(b_imu,a_imu,eul_x_T8,0)
            eul_y_T8 = signal.filtfilt(b_imu,a_imu,eul_y_T8,0)
            eul_z_T8 = signal.filtfilt(b_imu,a_imu,eul_z_T8,0)


            eul_recon_uwr = signal.filtfilt(b_imu,a_imu,np.rad2deg(eul_recon_uwr),0)

            if i_data_type == 'sen':
                eul_recon_uwr[:,1:2] = - eul_recon_uwr[:,1: 2]

            # ang_T8inGlob = pd.concat([ang_T8inGlob, eul_recon_uwr[:,1]], ignore_index=False, axis=1)
            # ang_T8inGlob = pd.concat([ang_T8inGlob, eul_recon_uwr[:,0]], ignore_index=False, axis=1)
            # ang_T8inGlob = pd.concat([ang_T8inGlob, eul_recon_uwr[:,2]], ignore_index=False, axis=1)
            
            # Identify start(first step) and stop (180 turn)

            ## Plot ##
            if flag_plot == True:# Check Euler with plots
                x_ang_lim = 25 # 3/2*np.pi
                x_ang_lim = 25
                z_ang_lim = 270

                fig.suptitle(i_ptnt + '_' + i_cond)
                ax1.plot(eul_x_T8[:], label="raw_euler_" + i_data_type)    
                # ax1.plot(np.rad2deg(roll[:]),label="RoPiYa")
                # ax1.plot(eul_recon[:,2],label="ZYX")
                # ax1.plot(eul_recon_filt[:,2], label="ZYXfilt")
                # ax1.plot(eul_recon_uwr[:,2], label="ZYX_f_U_" + i_data_type)
                ax1.set_ylim([-x_ang_lim,x_ang_lim])
                ax1.set_title('X')

                ax2.plot(eul_y_T8[:])
                # ax2.plot(np.rad2deg(pitch[:]))
                # ax2.plot(eul_recon[:,1])
                # ax2.plot(eul_recon_filt[:,1])
                # ax2.plot(eul_recon_uwr[:,1])
                ax2.set_ylim([-x_ang_lim,x_ang_lim])
                ax2.set_yticks([])
                ax2.set_title('Y')

                ax3.plot(eul_z_T8[:])
                # ax3.plot(np.rad2deg(yaw[:]))
                # ax3.plot(eul_recon[:,0])
                # ax3.plot(eul_recon_filt[:,0])
                # ax3.plot(eul_recon_uwr[:,0])
                ax3.set_ylim([-z_ang_lim,z_ang_lim])
                ax3.set_yticks([-z_ang_lim, z_ang_lim])
                ax3.set_title('Z')
                
                plt.plot(eul_x_T8,eul_y_T8)
                # plt.plot(eul_recon[:,2],eul_recon[:,1], label="ZYXfilt")
                plt.plot(eul_recon_uwr[:,2],eul_recon_uwr[:,1], label="ZYX_f_U_" + i_data_type)
                plt.xlim(-15, 25)
                plt.ylim(-20, 20) 
                plt.title(i_ptnt + '_' + i_cond)
                plt.xlabel('X rot')
                plt.xlabel('Y rot')
            
        if flag_plot == True:
            fig.legend()
            fig.savefig('../Out/Analysis/Paper/Figures/imu_signal_comp/recon_' + i_ptnt + '_' + i_cond + '.pdf')

            plt.legend()
            plt.savefig('../Out/Analysis/Paper/Figures/imu_signal_comp/XvsY_' + i_ptnt + '_' + i_cond + '.pdf')
            print("--- Plotted: " + csv_file[:-4] + ' ---')


        # ang_T8inGlob.to_csv(data_path + 'pre_and_6WK_pelinGlob.csv')

        # if flag_useAngle == False:
        #     xy_csv_df = pd.DataFrame(data = params, index = csv_files)
        #     params.to_csv(data_path + 'params_pos.csv')
        # else:
        #     xy_csv_df = pd.DataFrame(data = params, index = csv_files)
        #     params.to_csv(data_path + 'params_ang_T8inPel.csv')