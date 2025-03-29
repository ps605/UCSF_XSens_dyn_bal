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

# SETUP
plt.ioff()

flag_seperateXYZ    = False
flag_makeGIF        = False
flag_neckPevlis     = True
flag_useAngle       = True
flag_ACFandMP       = False # Compute Autocorreletion and Matrix Profile

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
data_type = ['seg']

ang_T8inGlob = pd.DataFrame(np.arange(100),columns=['initialise'])

# List .files in directory, loop through them and check for .csv
# csv_files = glob.glob(data_path + '*sen_eul.csv')

# params = np.zeros([len(csv_files),16])
# params = pd.DataFrame(columns=['Patient_ID','BLN_Dx', 'BLN_Dy', 'BLN_X_c', 'BLN_Y_c', 'BLN_semi_major', 'BLN_semi_minor','BLN_ecc', 'BLN_X_rot', 'BLN_area', \
                                # '6WK_Dx', '6WK_Dy', '6WK_X_c', '6WK_Y_c', '6WK_semi_major', '6WK_semi_minor', '6WK_ecc', '6WK_X_rot', '6WK_area'])

for i_ptnt in post_op_ptnts:
     i_ptnt = f"{i_ptnt:03d}"

     for i_cond in conditions:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        plt.ioff()
        plt.figure()

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
            # Unwrap and remove initial offset
            eul_recon_uwr[:,0] = np.unwrap(eul_recon[:,0]) 
            eul_recon_uwr[:,1] = np.unwrap(eul_recon[:,1])
            eul_recon_uwr[:,2] = np.unwrap(eul_recon[:,2]) 

            eul_recon_uwr[:,0] = eul_recon_uwr[:,0] - eul_recon_uwr[0,0]
            eul_recon_uwr[:,1] = eul_recon_uwr[:,1] - eul_recon_uwr[0,1]
            eul_recon_uwr[:,2] = eul_recon_uwr[:,2] - eul_recon_uwr[0,2]

            eul_recon = np.rad2deg(eul_recon)
            # eul_recon_filt = np.rad2deg(eul_recon_filt)
            eul_recon_uwr = signal.filtfilt(b_imu,a_imu,np.rad2deg(eul_recon_uwr),0)

            if i_data_type == 'sen':
                eul_recon_uwr[:,1:2] = - eul_recon_uwr[:,1: 2]

            # Check Euler with plots
            lim = 270 # 3/2*np.pi

            fig.suptitle(i_ptnt + '_' + i_cond)
            ax1.plot(eul_x_T8[:], label="raw_euler_" + i_data_type)    
            # ax1.plot(np.rad2deg(roll[:]),label="RoPiYa")
            # ax1.plot(eul_recon[:,2],label="ZYX")
            # ax1.plot(eul_recon_filt[:,2], label="ZYXfilt")
            ax1.plot(eul_recon_uwr[:,2], label="ZYX_f_U_" + i_data_type)
            ax1.set_ylim([-lim,lim])
            ax1.set_title('X')

            ax2.plot(eul_y_T8[:])
            # ax2.plot(np.rad2deg(pitch[:]))
            # ax2.plot(eul_recon[:,1])
            # ax2.plot(eul_recon_filt[:,1])
            ax2.plot(eul_recon_uwr[:,1])
            ax2.set_ylim([-lim,lim])
            ax2.set_yticks([])
            ax2.set_title('Y')

            ax3.plot(eul_z_T8[:])
            # ax3.plot(np.rad2deg(yaw[:]))
            # ax3.plot(eul_recon[:,0])
            # ax3.plot(eul_recon_filt[:,0])
            ax3.plot(eul_recon_uwr[:,0])
            ax3.set_ylim([-lim,lim])
            ax3.set_yticks([])
            ax3.set_title('Z')
            
            plt.plot(eul_x_T8,eul_y_T8)
            # plt.plot(eul_recon[:,2],eul_recon[:,1], label="ZYXfilt")
            plt.plot(eul_recon_uwr[:,2],eul_recon_uwr[:,1], label="ZYX_f_U_" + i_data_type)
            plt.xlim(-15, 25)
            plt.ylim(-20, 20) 
            plt.title(i_ptnt + '_' + i_cond)
            plt.xlabel('X rot')
            plt.xlabel('Y rot')
            
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