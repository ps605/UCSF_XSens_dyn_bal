from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import glob, os
from lsqEllipse import *

# SETUP
plt.ioff()
flag_seperateXYZ    = False
flag_makeGIF        = False
flag_neckPevlis     = True


f_cap = 60
# This is a script to develop metrics to estimate dynamic saggital balance (DSB) of patients with markerless mocap methods

# Where to read data from
data_path = '../In/Box_data/'

# List .files in directory, loop through them and check for .csv
csv_files = glob.glob(data_path + '*pos.csv')

# params = np.zeros([len(csv_files),16])
params = pd.DataFrame(columns=['Patient_ID','BLN_Dx', 'BLN_Dy', 'BLN_X_c', 'BLN_Y_c', 'BLN_semi_major', 'BLN_semi_minor','BLN_ecc', 'BLN_X_rot', \
                                '6WK_Dx', '6WK_Dy', '6WK_X_c', '6WK_Y_c', '6WK_semi_major', '6WK_semi_minor', '6WK_ecc', '6WK_X_rot'])

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

    # Get indeces of Pelvis and L/R Shoulders (Upper Arm segment) for position. Divide by three for split data
    idx_pelvis_p = int(data_xyz.columns.get_loc('Pelvis x')/3)
    idx_shld_r_p = int(data_xyz.columns.get_loc('Right Upper Arm x')/3)
    idx_shld_l_p = int(data_xyz.columns.get_loc('Left Upper Arm x')/3) 
    idx_neck_p  =  int(data_xyz.columns.get_loc('Neck x')/3)
    idx_foot_r_p = int(data_xyz.columns.get_loc('Right Foot x')/3)
    idx_foot_l_p = int(data_xyz.columns.get_loc('Left Foot x')/3)

    # TO np.array and into mm
    pose_xyz = np.array(data_xyz, dtype='float')*1000

    # split data into X, Y, Z
    pose_x = pose_xyz[:,0::3]
    pose_y = pose_xyz[:,1::3]
    pose_z = pose_xyz[:,2::3]
        
    # Get quaternion .csv     
    # Load in tracked 3D orientation and pass to array (quaternions)    
    data_q0123 = pd.read_csv(trial_name + '_qua.csv')  
    data_q0123 = data_q0123.drop(columns='Frame')      

    # Get indeces of Pelvis and L/R Shoulders. Divide by 4 for split data
    idx_pelvis_q = int(data_q0123.columns.get_loc('Pelvis q0')/4)

    # To np.array quaternion
    ori_quat = np.array(data_q0123, dtype='float')

    # split data into q0, q1, q2, q3
    # MVN uses Scalar First for quaternion and SciPy uses Scalar Last for quaternion
    ori_q0 = ori_quat[:,0::4]                       
    ori_q1 = ori_quat[:,1::4] 
    ori_q2 = ori_quat[:,2::4]
    ori_q3 = ori_quat[:,3::4]

    # Check both position and quat have the same frames        
    n_frames_pos, n_cols_pos = np.shape(data_xyz)
    n_frames_quat, n_cols_quat = np.shape(data_q0123)

    if n_frames_pos == n_frames_quat:
        n_frames = n_frames_pos
    else:
        n_frames = np.min([n_frames_pos, n_frames_quat])

    frame_cutoff = int(n_frames/3)
    
    # Calculate delta(mid shoulder, pelvis)
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

        # Loop through and transform to pelvis ref system
        pos_neck_inPelvis = np.zeros([n_frames, 3])
        for i_frame in range(n_frames):
            # Get rotation matrix at each frame
            rm_pelvis = R.from_quat([ori_q1[i_frame, idx_pelvis_q], ori_q2[i_frame, idx_pelvis_q], ori_q3[i_frame, idx_pelvis_q], ori_q0[i_frame, idx_pelvis_q]])
            rmi_pelvis = rm_pelvis.inv()
            rmi_pelvis.as_matrix()
            pos_neck_inPelvis[i_frame,:] = rmi_pelvis.apply(d_neckPelvis[i_frame,:])

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
        
        # params[i_csv, 0] = x_dist
        # params[i_csv, 1] = y_dist
        # params[i_csv, 2] = ellipse_fit_polr[0]
        # params[i_csv, 3] = ellipse_fit_polr[1]
        # params[i_csv, 4] = ellipse_fit_polr[2]
        # params[i_csv, 5] = ellipse_fit_polr[3]
        # params[i_csv, 6] = ellipse_fit_polr[4]
        # params[i_csv, 7] = ellipse_fit_polr[5]

    elif csv_file.__contains__('BLN') and id_int in params['Patient_ID'].values:
        idx = params[params['Patient_ID'] == id_int].index
        rows, cols = params.shape
        # data = [id_int, x_dist, y_dist, ellipse_fit_polr[0], ellipse_fit_polr[1], ellipse_fit_polr[2], ellipse_fit_polr[3], ellipse_fit_polr[4], ellipse_fit_polr[5],[],[],[],[],[],[],[],[]]
        params.loc[idx, 'BLN_Dx'] = x_dist
        params.loc[idx, 'BLN_Dy'] = y_dist
        params.loc[idx, 'BLN_X_c'] = ellipse_fit_polr[0]
        params.loc[idx, 'BLN_Y_c'] = ellipse_fit_polr[1]
        params.loc[idx, 'BLN_semi_major'] = ellipse_fit_polr[2]
        params.loc[idx, 'BLN_semi_minor'] = ellipse_fit_polr[3]
        params.loc[idx, 'BLN_ecc'] = ellipse_fit_polr[4]
        params.loc[idx, 'BLN_X_rot'] = ellipse_fit_polr[5]
    
    elif csv_file.__contains__('6WK') and id_int not in params['Patient_ID'].values:
        rows, cols = params.shape
        # data = [id_int,[],[],[],[],[],[],[],[], x_dist, y_dist, ellipse_fit_polr[0], ellipse_fit_polr[1], ellipse_fit_polr[2], ellipse_fit_polr[3], ellipse_fit_polr[4], ellipse_fit_polr[5]]
        params.loc[rows, 'Patient_ID'] = id_int
        params.loc[rows, '6WK_Dx'] = x_dist
        params.loc[rows, '6WK_Dy'] = y_dist
        params.loc[rows, '6WK_X_c'] = ellipse_fit_polr[0]
        params.loc[rows, '6WK_Y_c'] = ellipse_fit_polr[1]
        params.loc[rows, '6WK_semi_major'] = ellipse_fit_polr[2]
        params.loc[rows, '6WK_semi_minor'] = ellipse_fit_polr[3]
        params.loc[rows, '6WK_ecc'] = ellipse_fit_polr[4]
        params.loc[rows, '6WK_X_rot'] = ellipse_fit_polr[5]
        
        # params[i_csv, 0+8] = x_dist
        # params[i_csv, 1+8] = y_dist
        # params[i_csv, 2+8] = ellipse_fit_polr[0]
        # params[i_csv, 3+8] = ellipse_fit_polr[1]
        # params[i_csv, 4+8] = ellipse_fit_polr[2]
        # params[i_csv, 5+8] = ellipse_fit_polr[3]
        # params[i_csv, 6+8] = ellipse_fit_polr[4]
        # params[i_csv, 7+8] = ellipse_fit_polr[5]

    # If baseline has already been recorded
    elif csv_file.__contains__('6WK') and id_int in params['Patient_ID'].values:
        idx = params[params['Patient_ID'] == id_int].index
        rows, cols = params.shape
        # data = [id_int,[],[],[],[],[],[],[],[], x_dist, y_dist, ellipse_fit_polr[0], ellipse_fit_polr[1], ellipse_fit_polr[2], ellipse_fit_polr[3], ellipse_fit_polr[4], ellipse_fit_polr[5]]
        params.loc[idx, '6WK_Dx'] = x_dist
        params.loc[idx, '6WK_Dy'] = y_dist
        params.loc[idx, '6WK_X_c'] = ellipse_fit_polr[0]
        params.loc[idx, '6WK_Y_c'] = ellipse_fit_polr[1]
        params.loc[idx, '6WK_semi_major'] = ellipse_fit_polr[2]
        params.loc[idx, '6WK_semi_minor'] = ellipse_fit_polr[3]
        params.loc[idx, '6WK_ecc'] = ellipse_fit_polr[4]
        params.loc[idx, '6WK_X_rot'] = ellipse_fit_polr[5]

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