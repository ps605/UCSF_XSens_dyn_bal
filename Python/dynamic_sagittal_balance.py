from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os

# SETUP
plt.ioff()
flag_seperateXYZ    = False
flag_makeGIF        = False
flag_midShldrPevlis = True

# This is a script to develop metrics to estimate dynamic saggital balance (DSB) of patients with markerless mocap methods

# Where to read data from
data_path = '../In/Box_data/'

# List .files in directory, loop through them and check for .csv
csv_files = os.listdir(data_path)

for csv_file in csv_files:
    
    if csv_file.endswith('pos.csv'):
        
        # Get base trial name
        trial_name = data_path + csv_file[:-8] #'IMU_Segment_pos_xyz'

        # Get joint position data
        # Load in tracked joint data from 3D pose and pass to array (XYZ)   
        data_xyz = pd.read_csv(trial_name + '_pos.csv')
        data_xyz = data_xyz.drop(columns='Frame')

        # Get indeces of Pelvis and L/R Shoulders (Upper Arm segment) for position. Divide by three for split data
        idx_pelvis_p = int(data_xyz.columns.get_loc('Pelvis x')/3)
        idx_shld_r_p = int(data_xyz.columns.get_loc('Right Upper Arm x')/3)
        idx_shld_l_p = int(data_xyz.columns.get_loc('Left Upper Arm x')/3) 

        # TO np.array and into mm
        pose_xyz = np.array(data_xyz, dtype='float')*1000

        # split data into X, Y, Z
        pose_x = pose_xyz[:,0::3]
        pose_y = pose_xyz[:,1::3]
        pose_z = pose_xyz[:,2::3]
            
        # Get quaternion .csv     
        # Load in tracked joint data from 3D orientation and pass to array (quaternions)    
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


        # Calculate delta(mid shoulder, pelvis)
        if flag_midShldrPevlis == True:
            # ! should find better way of indexing data with header info from .mvn
            pos_pelvis = np.transpose(np.array([pose_x[:,idx_pelvis_p], pose_y[:,idx_pelvis_p], pose_z[:,idx_pelvis_p]]))
            pos_shldr_r = [pose_x[:,idx_shld_r_p], pose_y[:,idx_shld_r_p], pose_z[:,idx_shld_r_p]]
            pos_shldr_l = [pose_x[:,idx_shld_l_p], pose_y[:,idx_shld_l_p], pose_z[:,idx_shld_l_p]]
            # Calculate mid shoulders (in global frame)
            pos_midShldr = np.transpose(np.mean(np.array([pos_shldr_r, pos_shldr_l]), axis=0))
            # Calculate mid shoulder to pelvis (in global frame)
            d_midShldrPel = pos_midShldr - pos_pelvis

            # Loop through and transform to pelvis ref system
            pos_midShld_inPelvis = np.zeros([n_frames, 3])
            for i_frame in range(n_frames):
                # Get rotation matrix at each frame
                rm_pelvis = R.from_quat([ori_q1[i_frame, idx_pelvis_q], ori_q2[i_frame, idx_pelvis_q], ori_q3[i_frame, idx_pelvis_q], ori_q0[i_frame, idx_pelvis_q]])
                rmi_pelvis = rm_pelvis.inv()
                rmi_pelvis.as_matrix()
                pos_midShld_inPelvis[i_frame,:] = rmi_pelvis.apply(d_midShldrPel[i_frame,:])


        n_frames, n_cols = np.shape(pose_xyz)
        frames_v = range(n_frames)
        plot_to = 450

        # Plot in transverse plave vs frames
        plt.figure()
        plt.scatter(pos_midShld_inPelvis[0:plot_to,0], pos_midShld_inPelvis[0:plot_to,1], c=frames_v[0:plot_to], cmap='jet')
        clb = plt.colorbar()
        clb.ax.set_title('Frames')
        plt.grid()
        plt.xlim(-150, 150)
        plt.ylim(-150, 150)
        plt.xlabel('Anterior(+) / Posterior (-) displacement (mm)')
        plt.ylabel('Left (+) / Right (-) displacement (mm)') 
        plt.title(csv_file[0:-12])       
        plt.savefig(data_path + 'Figures/' + csv_file[0:-12] + '_d_MSPinP.png')
        plt.close()

        # Plot sagittal and coronal against frames
        plt.figure()
        plt.plot(frames_v[0:plot_to], pos_midShld_inPelvis[0:plot_to,0], c='r')
        plt.plot(frames_v[0:plot_to], pos_midShld_inPelvis[0:plot_to,1], c='b')
        plt.ylim(-100, 100)
        plt.xlabel('Frame Number')
        plt.ylabel('Mid-Shoulder Position in Pelvis Reference System') 
        plt.legend(['Red - Sagittal (A/P)', 'Blue - Coronal (L/R)'], loc='upper right')
        plt.savefig(data_path + 'Figures/' + csv_file[0:-12] + '_d_MSPinP_sep.png')
        plt.close()

        # Plot quiver plot
        plt.figure()
        plt.plot(pos_pelvis[0:plot_to,0], pos_pelvis[0:plot_to,1])
        plt.quiver(pos_pelvis[0:plot_to,0], pos_pelvis[0:plot_to,1], d_midShldrPel[0:plot_to,0], d_midShldrPel[0:plot_to,1], angles='xy', scale_units='xy', scale=0.5, headwidth=0, headlength=0, headaxislength=0)
        plt.xlabel('Anterior displacement (mm)')
        plt.ylabel('Lateral displacement (mm)') 
        plt.savefig(data_path + 'Figures/' + csv_file[0:-12] + '_d_MSPinP_sway.png')
        plt.close()

        print(data_path + 'Figures/' + csv_file[0:-12] + '_d_MSPinP.png saved')
