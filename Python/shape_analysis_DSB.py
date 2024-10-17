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

# Where to read data from
data_path = '../In/Box_data/'
post_op_ptnts = [2, 6, 8, 9, 16, 17, 18, 19, 22, 24, 27, \
                 31, 33, 34, 36, 38, 39, 44, 45, 46, 48, \
                 51, 52, 55, 60, 62, 64, 67, 70, 71, 79, \
                 102]

red_rgb = np.array([250, 95, 85])/255
blu_rgb = np.array([41, 128, 185])/255

# Load data
data_T8inGlob = pd.read_csv(data_path + 'pre_and_6WK_T8inGlob.csv')
T8inGlob = np.array(data_T8inGlob, dtype='float')

for ptnt_ID in post_op_ptnts:
    ptnt_ID_str = str(ptnt_ID).rjust(3,'0')
    ptnt_ID_BLN = ptnt_ID_str + '_BLN'
    ptnt_ID_6WK = ptnt_ID_str + '_6WK'

    idx_BLN = data_T8inGlob.columns.get_loc(ptnt_ID_BLN + '_LB')
    idx_6WK = data_T8inGlob.columns.get_loc(ptnt_ID_6WK + '_LB')

    ptnt_T8inGlob_BLN = T8inGlob[:,idx_BLN:idx_BLN+3]
    rows_BLN, cols = np.shape(ptnt_T8inGlob_BLN)
    ptnt_T8inGlob_6WK = T8inGlob[:,idx_6WK:idx_6WK+3]
    rows_6WK, cols = np.shape(ptnt_T8inGlob_BLN)


    cm = plt.colormaps.get('RdYlBu')
    plt.figure()
    plt.axvline(c='grey', zorder=0)
    plt.axhline(c='grey', zorder=0)
    plt.scatter(ptnt_T8inGlob_BLN[:,2], ptnt_T8inGlob_BLN[:,0], c = np.arange(rows_BLN), cmap='Reds')
    plt.scatter(ptnt_T8inGlob_6WK[:,2], ptnt_T8inGlob_6WK[:,0], c = np.arange(rows_6WK), cmap='Blues')
    # plt.grid()
    plt.xlim(-20, 40)
    plt.ylim(-20, 20)        
    plt.xlabel('Flexion (+) / Extension (-) (\N{DEGREE SIGN})')
    plt.ylabel('Left (+) / Right (-) Lateral Bending (\N{DEGREE SIGN})') 
    plt.title(ptnt_ID_str)       
    plt.savefig(data_path + 'Figures/'  + ptnt_ID_str + '_T8Global_BLN_6WK.png')
    plt.close()
