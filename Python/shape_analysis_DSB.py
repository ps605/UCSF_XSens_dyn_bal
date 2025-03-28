from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy import stats
from shapely import Polygon
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
import csv
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf as acf
import stumpy as sp
from lsqEllipse import *

# Setup
flag_plotKDE     = True
flag_plotGIF     = False
flag_plotConHull = False

# Where to read data from
data_path = '../In/Box_data/'
data_path_out = '../Out/Analysis/Paper/'

plt.ioff()


post_op_ptnts = [2, 6, 8, 9, 16, 17, 18, 19, 22, 24, 27, \
                 31, 33, 34, 36, 38, 39, 44, 45, 46, 48, \
                 51, 52, 55, 60, 62, 64, 67, 70, 71, 79, \
                 83, 99, 102, 112, 127, 146]


size = 2*post_op_ptnts.__len__()
bln_kde_centroid = np.array([0.00]*size).reshape(post_op_ptnts.__len__(),2)
_6wk_kde_centroid = np.array([0.00]*size).reshape(post_op_ptnts.__len__(),2)

bln_kde_area = np.array([0.00]*size).reshape(post_op_ptnts.__len__(),2)
_6wk_kde_area = np.array([0.00]*size).reshape(post_op_ptnts.__len__(),2)

bln_FE_range = np.array([0.00]*size).reshape(post_op_ptnts.__len__(),2)
_6wk_FE_range = np.array([0.00]*size).reshape(post_op_ptnts.__len__(),2)

bln_LB_range = np.array([0.00]*size).reshape(post_op_ptnts.__len__(),2)
_6wk_LB_range = np.array([0.00]*size).reshape(post_op_ptnts.__len__(),2)

red_rgb = np.array([250, 95, 85])/255
blu_rgb = np.array([41, 128, 185])/255

# Load data
data_T8inGlob = pd.read_csv(data_path + 'pre_and_6WK_T8inGlob.csv')
T8inGlob = np.array(data_T8inGlob, dtype='float')

ptnt_count = -1
ptnt_ID_list = []

for ptnt_ID in post_op_ptnts:

    ptnt_count = ptnt_count +1
    ptnt_ID_list.append(ptnt_ID)

    ptnt_ID_str = str(ptnt_ID).rjust(3,'0')
    ptnt_ID_BLN = ptnt_ID_str + '_BLN'
    ptnt_ID_6WK = ptnt_ID_str + '_6WK'

    idx_BLN = data_T8inGlob.columns.get_loc(ptnt_ID_BLN + '_LB')
    idx_6WK = data_T8inGlob.columns.get_loc(ptnt_ID_6WK + '_LB')

    T8inGlob_BLN = T8inGlob[:,idx_BLN:idx_BLN+3]
    
    T8inGlob_6WK = T8inGlob[:,idx_6WK:idx_6WK+3]
    

    # Remove NaNs
    ang_FE_BLN = T8inGlob_BLN[:,2]
    ang_FE_BLN = ang_FE_BLN[~np.isnan(ang_FE_BLN)]
    ang_LB_BLN = T8inGlob_BLN[:,0]
    ang_LB_BLN = ang_LB_BLN[~np.isnan(ang_LB_BLN)]

    ang_FE_6WK = T8inGlob_6WK[:,2]
    ang_FE_6WK = ang_FE_6WK[~np.isnan(ang_FE_6WK)]
    ang_LB_6WK = T8inGlob_6WK[:,0]
    ang_LB_6WK = ang_LB_6WK[~np.isnan(ang_LB_6WK)]

    rows_BLN,  = np.shape(ang_FE_BLN)
    rows_6WK,  = np.shape(ang_FE_6WK)
    
    if flag_plotKDE:
    # Bivariate scatter/histogram/kernel density plot

    # BASELINE    
        bln_kde = sns.kdeplot(x=ang_FE_BLN, y=ang_LB_BLN, levels=[0.25, 0.50, 0.75, 1], fill='True', color="r", linewidths=1)
        plt.scatter(ang_FE_BLN, ang_LB_BLN, c = np.arange(rows_BLN), cmap='Reds', s=0.50, alpha=0.75)

        # Get centroid of 90th percentile region
        pth_bln = bln_kde.collections[-1].get_paths()
        verts_in_bln = pth_bln[0].vertices
        x = verts_in_bln[:,0]
        y = verts_in_bln[:,1]
        mean_x, mean_y = verts_in_bln.mean(axis=0)
        # plt.scatter(x, y, c='b', s=1)
        # plt.scatter(mean_x, mean_y, c='r', edgecolors='black')

        bln_kde_centroid[ptnt_count,0] = mean_x
        bln_kde_centroid[ptnt_count,1] = mean_y
        # plt.scatter(x, y, c='b', s=1)
        # plt.scatter(mean_x, mean_y, c='r', edgecolors='black')

        # Get perimiter of region contours and calculate areas then sum them   
        bln_prob_area = 0
        for bln_pths in bln_kde.collections[0].get_paths():
            bln_verts = bln_pths.vertices
            # plt.scatter(bln_verts[:,0], bln_verts[:,1],c='r', s=1)         
            poly_verts = Polygon(bln_verts)
            poly_area = poly_verts.area
            bln_prob_area = bln_prob_area + poly_area
    
        bln_kde_area[ptnt_count,0] = bln_prob_area

        bln_FE_range[ptnt_count,0] = ang_FE_BLN.min()
        bln_FE_range[ptnt_count,1] = ang_FE_BLN.max()

        bln_LB_range[ptnt_count,0] = ang_LB_BLN.min()
        bln_LB_range[ptnt_count,1] = ang_LB_BLN.max()

        plt.close()

        # 6WK POST-OP 
        plt.figure(100)
        _6wk_kde_kde = sns.kdeplot(x=ang_FE_6WK, y=ang_LB_6WK, levels=[0.25, 0.50, 0.75, 1], fill='True', color="b", linewidths=1)
        plt.scatter(ang_FE_6WK, ang_LB_6WK, c = np.arange(rows_6WK), cmap='Blues', s=0.50, alpha=0.75)

        pth_6wk = _6wk_kde_kde.collections[-1].get_paths()
        verts_in_6wk = pth_6wk[0].vertices
        x = verts_in_6wk[:,0]
        y = verts_in_6wk[:,1]
        mean_x, mean_y = verts_in_6wk.mean(axis=0)
        # plt.scatter(x, y, c='b', s=1)
        # plt.scatter(mean_x, mean_y, c='b', edgecolors='black')
        
        _6wk_kde_centroid[ptnt_count,0] = mean_x
        _6wk_kde_centroid[ptnt_count,1] = mean_y
        
        # Get perimiter of outer region for 6WK POST OP    
        _6wk_kde_prob_area = 0
        # !!! NOTE: Caution with .collections indexin as with plotting items keep getting added to Axes list
        for _6wk_kde_pths in _6wk_kde_kde.collections[0].get_paths():
            _6wk_kde_verts = _6wk_kde_pths.vertices
            # plt.scatter(_6wk_kde_verts[:,0], _6wk_kde_verts[:,1],c='b', s=1) 
            poly_verts = Polygon(_6wk_kde_verts)
            poly_area = poly_verts.area
            _6wk_kde_prob_area = _6wk_kde_prob_area + poly_area

        _6wk_kde_area[ptnt_count,0] = _6wk_kde_prob_area

        _6wk_FE_range[ptnt_count,0] = ang_FE_6WK.min()
        _6wk_FE_range[ptnt_count,1] = ang_FE_6WK.max()

        _6wk_LB_range[ptnt_count,0] = ang_LB_6WK.min()
        _6wk_LB_range[ptnt_count,1] = ang_LB_6WK.max()
    
    # make GIF
    if flag_plotGIF:
        def update(i):
            x = ang_FE_BLN[:i]
            y = ang_LB_BLN[:i]
            data = np.stack([x,y]).T
            scat.set_offsets(data)
            return(scat)

        fig, ax = plt.subplots()
        scat = ax.scatter(ang_FE_BLN[0], ang_LB_BLN[0], c=[1,0,0], s=1, alpha=0.75)
        plt.xlim(-15, 25)
        plt.ylim(-20, 20) 
        plt.axvline(c='grey', zorder=0)
        plt.axhline(c='grey', zorder=0)
        plt.xlabel('Flexion (+) / Extension (-) (\N{DEGREE SIGN})')
        plt.ylabel('Left (+) / Right (-) Lateral Bending (\N{DEGREE SIGN})') 

                
        ani = animation.FuncAnimation(fig = fig, func = update, frames = ang_FE_BLN.size, interval = 1, repeat = False)
        writer = animation.PillowWriter(fps = 60,
                                        metadata = 'None',  #dict(artist = 'Me')
                                        bitrate = 1000)   #1800
        
        ani.save(data_path_out + 'Figures/Animation_KDE_'  + ptnt_ID_str + '_T8Global_BLN_scatterOnly.gif', writer = 'pillow')
    
    # plt.title(ptnt_ID_str)       
    # plt.savefig(data_path_out + 'Figures/KDE_'  + ptnt_ID_str + '_T8Global_BLN_scatterOnly.png')

   
if flag_plotConHull:
    # Compute Convex Hull
    ang_FELB_BLN = np.column_stack((ang_FE_BLN, ang_LB_BLN))
    c_hull_BLN = ConvexHull(ang_FELB_BLN)

    ang_FELB_6WK = np.column_stack((ang_FE_6WK, ang_LB_6WK))
    c_hull_6WK = ConvexHull(ang_FELB_6WK)

    # Plot Convex Hull
    cm = plt.colormaps.get('RdYlBu')
    plt.figure()
    plt.axvline(c='grey', zorder=0)
    plt.axhline(c='grey', zorder=0)

    plt.scatter(ang_FE_BLN, ang_LB_BLN, c = np.arange(rows_BLN), cmap='Reds')
    for i_simplices in c_hull_BLN.simplices:
        plt.plot(ang_FELB_BLN[i_simplices,0], ang_FELB_BLN[i_simplices,1], 'r--', lw=2)
   
    # plt.scatter(ang_FE_6WK, ang_LB_6WK, c = np.arange(rows_6WK), cmap='Blues')
    # for i_simplices in c_hull_6WK.simplices:
    #     plt.plot(ang_FELB_6WK[i_simplices,0], ang_FELB_6WK[i_simplices,1], 'b--', lw=2)  

    plt.xlim(-15, 25)
    plt.ylim(-20, 20)        
    plt.xlabel('Flexion (+) / Extension (-) (\N{DEGREE SIGN})')
    plt.ylabel('Left (+) / Right (-) Lateral Bending (\N{DEGREE SIGN})') 
    # plt.title(ptnt_ID_str)       
    plt.savefig(data_path_out + 'Figures/'  + ptnt_ID_str + '_T8Global_BLN_scatterOnly.png')
    plt.close()

data_out = pd.DataFrame({'Patient_ID': ptnt_ID_list,\
                         'BLN_cent_x': bln_kde_centroid[:,0], '6WK_cent_x': _6wk_kde_centroid[:,0], 'D_cent_x': _6wk_kde_centroid[:,0] - bln_kde_centroid[:,0], \
                          'BLN_cent_y': bln_kde_centroid[:,1], '6WK_cent_y': _6wk_kde_centroid[:,1], 'D_cent_y': _6wk_kde_centroid[:,1] - bln_kde_centroid[:,1],\
                             'BLN_area': bln_kde_area[:,0], '6WK_area': _6wk_kde_area[:,0], 'D_area': _6wk_kde_area[:,0] - bln_kde_area[:,0]})
data_out.to_csv(data_path_out + 'T8_cent_area_KDE.csv')
print("--- Analysis complete ---")