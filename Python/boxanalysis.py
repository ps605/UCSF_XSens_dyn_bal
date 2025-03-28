
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

data_path = '../In/Analysis/'
data_path_out = '../Out/Analysis/Paper/'

var = 'area_'
title = 'KDE Area'
csv_data = pd.read_csv(data_path + 'kdesva_new.csv')
csv_data = csv_data.drop([18,28])

# fig = sns.stripplot(y=csv_data['area'], x=csv_data['Time'], hue=csv_data['Group'])
# sns.lineplot(y=csv_data['area'], x=csv_data['Time'], hue=csv_data['Group'])
# plt.plot([0,1], [25,50], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)


# R for BLN
pcc_IN_BLN = stats.linregress(csv_data['area_BLN'][:9], csv_data['SVA_BLN'][:9])
pcc_NIN_BLN = stats.linregress(csv_data['area_BLN'][10:], csv_data['SVA_BLN'][10:])

# R for 6WK
pcc_IN_6WK = stats.linregress(csv_data['area_6WK'][:9], csv_data['SVA_6WK'][:9])
pcc_NIN_6WK = stats.linregress(csv_data['area_6WK'][10:], csv_data['SVA_6WK'][10:])

# R for Deltas
pcc_IN_DELTA = stats.linregress(csv_data['area_6WK'][:9]-csv_data['area_BLN'][:9], csv_data['SVA_6WK'][:9]-csv_data['SVA_BLN'][:9])
pcc_NIN_DELTA = stats.linregress(csv_data['area_6WK'][10:]-csv_data['area_BLN'][10:], csv_data['SVA_6WK'][10:]-csv_data['SVA_BLN'][10:])

# Plots
jitter = 0.05
x_jitter_pre = np.random.normal(loc=0, scale=jitter, size=32)
x_jitter_post = np.random.normal(loc=1, scale=jitter, size=32)

l_blue_rgb = np.array([183, 201, 226])/255
d_blue_rgb = np.array([21, 27, 141])/255

# Plot correlation of KDE and SVA at 6 WK
plt.scatter(csv_data['area_6WK'][:9],csv_data['SVA_6WK'][:9], facecolors=d_blue_rgb, edgecolors='k')
plt.scatter(csv_data['area_6WK'][10:],csv_data['SVA_6WK'][10:], facecolors=l_blue_rgb, edgecolors='k')
plt.plot(csv_data['area_6WK'][:9], pcc_IN_6WK.intercept+pcc_IN_6WK.slope*csv_data['area_6WK'][:9], color=d_blue_rgb, label='Incident(r=0.58, p=0.10)')
plt.plot(csv_data['area_6WK'][10:], pcc_NIN_6WK.intercept+pcc_NIN_6WK.slope*csv_data['area_6WK'][10:], color=l_blue_rgb, label='Non-Incident (r=-0.30, p=0.20)')
plt.xlabel('KDE Area (deg^2)')
plt.ylabel('SVA (mm)')
plt.axis('square')
plt.legend(loc='lower right')
plt.savefig(data_path_out + 'corr_areaVSsva_6WK.png')

plt.scatter(x_jitter_pre[0:9],csv_data[var + 'BLN'][0:9], facecolors=d_blue_rgb, edgecolors='k')
plt.scatter(x_jitter_pre[9:33],csv_data[var + 'BLN'][9:33], facecolors=l_blue_rgb,edgecolors='k')
plt.scatter(x_jitter_post[0:9],csv_data[var + '6WK'][0:9], facecolors=d_blue_rgb,edgecolors='k')
plt.scatter(x_jitter_post[9:33],csv_data[var + '6WK'][9:33], facecolors=l_blue_rgb, edgecolors='k')



c = [d_blue_rgb,d_blue_rgb,d_blue_rgb,d_blue_rgb,d_blue_rgb,d_blue_rgb,d_blue_rgb,d_blue_rgb,d_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb,l_blue_rgb]
for idx in range(x_jitter_post.size):
    plt.plot([x_jitter_pre[idx],x_jitter_post[idx]], [csv_data[var + 'BLN'][idx],csv_data[var + '6WK'][idx]], color = c[idx], linewidth = 1.1, alpha=0.5, linestyle = '-', zorder=-1)

plt.ylabel('deg^2')
plt.xticks([0,1], ['PRE-op', 'POST-op'])
plt.title(title)
plt.savefig(data_path_out +   var +'.png')
print('--END--')
    