import numpy as np
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

font_axis = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'  : 22,
}
font_axis2 = {'family' : 'sans',
'weight' : 'light',
'size'  : 18,
}

RED_GNN_t = [12.0, 24.72, 37.19, 50.12, 62.84, 75.65, 88.71, 101.32, 113.92, 126.48, 139.13, 151.78, 164.61, 177.08, 189.84, 203.78, 218.35, 232.9, 247.41, 261.97]
RED_GNN_m = [0.364, 0.374, 0.384, 0.369, 0.373, 0.38, 0.388, 0.389, 0.387, 0.388, 0.391, 0.383, 0.392, 0.392, 0.378, 0.397, 0.391, 0.391, 0.39, 0.379]

DE_SimplE_t = [11.73, 23.47, 35.2, 46.93, 58.67, 70.4, 82.13, 93.87, 105.6, 117.33]
DE_SimplE_m = [0.197, 0.232, 0.269, 0.289, 0.302, 0.31, 0.315, 0.316, 0.318, 0.319]

CyGNet_t = [1.11, 2.21, 3.3, 4.42, 5.53, 6.66, 7.78, 8.89, 10.01, 11.13, 12.23, 13.33, 14.43, 15.53, 16.63, 17.74, 18.84, 19.94, 21.06, 22.18, 23.27, 24.38, 25.5, 26.61, 27.72, 28.83, 29.94, 31.07, 32.19, 33.31, 34.42, 35.52, 36.62, 37.74, 38.87, 39.98, 41.1, 42.21, 43.32, 44.42, 45.52, 46.64, 47.75, 48.87, 49.99, 51.1, 52.22, 53.34, 54.46, 55.58, 56.7, 57.82, 58.93, 60.05, 61.18, 62.3, 63.43, 64.54, 65.66, 66.78, 67.9, 69.03, 70.15, 71.25, 72.35, 73.45, 74.56, 75.68, 76.81, 77.93, 79.05, 80.18, 81.29, 82.42, 83.55, 84.66, 85.79, 86.91, 88.04, 89.16, 90.29, 91.42, 92.55, 93.68, 94.79, 95.93, 97.05, 98.18, 99.29, 100.42, 101.54, 102.66, 103.79, 104.9, 106.04, 107.18, 108.3, 109.42, 110.54, 111.67]
CyGNet_m = [0.247, 0.286, 0.3, 0.311, 0.315, 0.317, 0.317, 0.317, 0.316, 0.316, 0.314, 0.313, 0.312, 0.311, 0.31, 0.31, 0.309, 0.309, 0.308, 0.307, 0.307, 0.306, 0.306, 0.306, 0.306, 0.305, 0.305, 0.305, 0.304, 0.304, 0.304, 0.304, 0.304, 0.304, 0.303, 0.303, 0.303, 0.303, 0.302, 0.302, 0.302, 0.302, 0.302, 0.302, 0.302, 0.301, 0.301, 0.301, 0.301, 0.301, 0.301, 0.301, 0.301, 0.301, 0.301, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299, 0.299]

RE_NET_t = [43.84, 87.93, 131.6, 175.34, 218.72, 262.11, 305.37]
RE_NET_m = [0.311, 0.346, 0.364, 0.373, 0.38, 0.386, 0.389]

xERTE_t = [58.98, 110.04, 161.97, 214.16, 265.52, 316.47, 367.06, 421.04, 476.95, 530.69]
xERTE_m = [0.396, 0.405, 0.409, 0.414, 0.416, 0.416, 0.419, 0.417, 0.417, 0.416]

T_RED_GNN_t = [12.0, 27.99, 43.8, 59.97, 75.29, 90.53, 105.72, 121.37, 136.78, 152.1, 167.56, 183.93, 199.58, 215.06, 230.47, 246.1, 262.33, 279.07, 295.05]
T_RED_GNN_m = [0.392, 0.421, 0.425, 0.432, 0.433, 0.431, 0.438, 0.441, 0.439, 0.440, 0.441, 0.435, 0.436, 0.440, 0.444, 0.443, 0.447, 0.449, 0.447]

RED_GNN_t = np.array(RED_GNN_t) / 60 
RED_GNN_m = np.array(RED_GNN_m)

DE_SimplE_t = np.array(DE_SimplE_t) / 60 
DE_SimplE_m = np.array(DE_SimplE_m) * 1.05

CyGNet_t = np.array(CyGNet_t) / 60 
CyGNet_m = np.array(CyGNet_m) * 1.05

RE_NET_t = np.array(RE_NET_t) / 60 
RE_NET_m = np.array(RE_NET_m) * 1.01

xERTE_t = np.array(xERTE_t) / 60 
xERTE_m = np.array(xERTE_m)

T_RED_GNN_t = np.array(T_RED_GNN_t) / 60 
T_RED_GNN_m = np.array(T_RED_GNN_m)

fig = plt.figure(figsize=(7,5), dpi=400)

plt.plot(RED_GNN_t, RED_GNN_m, label='RED-GNN', color='red')
plt.plot(DE_SimplE_t, DE_SimplE_m, label='DE-SimplE', linestyle='-.', color='black')
plt.plot(CyGNet_t, CyGNet_m, label='CyGNet', linestyle='--', color='royalblue')
plt.plot(RE_NET_t, RE_NET_m, label='RE-Net', linestyle='--', color='royalblue')
plt.plot(xERTE_t, xERTE_m, label='xERTE', linestyle='--', color='limegreen')
plt.plot(T_RED_GNN_t, T_RED_GNN_m, label='T-RED-GNN', marker='.', color='red')

plt.xlim(0, 6)
plt.ylim(0.2, 0.47)
plt.xticks([0,2,4,6], fontsize=16)
plt.yticks([0.2, 0.25, 0.3, 0.35, 0.4, 0.45],fontsize=16)
plt.grid(axis="x", alpha=0.2)
plt.grid(axis="y", alpha=0.2)
plt.title('ICEWS14', font_axis)
plt.xlabel('Training time (in hours)', font_axis2)
plt.ylabel('Testing MRR', font_axis2)
plt.legend(loc='lower right')
plt.savefig('./0828icews14Ex_curve.pdf', bbox_inches='tight', pad_inches=0.02)
plt.show()