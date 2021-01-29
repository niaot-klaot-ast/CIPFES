#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit
import csv
import matplotlib.pyplot as plt
from scipy.interpolate import splev, interp1d, LSQUnivariateSpline
from numpy.polynomial import legendre
import pickle

atlas_filename = 'table1.all.txt'
atlas_file = open(atlas_filename)
atlas_node = np.array([12, 24, 36, 48, 52, 60, 68, 76, 88, 96, 108, 120])
atlas_term = ['Wavenumber', 'Wavenumber_uncertainty', 'Vacuum_wavelength', 'Wavelength_uncertainty',
              'Line_width', 'Measured_S/N', 'Species', 'Odd_energy', 'Odd_J', 'Even_energy', 'Even_J', 'sigma-sigma_Ritz']
atlas = {}
for i in range(len(atlas_term)):
    atlas[atlas_term[i]] = []

for line in atlas_file: 
    atlas[atlas_term[0]].append(float(line[0:atlas_node[0]]))
    atlas[atlas_term[1]].append(float(line[atlas_node[0]:atlas_node[1]]))
    atlas[atlas_term[2]].append(float(line[atlas_node[1]:atlas_node[2]]))
    atlas[atlas_term[3]].append(float(line[atlas_node[2]:atlas_node[3]]))
    atlas[atlas_term[4]].append(float(line[atlas_node[3]:atlas_node[4]]))
    atlas[atlas_term[5]].append(float(line[atlas_node[4]:atlas_node[5]]))
    atlas[atlas_term[6]].append(line[atlas_node[5]:atlas_node[6]].strip())
    atlas[atlas_term[7]].append(line[atlas_node[6]:atlas_node[7]].strip())
    atlas[atlas_term[8]].append(line[atlas_node[7]:atlas_node[8]].strip())
    atlas[atlas_term[9]].append(line[atlas_node[8]:atlas_node[9]].strip())
    atlas[atlas_term[10]].append(line[atlas_node[9]:atlas_node[10]].strip())
    if len(line[atlas_node[10]:atlas_node[11]].strip()) == 0:
         atlas[atlas_term[11]].append(np.inf)
    else:
         atlas[atlas_term[11]].append(float(line[atlas_node[10]:atlas_node[11]]))
    
atlas[atlas_term[0]] = np.array(atlas[atlas_term[0]])
atlas[atlas_term[1]] = np.array(atlas[atlas_term[1]])
atlas[atlas_term[2]] = np.array(atlas[atlas_term[2]])
atlas[atlas_term[3]] = np.array(atlas[atlas_term[3]])
atlas[atlas_term[4]] = np.array(atlas[atlas_term[4]])
atlas[atlas_term[5]] = np.array(atlas[atlas_term[5]])
atlas[atlas_term[11]] = np.array(atlas[atlas_term[11]])

mask1 = (atlas['Vacuum_wavelength']>400) & (atlas['Vacuum_wavelength']<740)
mask2 = atlas['sigma-sigma_Ritz']!=np.inf
mask = mask1 & mask2
x1 = atlas['Wavenumber'][mask]
x2 = atlas['Line_width'][mask]/1e3
width_frac1 = np.zeros(shape=(0,3), dtype=float)
for i in range(len(x1)):
    width_frac1 = np.append(width_frac1, [[x1[i], x2[i], x2[i]/x1[i]]], axis=0)
    
mask1 = (atlas['Vacuum_wavelength']>400) & (atlas['Vacuum_wavelength']<740)
mask2 = np.zeros(shape=len(mask1), dtype=bool)
for i in range(len(mask2)):
    if atlas['Species'][i].split(' ')[0] == 'Ar':
        mask2[i] = True
    else:
        mask2[i] = False
mask = mask1 & mask2
x1 = atlas['Wavenumber'][mask]
x2 = atlas['Line_width'][mask]/1e3
width_frac2 = np.zeros(shape=(0,3), dtype=float)
for i in range(len(x1)):
    width_frac2 = np.append(width_frac2, [[x1[i], x2[i], x2[i]/x1[i]]], axis=0)
    
    
ep1 = width_frac1[:,2].min()*1e6
ep2 = width_frac2[:,2].max()*1e6
fig1, ax1 = plt.subplots()
ax1.hist(width_frac1[:,2]*1e6, bins=np.linspace(ep1, ep2, 40), facecolor='darkblue', edgecolor='none', alpha=0.3)
ax1.hist(width_frac2[:,2]*1e6, bins=np.linspace(ep1, ep2, 40), facecolor='darkred', edgecolor='none', alpha=0.3)

wf1 = np.median(width_frac1[:,2])*1e6
wf2 = np.median(width_frac2[:,2])*1e6
ax1.plot([wf1, wf1], [0, 1000], '--', color='darkblue', lw=2)
ax1.plot([wf2, wf2], [0, 1000], '--', color='darkred', lw=2)
ax1.set_ylim([0,1000])
ax1.set_xticks([0,1,2,3,4,5,6,7])
# ax1.set_xticklabels(['0', r'$1\times10^{-6}$', r'$2\times10^{-6}$', r'$3\times10^{-6}$', r'$4\times10^{-6}$', 
                    # r'$5\times10^{-6}$', r'$6\times10^{-6}$', r'$7\times10^{-6}$'])
ax1.set_xlabel(''.join([r'$\rm{FWHM}_{\lambda}/\lambda$', ' ', r'$[10^{-6}]$']), fontsize = 14, verticalalignment='center')
ax1.set_ylabel('number of lines', fontsize = 14, horizontalalignment='center', verticalalignment='center')
ax1.tick_params(which='major', labelsize = 12)
fig1.set_size_inches(6.0, 4.5)
fig1.savefig('linewidth.png', dpi=200)

print (wf1, wf2)
