#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.optimize import root_scalar, curve_fit
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.interpolate import LSQUnivariateSpline, splev, interp1d, griddata, interp2d
from funcs import rmsd, CDG_fit, CDG_param1, moffat_fit, moffat_param1, super_gaussian_fit, super_gaussian_param1, leg2d_fit
from scipy.stats import norm
from numpy.polynomial import legendre
import pickle
from refractive_index import refractive_index_moist_air
from copy import deepcopy

speed_of_light = 299792458.0  
order_init = 64

def linear_gaussian(x, inten, x0, sigma, background):
    return inten*np.exp(-(x-x0)**2/(2*sigma**2)) + background
    
def gaussian(p,x):
    a, x0, sig = p
    return a * np.exp(-(x-x0)**2/(2.0*sig**2))

def CDG(x, mc, dc, a1, sig1, a2, sig2, background):
    x01 = (mc + dc)/2.0
    x02 = (mc - dc)/2.0
    p1 = [a1, x01, sig1]
    p2 = [a2, x02, sig2]
    return gaussian(p1, x) + gaussian(p2, x) + background
    
def moffat(x, inten, x0, theta, beta, delta, background):
    return inten * (1.0 + ((x-x0)/theta)**2)**(-beta * ((x-x0)/delta)**2) + background

def super_gaussian(x, inten, x0, sig, beta, background):
    return inten * np.exp(-(abs(x-x0)/sig)**beta) + background
    
readout_noise2 = 15.0 * 4.0**2    
    
filename = []
with open('thar_files.csv', 'r') as pt:
    reader = csv.reader(pt)
    for row in reader:
        filename.append(row)
        
        
f_lines = {}
f_lines_mask = {}
f_lines_len = {}
f_lines_param1 = {}
f_lines_param2 = {}
for h in range(len(filename)):
    
    f = pf.getdata(filename[h][0], header=False)
    g = pickle.load(open(filename[h][1], 'rb'))
    g_line_list = g['line_list']
    #0order, 1line_center, 2wav, 3type
    g_orders = np.unique(g_line_list[:,0]).astype(int)
    thar_bleed_mask = pickle.load(open(filename[h][2], "rb"))
    left_edge = int(thar_bleed_mask['x_left_edge'])
    right_edge = int(thar_bleed_mask['x_right_edge'])
    xind = np.arange(left_edge, right_edge+1)
    
    # obtain the accurate line-centers of the "initial guess" line list by Gaussian fit
    f_lines_1 = np.zeros(shape=(0,4), dtype=float)
    for i in range(len(f)):
        if i+order_init in g_orders:
            
            g_line_list_order = g_line_list[(g_line_list[:,0]==i+order_init) & (g_line_list[:,1]>left_edge) & 
                                            (g_line_list[:,1]<right_edge) & (g_line_list[:,3]-10<5)]
            # discard the Ar lines
            for j in range(len(g_line_list_order)):
                center_init = int(np.round(g_line_list_order[j,1]))
                x1 = np.arange(center_init-9, center_init+10)  
                x1_args = [k in x1 for k in xind]
                y1 = f[i, x1_args]
                inten = y1.max()-y1.min();  x0 = g_line_list_order[j,1];  sigma = 2.25;  background = 0.0
                par0 = [inten, x0, sigma, background]
                bounds0 = ([(0.0, x0-10.0, 0.0, 0.0), (np.inf, x0+10.0, np.inf, np.inf)])
                
                try:
                    popt, pcov = curve_fit(linear_gaussian, x1, y1, p0=par0, bounds=bounds0, sigma=np.ones(len(x1))*np.sqrt(np.average(y1)))
                    f_lines_1 = np.append(f_lines_1, [[i+order_init, popt[1], np.sqrt(pcov[1,1]), g_line_list_order[j,2]]], axis=0)
                    #0order, 1line_center, 2line_center_uncertainty, 3vacuum_wavelength
                except:
                    print (f'{g_line_list_order[j,2]} cannot be fitted in the Gaussian-fitting process')
                    pass
    
    
    # obtain the wavelength solution, and discard the lines which have large residuals
    m1 = 5+1; m2 = 3+1
    f_lines_1_rf = deepcopy(f_lines_1)
    normalizing_constant0 = 150.0
    normalizing_constant1 = 4096.0
    f_lines_1_rf[:,0] = f_lines_1_rf[:,0] / normalizing_constant0
    f_lines_1_rf[:,1] = f_lines_1_rf[:,1] / normalizing_constant1
    while 1:
        X = np.zeros(shape=(len(f_lines_1_rf), m1*m2), dtype=float)
        for i in range(len(f_lines_1_rf)):
            for j in range(m1):
                for k in range(m2):
                    c = np.zeros(shape=(m1,m2), dtype=float)
                    c[j,k] = 1.0
                    X[i,j*m2+k] = legendre.legval2d(f_lines_1_rf[i,0], f_lines_1_rf[i,1], c)
        
        wav_sol_1 = np.linalg.lstsq(X, f_lines_1_rf[:,0]*f_lines_1_rf[:,3], rcond=-1)
        wav_sol2d_1 = wav_sol_1[0].reshape(m1, m2)
        
        fitwv1 = np.array([legendre.legval2d(f_lines_1_rf[i,0], f_lines_1_rf[i,1], wav_sol2d_1)/f_lines_1_rf[i,0] for i in range(len(f_lines_1_rf))])
        res1 = fitwv1 - f_lines_1_rf[:,3]
        rms1 = rmsd(fitwv1, f_lines_1_rf[:,3])
        if rms1<3.0e-3: 
            print (rms1, len(f_lines_1_rf))
            break
        else:
            mask1 = abs(res1) > 3.0*rms1
            f_lines_1_rf = f_lines_1_rf[~mask1]
        
    f_lines_1_mask = np.array([i in f_lines_1_rf[:,3] for i in f_lines_1[:,3]])
    f_lines_2 = deepcopy(f_lines_1)
    f_lines_2_mask = deepcopy(f_lines_1_mask)
    f_lines_3 = deepcopy(f_lines_1)
    f_lines_3_mask = deepcopy(f_lines_1_mask)
    
    
    # obtain the parameters of the selected lines by super-Gaussian fit
    f_lines_3_param1 = np.zeros(shape=(len(f_lines_3), 10), dtype=float)
    f_lines_3_param2 = np.zeros(shape=(len(f_lines_3), 6), dtype=float)
    for i in range(len(f_lines_3)):
            
        if f_lines_3_mask[i] == True:
            cg = f_lines_3[i,1]
            x1 = np.arange(int(np.round(cg))-7, int(np.round(cg))+8)
            o1 = int(f_lines_3[i,0])
            x1_args = [k in x1 for k in xind]
            y1 = f[o1-order_init, x1_args]
            
            try:
                popt, pcov = super_gaussian_fit(x1, y1)
                rss = ((y1 - super_gaussian(x1, *popt))**2).sum()
                ess = ((y1 - np.average(y1))**2).sum()
                r2 = 1 - rss/ess
                if r2>0.995:
                    f_lines_3_param1[i] = np.concatenate((popt, [np.sqrt(pcov[j,j]) for j in range(5)]))
                    xm, sig_xm, height, sig_height, FWHM, sig_FWHM = super_gaussian_param1(popt, pcov)
                    f_lines_3_param2[i,0] = xm
                    f_lines_3_param2[i,1] = sig_xm
                    f_lines_3_param2[i,2] = height
                    f_lines_3_param2[i,3] = sig_height
                    f_lines_3_param2[i,4] = FWHM
                    f_lines_3_param2[i,5] = sig_FWHM
                else:
                    f_lines_3_mask[i] = False
                    print (f'{f_lines_3[i,3]} cannot pass the r^2 check')
            except:
                f_lines_3_mask[i] = False
                print (f'{f_lines_3[i,3]} cannot be fitted by super-Gaussian function')
                
    f_lines[h] = f_lines_3
    f_lines_mask[h] = f_lines_3_mask
    f_lines_param1[h] = f_lines_3_param1
    f_lines_param2[h] = f_lines_3_param2
                

# find the non-repetitive range of each order
non_rep_range = np.zeros(shape=(len(g_orders), 3), dtype=float)
for i in range(len(g_orders)-1):
    right_wav1 = legendre.legval2d(g_orders[i]/normalizing_constant0, 4095.0/normalizing_constant1, wav_sol2d_1)/(g_orders[i]/normalizing_constant0)
    left_wav2 = legendre.legval2d((g_orders[i+1])/normalizing_constant0, 0.0/normalizing_constant1, wav_sol2d_1)/((g_orders[i+1])/normalizing_constant0)
    cross_wav = (right_wav1 + left_wav2) / 2.0
    def wav2pos(x, o, w):
        return legendre.legval2d(o/normalizing_constant0, x/normalizing_constant1, wav_sol2d_1)/(o/normalizing_constant0) - w
    right_x1 = np.ceil(root_scalar(wav2pos, args=(g_orders[i], cross_wav), bracket=[0, 4095], method='ridder').root)
    left_x2 = np.floor(root_scalar(wav2pos, args=(g_orders[i+1], cross_wav), bracket=[0, 4095], method='ridder').root)
    non_rep_range[i,0] = g_orders[i]
    non_rep_range[i,2] = right_x1
    non_rep_range[i+1,1] = left_x2
    if i==0:
        non_rep_range[i,1] = 0.0
    if i==len(g_orders)-2:
        non_rep_range[i+1,0] = g_orders[i+1]
        non_rep_range[i+1,2] = 4095.0
        
popt_non_rep1 = np.polyfit(non_rep_range[1:,0], non_rep_range[1:,1], 3)
non_rep_range[0,1] = np.floor(np.poly1d(popt_non_rep1)(non_rep_range[0,0]))
popt_non_rep2 = np.polyfit(non_rep_range[:-1,0], non_rep_range[:-1,2], 3)
non_rep_range[-1,2] = np.ceil(np.poly1d(popt_non_rep2)(non_rep_range[-1,0]))


fl = np.zeros(shape=(0,4), dtype=float)
flm = np.zeros(shape=(0), dtype=bool)
flpm1 = np.zeros(shape=(0,10), dtype=float)
flpm2 = np.zeros(shape=(0,6), dtype=float)
for h in range(len(filename)):
    fl = np.append(fl, f_lines[h], axis=0)
    flm = np.append(flm, f_lines_mask[h], axis=0)
    flpm1 = np.append(flpm1, f_lines_param1[h], axis=0)
    flpm2 = np.append(flpm2, f_lines_param2[h], axis=0)
fl_nodes = np.array([0], dtype=int)
for i in f_lines.keys():
    fl_nodes = np.append(fl_nodes, [fl_nodes[-1]+len(f_lines[i])], axis=0)
    
    
deg_o = 1;  deg_x = 3
# fit the relation for sig with respect to order and position. compete different degrees, 1-3 wins
o1 = deepcopy(fl[flm, 0]) / normalizing_constant0
x1 = deepcopy(flpm1[flm, 1]) / normalizing_constant1
y1 = deepcopy(flpm1[flm, 2])
s1 = deepcopy(flpm1[flm, 7])
sig_sol2d, mask_sig = leg2d_fit(o1, x1, y1, deg_o, deg_x, sigma_clipping=True, sigma=s1)
mask_sig = np.array([i in y1[mask_sig] for i in flpm1[:,2]])
mask_o1 = fl[:,0]<np.median(g_orders);  mask_o2 = fl[:,0]>=np.median(g_orders)
mask_sig_o1 = mask_o1 & mask_sig
mask_sig_o2 = mask_o2 & mask_sig

plt.figure()
plt.errorbar(flpm1[flm,1], flpm1[flm,2], yerr=flpm1[flm,7], fmt='rs', zorder=0)
plt.errorbar(flpm1[mask_sig_o1,1], flpm1[mask_sig_o1,2], yerr=flpm1[mask_sig_o1,7], fmt='bs', zorder=1)
plt.errorbar(flpm1[mask_sig_o2,1], flpm1[mask_sig_o2,2], yerr=flpm1[mask_sig_o2,7], fmt='gs', zorder=1)
ot1 = np.arange(g_orders[0], g_orders[-1], 10)
xt1 = np.linspace(left_edge, right_edge, 1000)
for i in range(len(ot1)):
    plt.plot(xt1, legendre.legval2d(ot1[i]/normalizing_constant0, xt1/normalizing_constant1, sig_sol2d), '--', color=plt.cm.jet(i/len(ot1)))
plt.xlabel('pixel number', fontsize = 16, verticalalignment='top')
plt.ylabel(r'$\sigma$', fontsize = 16, horizontalalignment='center')
plt.gca().tick_params(which='major', labelsize = 14)

plt.figure()
ot1 = np.arange(g_orders[0], g_orders[-1], 5)
for i in range(len(ot1)):
    ax = plt.subplot(len(ot1),1,(i+1))
    xt1 = flpm1[mask_sig][fl[mask_sig][:,0]==ot1[i]][:,1]
    yt1 = flpm1[mask_sig][fl[mask_sig][:,0]==ot1[i]][:,2]
    st1 = flpm1[mask_sig][fl[mask_sig][:,0]==ot1[i]][:,7]
    ax.errorbar(xt1, yt1, yerr=st1, fmt='ks', ms=5)
    xt2 = np.linspace(left_edge, right_edge, 1000)
    ax.plot(xt2, legendre.legval2d(ot1[i]/normalizing_constant0, xt2/normalizing_constant1, sig_sol2d), 'g--')
    ax.set_ylabel(r'$\sigma$', fontsize = 16, horizontalalignment='center')
    ax.set_ylim((3.0, 4.0))
    ax.set_yticks([3.2, 3.5, 3.8])
    ax.tick_params(which='major', labelsize = 14)
    ax.text(0.05, 0.15, f'order = {ot1[i]}', ha='center', va='center', fontsize=12, transform=ax.transAxes)
    if i<len(ot1)-1:
        plt.setp(ax.get_xticklabels(), visible=False)
plt.xlabel('pixel number', fontsize = 16, verticalalignment='top')

plt.figure()
XIND, YIND = np.meshgrid(np.linspace(384, 3712, 1000), g_orders)
sig_2d = legendre.legval2d(YIND/normalizing_constant0, XIND/normalizing_constant1, sig_sol2d)
im1 = plt.gca().imshow(sig_2d, extent=(384, 3712, g_orders[0], g_orders[-1]), origin='lower', aspect=40)
cbar1 = plt.gcf().colorbar(im1, extend='both', shrink=0.75, ax=plt.gca())
cbar1.ax.set_xlabel(r'$\sigma$', fontsize=20)
cbar1.ax.tick_params(which='major', labelsize=18)
plt.xlabel('pixel position in the principal dispersion direction', fontsize = 20, verticalalignment='top')
plt.ylabel('order number', fontsize = 20, horizontalalignment='center', verticalalignment='bottom')
plt.gca().tick_params(which='major', labelsize=18)
cbar1.ax.tick_params(which='major', labelsize=18)
plt.gcf().set_size_inches(16.0, 9.0)
plt.gcf().savefig('zsig2d.png', dpi=300)


# fit the relation for beta with respect to order and position
o2 = deepcopy(fl[flm, 0]) / normalizing_constant0
x2 = deepcopy(flpm1[flm, 1]) / normalizing_constant1
y2 = deepcopy(flpm1[flm, 3])
s2 = deepcopy(flpm1[flm, 8])
beta_sol2d, mask_beta = leg2d_fit(o2, x2, y2, deg_o, deg_x, sigma_clipping=True, sigma=s2)
mask_beta = np.array([i in y2[mask_beta] for i in flpm1[:,3]])
mask_o1 = fl[:,0]<np.median(g_orders);  mask_o2 = fl[:,0]>=np.median(g_orders)
mask_beta_o1 = mask_o1 & mask_beta
mask_beta_o2 = mask_o2 & mask_beta

plt.figure()
plt.errorbar(flpm1[flm,1], flpm1[flm,3], yerr=flpm1[flm,8], fmt='rs', zorder=0)
plt.errorbar(flpm1[mask_beta_o1,1], flpm1[mask_beta_o1,3], yerr=flpm1[mask_beta_o1,8], fmt='bs', zorder=1)
plt.errorbar(flpm1[mask_beta_o2,1], flpm1[mask_beta_o2,3], yerr=flpm1[mask_beta_o2,8], fmt='gs', zorder=1)
ot1 = np.arange(g_orders[0], g_orders[-1], 10)
xt1 = np.linspace(left_edge, right_edge, 1000)
for i in range(len(ot1)):
    plt.plot(xt1, legendre.legval2d(ot1[i]/normalizing_constant0, xt1/normalizing_constant1, beta_sol2d), '--', color=plt.cm.jet(i/len(ot1)))
plt.xlabel('pixel number', fontsize = 16, verticalalignment='center')
plt.ylabel(r'$\beta$', fontsize = 16, horizontalalignment='center')
plt.gca().tick_params(which='major', labelsize = 14)

plt.figure()
ot1 = np.arange(g_orders[0], g_orders[-1], 5)
for i in range(len(ot1)):
    ax = plt.subplot(len(ot1),1,(i+1))
    xt1 = flpm1[mask_beta][fl[mask_beta][:,0]==ot1[i]][:,1]
    yt1 = flpm1[mask_beta][fl[mask_beta][:,0]==ot1[i]][:,3]
    st1 = flpm1[mask_beta][fl[mask_beta][:,0]==ot1[i]][:,8]
    ax.errorbar(xt1, yt1, yerr=st1, fmt='ks', ms=5)
    x2 = np.linspace(left_edge, right_edge, 1000)
    ax.plot(x2, legendre.legval2d(ot1[i]/normalizing_constant0, x2/normalizing_constant1, beta_sol2d), 'g--')
    ax.set_ylabel(r'$\beta$', fontsize = 16, horizontalalignment='center')
    ax.set_ylim((2.0, 3.0))
    ax.set_yticks([2.2, 2.5, 2.8])
    ax.tick_params(which='major', labelsize = 14)
    ax.text(0.05, 0.15, f'order = {ot1[i]}', ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.tick_params(which='major', labelsize = 14)
    if i<len(ot1)-1:
        plt.setp(ax.get_xticklabels(), visible=False)
plt.xlabel('pixel number', fontsize = 16, verticalalignment='top')

plt.figure()
XIND, YIND = np.meshgrid(np.linspace(384, 3712, 1000), g_orders)
beta_2d = legendre.legval2d(YIND/normalizing_constant0, XIND/normalizing_constant1, beta_sol2d)
im1 = plt.gca().imshow(beta_2d, extent=(384, 3712, g_orders[0], g_orders[-1]), origin='lower', aspect=40)
cbar1 = plt.gcf().colorbar(im1, extend='both', shrink=0.75, ax=plt.gca())
cbar1.ax.set_xlabel(r'$\beta$', fontsize=20)
cbar1.ax.tick_params(which='major', labelsize=18)
plt.xlabel('pixel position in the principal dispersion direction', fontsize=20, verticalalignment='top')
plt.ylabel('order number', fontsize=20, horizontalalignment='center', verticalalignment='bottom')
plt.gca().tick_params(which='major', labelsize=18)
cbar1.ax.tick_params(which='major', labelsize=18)
plt.gcf().set_size_inches(16.0, 9.0)
plt.gcf().savefig('zbeta2d.png', dpi=200)

fig = plt.figure(figsize=(10,10))
ax1 = plt.Axes(fig, [0.14, 0.58, 0.82, 0.4])
fig.add_axes(ax1)
ax2 = plt.Axes(fig, [0.14, 0.08, 0.82, 0.4])
fig.add_axes(ax2)
im1 = ax1.imshow(beta_2d, extent=(384, 3712, g_orders[0], g_orders[-1]), origin='lower', aspect=40)
im2 = ax2.imshow(sig_2d, extent=(384, 3712, g_orders[0], g_orders[-1]), origin='lower', aspect=40)
cbar1 = plt.gcf().colorbar(im1, extend='both', shrink=0.75, ax=ax1)
cbar1.ax.set_xlabel(r'$\beta$', fontsize=16)
cbar1.ax.tick_params(which='major', labelsize=14)
cbar2 = plt.gcf().colorbar(im2, extend='both', shrink=0.75, ax=ax2)
cbar2.ax.set_xlabel(r'$\sigma$', fontsize=16)
cbar2.ax.tick_params(which='major', labelsize=14)
ax1.set_xlabel('pixel position in the principal dispersion direction', fontsize=16, verticalalignment='top')
ax1.set_ylabel('order number', fontsize=16, horizontalalignment='center', verticalalignment='bottom')
ax2.set_xlabel('pixel position in the principal dispersion direction', fontsize=16, verticalalignment='top')
ax2.set_ylabel('order number', fontsize=16, horizontalalignment='center', verticalalignment='bottom')
ax1.tick_params(which='major', labelsize=14)
ax2.tick_params(which='major', labelsize=14)
cbar1.ax.tick_params(which='major', labelsize=14)
cbar2.ax.tick_params(which='major', labelsize=14)
plt.gcf().savefig('zbetasig2d.png', dpi=200)


# fit the relation for FWHM with respect to order and position
o3 = deepcopy(fl[flm, 0]) / normalizing_constant0
x3 = deepcopy(flpm2[flm,0]) / normalizing_constant1
y3 = deepcopy(flpm2[flm,4])
s3 = deepcopy(flpm2[flm,5])
FWHM_sol2d, mask_FWHM = leg2d_fit(o3, x3, y3, deg_o, deg_x, sigma_clipping=True, sigma=s3)
mask_FWHM = np.array([i in y3[mask_FWHM] for i in flpm2[:,4]])
mask_o1 = fl[:,0]<np.median(g_orders);  mask_o2 = fl[:,0]>=np.median(g_orders)
mask_FWHM_o1 = mask_o1 & mask_FWHM
mask_FWHM_o2 = mask_o2 & mask_FWHM

plt.figure()
plt.errorbar(flpm2[flm,0], flpm2[flm,4], yerr=flpm2[flm,5], fmt='rs', zorder=0)
plt.errorbar(flpm2[mask_FWHM_o1,0], flpm2[mask_FWHM_o1,4], yerr=flpm2[mask_FWHM_o1,5], fmt='bs', zorder=1)
plt.errorbar(flpm2[mask_FWHM_o2,0], flpm2[mask_FWHM_o2,4], yerr=flpm2[mask_FWHM_o2,5], fmt='gs', zorder=1)
ot1 = np.arange(g_orders[0], g_orders[-1], 10)
xt1 = np.linspace(left_edge, right_edge, 1000)
for i in range(len(ot1)):
    FWHM1 = np.zeros(shape=len(xt1), dtype=float)
    for j in range(len(xt1)):
        sig1 = legendre.legval2d(ot1[i]/normalizing_constant0, xt1[j]/normalizing_constant1, sig_sol2d)
        beta1 = legendre.legval2d(ot1[i]/normalizing_constant0, xt1[j]/normalizing_constant1, beta_sol2d)
        popt1 = [1.0, 0.0, sig1, beta1, 0.0]
        func1 = lambda x: super_gaussian(x, *popt1) - 0.5
        xL1 = root_scalar(func1, bracket=[-10.0, 0.0], method='ridder').root
        xR1 = root_scalar(func1, bracket=[0.0, 10.0], method='ridder').root
        FWHM1[j] = xR1 - xL1
    plt.plot(xt1, legendre.legval2d(ot1[i]/normalizing_constant0, xt1/normalizing_constant1, FWHM_sol2d), '--', color=plt.cm.jet(i/len(ot1)))
    plt.plot(xt1, FWHM1, '--', color='pink')
    
plt.figure()
ot1 = np.arange(g_orders[0], g_orders[-1], 5)
for i in range(len(ot1)):
    x2 = np.linspace(left_edge, right_edge, 1000)
    FWHM1 = np.zeros(shape=len(x2), dtype=float)
    for j in range(len(x2)):
        sig1 = legendre.legval2d(ot1[i]/normalizing_constant0, x2[j]/normalizing_constant1, sig_sol2d)
        beta1 = legendre.legval2d(ot1[i]/normalizing_constant0, x2[j]/normalizing_constant1, beta_sol2d)
        popt1 = [1.0, 0.0, sig1, beta1, 0.0]
        func1 = lambda x: super_gaussian(x, *popt1) - 0.5
        xL1 = root_scalar(func1, bracket=[-10.0, 0.0], method='ridder').root
        xR1 = root_scalar(func1, bracket=[0.0, 10.0], method='ridder').root
        FWHM1[j] = xR1 - xL1
    ax = plt.subplot(len(ot1),1,(i+1))
    xt1 = flpm2[mask_FWHM][fl[mask_FWHM][:,0]==ot1[i]][:,0]
    yt1 = flpm2[mask_FWHM][fl[mask_FWHM][:,0]==ot1[i]][:,4]
    st1 = flpm2[mask_FWHM][fl[mask_FWHM][:,0]==ot1[i]][:,5]
    ax.errorbar(xt1, yt1, yerr=st1, fmt='ks', ms=5)
    ax.plot(x2, legendre.legval2d(ot1[i]/normalizing_constant0, x2/normalizing_constant1, FWHM_sol2d), 'g--')
    ax.plot(x2, FWHM1, '--', color='pink')
    ax.text(0.05, 0.1, f'order = {ot1[i]}', ha='center', va='center', fontsize=10, transform=ax.transAxes)
    if i<len(ot1)-1:
        plt.setp(ax.get_xticklabels(), visible=False)

plt.figure()
XIND, YIND = np.meshgrid(np.linspace(384, 3712, 1000), g_orders)
FWHM_2d = np.zeros(shape=XIND.shape, dtype=float)
for i in range(XIND.shape[0]):
    for j in range(XIND.shape[1]):
        sig1 = legendre.legval2d(YIND[i,j]/normalizing_constant0, XIND[i,j]/normalizing_constant1, sig_sol2d)
        beta1 = legendre.legval2d(YIND[i,j]/normalizing_constant0, XIND[i,j]/normalizing_constant1, beta_sol2d)
        popt1 = [1.0, 0.0, sig1, beta1, 0.0]
        func1 = lambda x: super_gaussian(x, *popt1) - 0.5
        xL1 = root_scalar(func1, bracket=[-10.0, 0.0], method='ridder').root
        xR1 = root_scalar(func1, bracket=[0.0, 10.0], method='ridder').root
        FWHM_2d[i,j] = xR1 - xL1
im1 = plt.gca().imshow(FWHM_2d, extent=(384, 3712, g_orders[0], g_orders[-1]), origin='lower', aspect=40)
cbar1 = plt.gcf().colorbar(im1, extend='both', shrink=0.75, ax=plt.gca())
cbar1.ax.set_xlabel('FWHM', fontsize=20)
cbar1.ax.tick_params(which='major', labelsize=18)
plt.xlabel('pixel position in the principal dispersion direction', fontsize = 20, verticalalignment='top')
plt.ylabel('order number', fontsize = 20, horizontalalignment='center', verticalalignment='bottom')
plt.gca().tick_params(which='major', labelsize=18)
cbar1.ax.tick_params(which='major', labelsize=18)
plt.gcf().set_size_inches(16.0, 9.0)
plt.gcf().savefig('zFWHM2d.png', dpi=200)


mask_f = (mask_sig & mask_beta) & mask_FWHM

chunk1 = np.arange(512, 4096-512+1, 256)
order_c1 = np.array([np.percentile(g_orders, i*100/6.0) for i in [1,3,5]])
chunk_diff = chunk1[1] - chunk1[0]
order_diff = order_c1[1] - order_c1[0]
temp1, temp2 = np.meshgrid(chunk1, order_c1)
chunk = np.vstack((temp1.flatten(), temp2.flatten())).T

plt.figure()
plt.plot(np.poly1d(popt_non_rep1)(g_orders), g_orders, 'k--')
plt.plot(np.poly1d(popt_non_rep2)(g_orders), g_orders, 'k--')
rec1 = []
for i in range(len(chunk)):
    rec1.append(Rectangle((chunk[i,0]-chunk_diff/2, chunk[i,1]-order_diff/2), chunk_diff, order_diff))
    plt.plot(chunk[i,0], chunk[i,1], 'b*', ms=10, zorder=100)
plt.plot([chunk[0,0], chunk[-1,0]], [chunk[0,1], chunk[0,1]], 'b:', lw=2)
plt.plot([chunk[0,0], chunk[-1,0]], [chunk[-1,1], chunk[-1,1]], 'b:', lw=2)
plt.plot([chunk[0,0], chunk[0,0]], [chunk[0,1], chunk[-1,1]], 'b:', lw=2)
plt.plot([chunk[-1,0], chunk[-1,0]], [chunk[0,1], chunk[-1,1]], 'b:', lw=2)
plt.gca().add_collection(PatchCollection(rec1, facecolor='b', alpha=0.15))
plt.gca().add_collection(PatchCollection(rec1, facecolor='none', alpha=1, edgecolor='k', linewidth=1))
plt.errorbar(flpm1[mask_f, 1], fl[mask_f, 0], fmt='o', mfc='yellow', mec='k')
plt.plot([512, 2048, 3584], [82, 82, 82], 'rx', ms=12, mew=3, zorder=1000)
plt.plot([512, 2048, 3584], [102, 102, 102], 'rx', ms=12, mew=3, zorder=1000)
plt.plot([512, 2048, 3584], [121, 121, 121], 'rx', ms=12, mew=3, zorder=1000)
plt.xlim([-0.5, 4095.5])
plt.ylim([g_orders[0]-0.5, g_orders[-1]+0.5])
plt.gca().set_aspect(4096*9.0/16/(g_orders[-1]-g_orders[0]+1))
plt.xlabel('pixel position in the principal dispersion direction', fontsize=16, verticalalignment='top')
plt.ylabel('order number', fontsize=16, horizontalalignment='center', verticalalignment='bottom')
plt.gca().tick_params(which='major', labelsize=14)
plt.gcf().set_size_inches(18.0, 9.0)
plt.gcf().savefig('zchunk_divide.png', dpi=200)
plt.show()


psf_ref = {}
psf_ref['mask'] = {}    # for extracting the lines belong to a certain chunk from f_lines
psf_ref['exp'] = {}     # record the corresponding exposure number
psf_ref['x'] = {}
psf_ref['y'] = {}
psf_ref['sig_y'] = {}
for i in range(len(chunk)):
    psf_ref['mask'][i] = np.zeros(shape=(len(fl)), dtype=bool)
    psf_ref['exp'][i] = np.zeros(shape=(0), dtype=int)
    psf_ref['x'][i] = np.zeros(shape=(0,15), dtype=float)
    psf_ref['y'][i] = np.zeros(shape=(0,15), dtype=float)
    psf_ref['sig_y'][i] = np.zeros(shape=(0,15), dtype=float)

for i in range(len(chunk)):
    for j in range(len(fl)):
        if mask_f[j]==True and abs(flpm1[j,1]-chunk[i,0])<=chunk_diff+0.1 and abs(fl[j,0]-chunk[i,1])<=order_diff/2.0+0.1:
            x1 = np.arange(int(np.round(flpm1[j,1]))-7, int(np.round(flpm1[j,1]))+8)
            o1 = int(fl[j,0])
            x1_args = [k in x1 for k in xind]
            y1 = f[o1-order_init, x1_args]
            inten = y1.max()-y1.min();  x0 = flpm1[j,1];  background = 0.0
            sig = legendre.legval2d(o1/normalizing_constant0, x0/normalizing_constant1, sig_sol2d)
            beta = legendre.legval2d(o1/normalizing_constant0, x0/normalizing_constant1, beta_sol2d)
            par0 = [inten, x0, background]
            bounds0 = ([(0.0, x0-10.0, 0.0), 
                        (np.inf, x0+10.0, np.inf)])
                        
            def sgr(x, inten, x0, background):
                return inten * np.exp(-(abs(x-x0)/sig)**beta) + background
            popt, pcov = curve_fit(sgr, x1, y1, p0=par0, bounds=bounds0, sigma=np.ones(len(x1))*np.sqrt(np.average(y1)))
            
            temp1 = fl_nodes <= j
            temp2 = np.where(temp1==True)[0][-1]
            s1 = np.sqrt(abs(y1) + readout_noise2) / popt[0]
            rel_res1 = (y1 - sgr(x1, *popt)) / popt[0]
            psf_ref['mask'][i][j] = True
            psf_ref['exp'][i] = np.append(psf_ref['exp'][i], [temp2], axis=0)
            psf_ref['x'][i] = np.append(psf_ref['x'][i], [x1-popt[1]], axis=0)
            psf_ref['y'][i] = np.append(psf_ref['y'][i], [rel_res1], axis=0)
            psf_ref['sig_y'][i] = np.append(psf_ref['sig_y'][i], [s1], axis=0)
            
        else:
            psf_ref['mask'][i][j] = False
    
    print (i, chunk[i,0], chunk[i,1], len(psf_ref['x'][i]))


xb = -10.0;  xe = 10.0;  deg2 = 3;  nknots = 21
# knots = np.linspace(xb, xe, nknots+2)[1:-1]
knots = np.array([-8.75, -7.5, -6.25, -5.25, -4.5, -3.75, -3.0, -2.25, -1.5, -0.75, 0.0, 0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 6.25, 7.5, 8.75])
kns_b = np.ones(shape=(deg2+1), dtype=float)*xb
kns_e = np.ones(shape=(deg2+1), dtype=float)*xe
KNS = np.concatenate((kns_b, knots, kns_e), axis=0)

def window(x):
    """
    the PSF is less than 0.05 times intensity when its distance is 5 pixels from center
               less than 0.03 times intensity when its distance is 5.5 pixels from center
    
    """
    y = np.zeros(shape=len(x), dtype=float)
    for i in range(len(x)):
        if x[i]>=-5 and x[i]<=5:
            y[i] = 1.0
        elif x[i]>=-7.5 and x[i]<-5:
            y[i] = (x[i]-(-7.5))/2.5
            # y[i] = 1.0
        elif x[i]<-7.5:
            y[i] = 0.0
        elif x[i]>5 and x[i]<=7.5:
            y[i] = (7.5-x[i])/2.5
            # y[i] = 1.0
        elif x[i]>7.5:
            y[i] = 0.0
        else:
            assert 0
    return y

fig = plt.figure(figsize=(10,10))
subfig_left = np.array([0.08, 0.4, 0.72])
subfig_bottom = np.array([0.83, 0.64, 0.45, 0.26, 0.07])
tck_coeffs1 = np.zeros(shape=(0, nknots+deg2+1), dtype=float)
flag1 = 0
flag2 = 1
for i in range(len(chunk)):
        
    arg1 = psf_ref['x'][i].flatten().argsort()
    x1 = psf_ref['x'][i].flatten()[arg1]
    y1 = psf_ref['y'][i].flatten()[arg1] * window(x1)
    x1 = np.concatenate((np.linspace(xb, x1[0]-1e-2, len(arg1)//10), x1, np.linspace(x1[-1]+1e-2, xe, len(arg1)//10)))
    y1 = np.concatenate((np.zeros(shape=len(arg1)//10), y1, np.zeros(shape=len(arg1)//10)))
    s1 = psf_ref['sig_y'][i].flatten()[arg1]
    s1 = np.concatenate((np.average(s1)*np.ones(shape=len(arg1)//10), s1, np.average(s1)*np.ones(shape=len(arg1)//10)))
    ym1 = median_filter(y1, size=len(y1)//20)
    res1 = y1 - ym1
    rms1 = rmsd(y1, ym1)
    mask1 = abs(res1) > 3.0*rms1
    x2 = x1[~mask1]; y2 = y1[~mask1]; s2 = s1[~mask1]
    ya2 = uniform_filter1d(y2, size=len(y2)//20)
        
    splines2 = LSQUnivariateSpline(x2, y2, t=knots, bbox=[xb, xe], k=deg2)
    tck_coeffs1 = np.append(tck_coeffs1, [splines2.get_coeffs()], axis=0)
    
    ax1 = plt.Axes(fig, [subfig_left[flag1%3], subfig_bottom[flag1//3], 0.25, 0.13])
    fig.add_axes(ax1)
    fig2, ax2 = plt.subplots()
    ax1.errorbar(x1[mask1], y1[mask1], yerr=s1[mask1], fmt='r.', ms=6, zorder=0, alpha=0.5)
    ax1.errorbar(x2, y2, yerr=s2, fmt='.', color='lightblue', ms=6, zorder=1, alpha=0.3)
    ax2.errorbar(x1[mask1], y1[mask1], yerr=s1[mask1], fmt='r.', ms=6, zorder=0, alpha=0.5)
    ax2.errorbar(x2, y2, yerr=s2, fmt='.', color='lightblue', ms=6, zorder=1, alpha=0.3)
        
    tck1 = (KNS, splines2.get_coeffs(), deg2)
    ax1.plot(np.linspace(xb, xe, 1000), splev(np.linspace(xb, xe, 1000), tck1), color='g', label=f'{int(chunk[i,0])}', zorder=10, lw=3)
    ax1.set_ylim([-0.1, 0.1])
    ax1.set_title(f'x={int(chunk[i,0])}', ha='center', va='center', fontsize=12)
        
    ax2.plot(np.linspace(xb, xe, 1000), splev(np.linspace(xb, xe, 1000), tck1), color='g', label=f'{int(chunk[i,0])}', zorder=10, lw=3)
    ax2.set_ylim([-0.1, 0.1])
    ax2.set_title(f'x={int(chunk[i,0])}', ha='center', va='center', fontsize=14)
    ax2.set_xlabel("x' (pixel position relative to the IP center)", va='top', fontsize=14)
    ax2.set_ylabel('normalized residuals', va='bottom', ha='center', fontsize=14)
    ax2.tick_params(which='major', labelsize=12)
    fig2.set_size_inches(8.0, 6.0)
    fig2.savefig(f'zn{chunk[i,1]:.2f}_{int(chunk[i,0])}.png', dpi=100)

    flag1 = flag1 + 1

    if i<len(chunk)-1 and chunk[i+1,0]<chunk[i,0]:
        fig.text(0.53, 0.03, "x' (pixel position relative to the IP center)", ha='center', va='center', fontsize=14)
        fig.text(0.02, 0.52, 'normalized residuals', ha='center', va='center', rotation='vertical', fontsize=14)
        fig.savefig(f'za{flag2}.png', dpi=300)
        fig = plt.figure(figsize=(10,10))
        flag1 = 0
        flag2 = flag2 + 1
    
fig.text(0.53, 0.03, "x' (pixel position relative to the IP center)", ha='center', va='center', fontsize=14)
fig.text(0.02, 0.52, 'normalized residuals', ha='center', va='center', rotation='vertical', fontsize=14)
fig.savefig(f'za{flag2}.png', dpi=300)

    
function1 = []
for k in range(tck_coeffs1.shape[1]):
    function1.append(interp2d(chunk1, order_c1, tck_coeffs1[:,k].reshape(len(order_c1), len(chunk1)), kind='linear'))

xp1 = np.linspace(np.min(chunk[:,0]), np.max(chunk[:,0]), 1000)
yp1 = np.linspace(np.min(g_orders), np.max(g_orders), 100)
tck_coeffs_p1 = np.zeros(shape=(len(yp1), len(xp1), tck_coeffs1.shape[1]), dtype=float)
for i in range(len(yp1)):
    for j in range(len(xp1)):
        for k in range(tck_coeffs1.shape[1]):
            tck_coeffs_p1[i,j,k] = function1[k](xp1[j], yp1[i])

tck_coeffs_farray = []
for k in range(tck_coeffs1.shape[1]):
    tck_coeffs_farray.append(interp2d(xp1, yp1, tck_coeffs_p1[:,:,k], kind='linear', bounds_error=False))

    
plt.figure()
xpt1 = np.linspace(np.min(chunk[:,0])-256, np.max(chunk[:,0])+256, 3000)
ypt1 = np.linspace(g_orders[0], g_orders[-1], 100)
COEFFST1 = np.zeros(shape=(len(ypt1), len(xpt1)), dtype=float)
for i in range(len(ypt1)):
    for j in range(len(xpt1)):
        COEFFST1[i,j] = tck_coeffs_farray[0](xpt1[j], ypt1[i])
plt.imshow(COEFFST1, extent=(xpt1[0], xpt1[-1], ypt1[0], ypt1[-1]), origin='lower', aspect=40)
plt.show()
    
    
psf = {}

psf['line_list'] = np.zeros(shape=(0,3), dtype=float)
for i in range(len(chunk)):
    temp1 = np.vstack((fl[psf_ref['mask'][i],0], flpm1[psf_ref['mask'][i],1], fl[psf_ref['mask'][i],3])).T
    psf['line_list'] = np.append(psf['line_list'], temp1, axis=0)
psf['line_list'] = np.unique(psf['line_list'], axis=0)

psf['model'] = 'super-Gaussian + residual'
psf['sig'] = sig_sol2d
psf['beta'] = beta_sol2d
psf['normalizing_constant_o'] = normalizing_constant0
psf['normalizing_constant_x'] = normalizing_constant1
psf['residual'] = {}
psf['residual']['knots'] = KNS
psf['residual']['deg'] = deg2
psf['residual']['coeffs'] = tck_coeffs_farray
psf['residual']['chunk'] = chunk
pickle.dump(psf, open('IP_20170922_br_3rows.p', "wb"))
    


