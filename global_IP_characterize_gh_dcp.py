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
from scipy.integrate import quad
from scipy.special import hermite
from funcs import rmsd, CDG_fit, CDG_param1, moffat_fit, moffat_param1, super_gaussian_fit, super_gaussian_param1, \
                  leg2d_fit, gauss_hermite_fit, cheby_fit, cheby_ev
from scipy.stats import norm
from numpy.polynomial import legendre
import pickle
from refractive_index import refractive_index_moist_air
from copy import deepcopy
from functools import reduce

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

def coef(i):
    if i==0 or i==1:
        return 1.0/0.3989422804014329
    else:
        return 1.0/0.3989422804014329 * 1.0/reduce(int.__mul__, range(2*i-2, 0, -4))

def gauss_hermite(x, sig, *An):
    try:
        x = float(x)
        x = np.array([x])
        flag = 1
    except:
        x = np.array(x)
        flag = 2
    
    y = np.zeros(shape=len(x), dtype=float)
    for i in range(len(An)):
        y = y + An[i] * 1.0/(2.0*np.pi*sig) * coef(i) * hermite(i)(x/sig) * np.exp(-x**2/(2*sig**2))
    
    if flag == 1:
        return y[0]
    elif flag == 2:
        return y

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
        elif x[i]<-7.5:
            y[i] = 0.0
        elif x[i]>5 and x[i]<=7.5:
            y[i] = (7.5-x[i])/2.5
        elif x[i]>7.5:
            y[i] = 0.0
        else:
            assert 0
    return y
    
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
    # 0order, 1line_center, 2wav, 3type
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
                else:
                    f_lines_3_mask[i] = False
                    print (f'{f_lines_3[i,3]} cannot pass the r^2 check')
            except:
                f_lines_3_mask[i] = False
                print (f'{f_lines_3[i,3]} cannot be fitted by super-Gaussian function')
                
    f_lines[h] = f_lines_3
    f_lines_mask[h] = f_lines_3_mask
    f_lines_param1[h] = f_lines_3_param1
                

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
for h in range(len(filename)):
    fl = np.append(fl, f_lines[h], axis=0)
    flm = np.append(flm, f_lines_mask[h], axis=0)
    flpm1 = np.append(flpm1, f_lines_param1[h], axis=0)
fl_nodes = np.array([0], dtype=int)
for i in f_lines.keys():
    fl_nodes = np.append(fl_nodes, [fl_nodes[-1]+len(f_lines[i])], axis=0)
    
    
deg_o = 1;  deg_x = 3

sg = {}
o1 = deepcopy(fl[flm, 0]) / normalizing_constant0
x1 = deepcopy(flpm1[flm, 1]) / normalizing_constant1
y1sig = deepcopy(flpm1[flm, 2])
y1beta = deepcopy(flpm1[flm, 3])
s1sig = deepcopy(flpm1[flm, 7])
s1beta = deepcopy(flpm1[flm, 8])
sg['sig'], _ = leg2d_fit(o1, x1, y1sig, deg_o, deg_x, sigma_clipping=True, sigma=s1sig)
sg['beta'], _ = leg2d_fit(o1, x1, y1beta, deg_o, deg_x, sigma_clipping=True, sigma=s1beta)
pickle.dump(sg, open('IP_20170922_sg.p', "wb"))


# characterize the IP with gauss_hermite model
N_gh = 4
deg_o = 1;  deg_x = 3
f_lines_gh_param1 = {}
f_lines_gh_mask1 = {}
for h in range(len(filename)):
    f_lines_gh_param1[h] = np.zeros(shape=(len(f_lines[h]), (1+N_gh+1)*2), dtype=float)
    f_lines_gh_mask1[h] = np.ones(shape=(len(f_lines[h])), dtype=bool)
    for i in range(len(f_lines[h])):
        if f_lines_mask[h][i]==True:
            try:
                x1 = np.arange(int(np.round(f_lines_param1[h][i,1]))-7, int(np.round(f_lines_param1[h][i,1]))+8)
                o1 = int(f_lines[h][i,0])
                x1_args = [k in x1 for k in xind]
                y1 = f[o1-order_init, x1_args]
                inten = y1.max()-y1.min();  x0 = f_lines_param1[h][i,1];  background = 0.0
                sig = legendre.legval2d(o1/normalizing_constant0, x0/normalizing_constant1, sg['sig'])
                beta = legendre.legval2d(o1/normalizing_constant0, x0/normalizing_constant1, sg['beta'])
                par0 = [inten, x0, background]
                bounds0 = ([(0.0, x0-10.0, 0.0), 
                            (np.inf, x0+10.0, np.inf)])
                            
                def sgr(x, inten, x0, background):
                    return inten * np.exp(-(abs(x-x0)/sig)**beta) + background
                popt, pcov = curve_fit(sgr, x1, y1, p0=par0, bounds=bounds0, sigma=np.ones(len(x1))*np.sqrt(np.average(y1)))
                
                x2 = x1 - popt[1]
                y2 = (y1 - popt[2]) / popt[0]
                popt, pcov = gauss_hermite_fit(x2, y2, n=N_gh)
                pcov_diag = np.array([np.sqrt(pcov[j,j]) for j in range(1+N_gh+1)])
                f_lines_gh_param1[h][i,:] = np.concatenate((popt, pcov_diag))

                ess = ((y2 - np.average(y2))**2).sum()
                rss = ((y2 - gauss_hermite(x2, *popt))**2).sum()
                r_squared = 1 - (rss/(len(y2)-len(popt)-1)) / (ess/(len(y2)-1))
                print (f'gauss-hermite: {f_lines[h][i,3]}, r_squared = {r_squared:6f}')
                # 0x0, 1sig, 2background, 3-7An

            except:
                print (f'{f_lines[h][i,3]} cannot be fit with the gauss-hermite model')
                f_lines_gh_mask1[h][i] = False
                pass
        else:
            f_lines_gh_mask1[h][i] = False
            pass

flpm2 = np.zeros(shape=(0,(1+N_gh+1)*2), dtype=float)
flm2 = np.zeros(shape=(0), dtype=bool)
for h in range(len(filename)):
    flpm2 = np.append(flpm2, f_lines_gh_param1[h], axis=0)
    flm2 = np.append(flm2, f_lines_gh_mask1[h], axis=0)

gh = {}
mask1 = flm & flm2
o2 = deepcopy(fl[mask1, 0]) / normalizing_constant0
x2 = deepcopy(flpm1[mask1, 1]) / normalizing_constant1
y2 = deepcopy(flpm2[mask1, 0])
gh['sig'], mask2d = leg2d_fit(o2, x2, y2, deg_o, deg_x, sigma_clipping=True)
res2 = y2[mask2d] - legendre.legval2d(o2[mask2d], x2[mask2d], gh['sig'])
print ('sig', np.average(abs(y2[mask2d])), np.average(abs(res2)), len(y2[mask2d])/len(y2))
gh['An'] = []
for i in range(N_gh+1):
    y2 = deepcopy(flpm2[mask1, 1+i])
    temp1, mask2d = leg2d_fit(o2, x2, y2, deg_o, deg_x, sigma_clipping=True)
    gh['An'].append(temp1)
    res2 = y2[mask2d] - legendre.legval2d(o2[mask2d], x2[mask2d], temp1)
    print (i, np.average(abs(y2[mask2d])), np.average(abs(res2)), len(y2[mask2d])/len(y2))
pickle.dump(gh, open('IP_20170922_gh.p', "wb"))


# characterize the IP with Chebyshev model
N_cheby = 8
deg_o = 1;  deg_x = 3
f_lines_cheby_param1 = {}
f_lines_cheby_mask1 = {}
for h in range(len(filename)):
    f_lines_cheby_param1[h] = np.zeros(shape=(len(f_lines[h]), N_cheby+1), dtype=float)
    f_lines_cheby_mask1[h] = np.ones(shape=(len(f_lines[h])), dtype=bool)
    for i in range(len(f_lines[h])):
        if f_lines_mask[h][i]==True:
            try:

                x1 = np.arange(int(np.round(f_lines_param1[h][i,1]))-7, int(np.round(f_lines_param1[h][i,1]))+8)
                o1 = int(f_lines[h][i,0])
                x1_args = [k in x1 for k in xind]
                y1 = f[o1-order_init, x1_args]
                inten = y1.max()-y1.min();  x0 = f_lines_param1[h][i,1];  background = 0.0
                sig = legendre.legval2d(o1/normalizing_constant0, x0/normalizing_constant1, sg['sig'])
                beta = legendre.legval2d(o1/normalizing_constant0, x0/normalizing_constant1, sg['beta'])
                par0 = [inten, x0, background]
                bounds0 = ([(0.0, x0-10.0, 0.0), 
                            (np.inf, x0+10.0, np.inf)])
                            
                def sgr(x, inten, x0, background):
                    return inten * np.exp(-(abs(x-x0)/sig)**beta) + background
                popt, pcov = curve_fit(sgr, x1, y1, p0=par0, bounds=bounds0, sigma=np.ones(len(x1))*np.sqrt(np.average(y1)))
                
                x2 = x1 - popt[1]
                y2 = (y1 - popt[2]) / popt[0]
                T = cheby_fit(x2, y2, N_cheby)
                fun2 = interp1d(x2, y2, kind='quadratic', bounds_error=False, fill_value=(0.0, 0.0))
                xIP = np.arange(len(x2)) - len(x2)//2
                yIP = fun2(xIP)
                TIP = cheby_fit(xIP, yIP, N_cheby)
                f_lines_cheby_param1[h][i,:] = TIP
                
                ess = ((y2 - np.average(y2))**2).sum()
                rss = ((y2 - cheby_ev(x2, x2, T))**2).sum()
                r_squared = 1 - (rss/(len(y2)-(N_cheby+1)-1)) / (ess/(len(y2)-1))
                print (f'cheby: {f_lines[h][i,3]}, r_squared = {r_squared:6f}')
            
            except:
                print (f'{f_lines[h][i,3]} cannot be expanded with the chebyshev basis')
                f_lines_cheby_mask1[h][i] = False
                pass
        
        else:
            f_lines_cheby_mask1[h][i] = False
            pass

flpm4 = np.zeros(shape=(0, N_cheby+1), dtype=float)
flm4 = np.zeros(shape=(0), dtype=bool)
for h in range(len(filename)):
    flpm4 = np.append(flpm4, f_lines_cheby_param1[h], axis=0)
    flm4 = np.append(flm4, f_lines_cheby_mask1[h], axis=0)

cheby = {}
cheby['T'] = []
mask1 = flm & flm4
o4 = deepcopy(fl[mask1, 0]) / normalizing_constant0
x4 = deepcopy(flpm1[mask1, 1]) / normalizing_constant1
for i in range(N_cheby+1):
    y4 = deepcopy(flpm4[mask1, i])
    temp1, mask2d = leg2d_fit(o4, x4, y4, deg_o, deg_x, sigma_clipping=True)
    cheby['T'].append(temp1)
    res4 = y4[mask2d] - legendre.legval2d(o4[mask2d], x4[mask2d], temp1)
    print (i, np.average(abs(y4[mask2d])), np.average(abs(res4)), len(y4[mask2d])/len(y4))
cheby['x'] = xIP
pickle.dump(cheby, open('IP_20170922_cheby.p', "wb"))
plt.show()