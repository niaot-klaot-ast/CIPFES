#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit, root_scalar, minimize
from scipy.interpolate import splev, interp1d, LSQUnivariateSpline
from numpy.polynomial import legendre
from funcs import rmsd, cheby_ev
from copy import deepcopy
from matplotlib.patches import Rectangle
import pickle
    

speed_of_light = 299792458.0
f_0 = 9.52                            # GHz
f_rep = 25.0                          # GHz
readout_noise2 = 15.0 * 5.0**2
# 以上是基本信息输入区

def linear_gaussian(x, a, x0, sigma, background, slope):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + background + slope*x
# 以上是函数定义区


np.set_printoptions(threshold=100000)
input_filename = 'a201709220026.fits'
data_oct = pf.getdata(input_filename, header=False)
order_init = 77

wavelength_solution_filename = 'wavelength_solution_a201709220026.p'
ws = pickle.load(open(wavelength_solution_filename, 'rb'))
ws_orders = np.unique(ws['line_list'][:,0]).astype(int)
normalizing_constant0 = ws['normalizing_constant_o']
normalizing_constant1 = ws['normalizing_constant_x']
wav_sol2d_1 = ws['solution']
dodo = ws['domain_of_definition_o']
dodx = ws['domain_of_definition_x']

psf_filename = 'IP_20170922_br_3rows.p'
psf = pickle.load(open(psf_filename, "rb"))
line_list = psf['line_list']
normalizing_constant0 = psf['normalizing_constant_o']
normalizing_constant1 = psf['normalizing_constant_x']
res_chunk = psf['residual']['chunk']

ip_filename = 'IP_20170922_cheby.p'
cheby = pickle.load(open(ip_filename, "rb"))
T_sol2d = cheby['T']

deg_leg = 3

z = np.zeros(shape=(0,5), dtype=float)
zt = np.zeros(shape=(0,8), dtype=float)
chunk_nodes_init = np.arange(0, data_oct.shape[1], 30)

flag = 0
for order in range(len(data_oct)):

    if order+order_init in np.arange(82, 125) and order+order_init>=dodo[0] and order+order_init<=dodo[1]:
        
        x = np.arange(data_oct.shape[1])
        y = data_oct[order, x]

        xind = np.arange(512-128, 4096-(512-128)+1)
        c1 = chunk_nodes_init[(xind[0] - chunk_nodes_init)<0][0]
        c2 = chunk_nodes_init[(xind[-1] - chunk_nodes_init)>0][-1]
        chunk = chunk_nodes_init[np.where(chunk_nodes_init==c1)[0][0]:np.where(chunk_nodes_init==c2)[0][0]+1]
        
        left_wav = legendre.legval2d((order+order_init)/normalizing_constant0, chunk[0]/normalizing_constant1, wav_sol2d_1)/((order+order_init)/normalizing_constant0)
        right_wav = legendre.legval2d((order+order_init)/normalizing_constant0, chunk[-1]/normalizing_constant1, wav_sol2d_1)/((order+order_init)/normalizing_constant0)
        low_n = int(np.floor((speed_of_light/left_wav*10.0 - f_0) / f_rep)) - 1
        high_n = int(np.ceil((speed_of_light/right_wav*10.0 - f_0) / f_rep)) + 1
        line_n_order = np.arange(low_n, high_n+1)
        line_wav_order = speed_of_light / (f_0 + f_rep*line_n_order) * 10.0
        def wav2pos(x, o, w):
            return legendre.legval2d(o/normalizing_constant0, x/normalizing_constant1, wav_sol2d_1)/(o/normalizing_constant0) - w
        line_order = np.zeros(shape=(0,3), dtype=float)
        for i in range(len(line_wav_order)):
            xtemp1 = root_scalar(wav2pos, args=(order+order_init, line_wav_order[i]), bracket=[0, 4095], method='ridder').root
            line_order = np.append(line_order, [[line_n_order[i], line_wav_order[i], xtemp1]], axis=0)
            # 0n, 1wav, 2line_center_(ThAr_solution)
        
        # plt.figure()
        # plt.plot(x, y)
        # plt.plot(line_order[:,2], np.zeros(shape=len(line_order)), 'gx', ms=6)
        # plt.show()
        
        # 1. use Gaussian function to fit the comb teeth
        z1 = np.zeros(shape=(0,9), dtype=float)
    #--------------------------------------------------------------------1
        for i in range(1, len(line_order)-1):                                   
            
            x0 = line_order[i,2]
            lb = int(np.ceil((line_order[i,2] + line_order[i-1,2]) / 2.0))
            rb = int(np.floor((line_order[i,2] + line_order[i+1,2]) / 2.0))
            x1 = np.arange(lb, rb+1)
            y1 = y[x1]
            
            slope = (y1[-1]-y1[0])/(x1[-1]-x1[0]);  background = (y1[-1]+y1[0])/2.0 - slope*x0
            a = y1.max()-y1.min();  sigma = 2.25
            par0 = [a, x0, sigma, background, slope]
            bounds0 = ([(0.0, x0-10.0, 0.0, -np.inf, -np.inf), (np.inf, x0+10.0, np.inf, np.inf, np.inf)])
            
            try:
                popt, pcov = curve_fit(linear_gaussian, x1, y1, p0=par0, bounds=bounds0, sigma=np.ones(len(x1))*np.sqrt(np.average(y1) + readout_noise2))
                z1 = np.append(z1, [[order+order_init, line_order[i,0], line_order[i,1], 1.0, *popt]], axis=0)
                # 0order, 1n, 2wav, 3flag, 4-8popt
            except:
                z1 = np.append(z1, [[order+order_init, line_order[i,0], line_order[i,1], 0.0, *np.zeros(shape=(5), dtype=float)]], axis=0)
        
        mask1 = (z1[:,3] == 1)
        x1 = z1[mask1, 1]
        y1 = z1[mask1, 5]
        popt1 = np.polyfit(x1, y1, 10)
        res1 = y1 - np.poly1d(popt1)(x1)
        rms1 = rmsd(y1, np.poly1d(popt1)(x1))
        mask2 = abs(res1) > 3.0*rms1
        x2 = x1[~mask2]
        y2 = y1[~mask2]
        popt_nx = np.polyfit(x2, y2, 10)
        line_order = np.vstack((line_order.T, np.poly1d(popt_nx)(line_order[:,0]))).T
        # 0n, 1wav, 2line_center_(ThAr_solution), 3line_center_(Gaussian_solution)
        
        # plt.figure()
        # plt.plot(x1, y1 - np.poly1d(popt_nx)(x1), 'rs')
        # plt.plot(x2, y2 - np.poly1d(popt_nx)(x2), 'ks')
        # plt.show()
        
        
        for i in range(len(chunk)-1):
            
            # try:
                # j1 = np.where(line_order[:,3] < chunk[i])[0][-1]
            # except:
                # j1 = np.where(line_order[:,3] < chunk[i]+0.5)[0][-1]
                
            # try:
                # j2 = np.where(line_order[:,3] > chunk[i+1])[0][0]
            # except:
                # j2 = np.where(line_order[:,3] > chunk[i+1]-0.5)[0][0]

            j1 = np.where(line_order[:,3] < chunk[i])[0][-1]
            j2 = np.where(line_order[:,3] > chunk[i+1])[0][0]
            n1 = int(line_order[j1, 0])
            n2 = int(line_order[j2, 0])
            delta_n = n2 - n1
            
            T = []
            for j in range(len(T_sol2d)):
                T.append(legendre.legval2d((order+order_init)/normalizing_constant0, 0.5*(chunk[i]+chunk[i+1])/normalizing_constant1, T_sol2d[j]))
                
            def sgr(x, inten, x0):
                
                try:
                    x = float(x)
                    x = np.array([x])
                    flag = 1
                except:
                    x = np.array(x)
                    flag = 2
                
                y = np.zeros(shape=len(x), dtype=float)
                mask1 = (x-x0 >= -7.0) & (x-x0 <= 7.0)
                y[mask1] = cheby_ev(x[mask1]-x0, cheby['x'], T)
                y[~mask1] = 0.0
                
                if flag == 1:
                    return inten * y[0]
                elif flag == 2:
                    return inten * y
            
            x1 = np.arange(chunk[i], chunk[i+1]+1)
            y1 = y[x1]
            popt_temp1 = np.polyfit([x1[0], x1[-1]], [-1, 1], 1)  # convert to the definitive domain of Legrandre polynomial
            x2 = np.poly1d(popt_temp1)(x1)
            
            def f_chunk(x, *popt):
                inten = popt[:(delta_n+1)]
                x0 = popt[(delta_n+1):2*(delta_n+1)]
                c = popt[2*(delta_n+1):]
                
                y = 0.0
                for j in range(len(inten)):
                    y = sgr(x, inten[j], x0[j]) + y
                y = legendre.legval(np.poly1d(popt_temp1)(x), c) + y
                return y
            
            # solve the linear system
            X = np.zeros(shape=(len(x1), delta_n+1+deg_leg+1), dtype=float)
            Y = np.zeros(shape=len(y1), dtype=float)
            for j in range(len(x1)):
                for k in range(delta_n+1):
                    X[j,k] = sgr(x1[j], 1.0, line_order[j1+k,3])
                for k in range(delta_n+1, delta_n+1+deg_leg+1):
                    c1 = np.zeros(shape=(k - delta_n), dtype=float)
                    c1[-1] = 1.0
                    X[j,k] = legendre.legval(x2[j], c1)
                Y[j] = y1[j]
            
            sol1 = np.linalg.lstsq(X, Y, rcond=-1)[0]

            popt = np.concatenate((abs(sol1[:delta_n+1]), line_order[j1:j2+1,3], sol1[delta_n+1:]))
                
            temp1 = np.vstack(((order+order_init)*np.ones(shape=(delta_n-1)), line_order[j1+1:j2, 0], line_order[j1+1:j2, 1], 
                                popt[1:delta_n], popt[delta_n+1+1:2*(delta_n+1)-1]))
            z = np.append(z, temp1.T, axis=0)
            # 0order, 1n, 2wav, 3intensity, 4center

            yf = f_chunk(x1, *popt)
            rss = ((y1 - yf)**2).sum()
            ess = ((y1 - np.average(y1))**2).sum()
            r_squared = 1 - (rss/(len(y1)-len(popt)-1)) / (ess/(len(y1)-1))
            rms = rmsd(y1, yf)
            red_chi_squared = ((y1 - yf)**2/(abs(y1) + readout_noise2)).sum() / (len(y1)-len(popt))

            eps1 = 0.02
            eps2 = 0.017
            eps3 = 0.014
            eps4 = 0.011
            xi = [-1, 1]

            s_err1 = np.zeros(shape=len(x1), dtype=float)
            s_err2 = np.zeros(shape=len(x1), dtype=float)
            s_err3 = np.zeros(shape=len(x1), dtype=float)
            s_err4 = np.zeros(shape=len(x1), dtype=float)
            for j in range(len(x1)):
                temp_arg1 = abs(x1[j] - popt[delta_n+1:2*(delta_n+1)]).argsort()[0]
                if abs(x1[j] - popt[delta_n+1+temp_arg1])<=7.5:
                    s_err1[j] = np.random.normal(0, eps1 * popt[temp_arg1])
                    s_err2[j] = np.random.normal(0, eps2 * popt[temp_arg1])
                    s_err3[j] = np.random.normal(0, eps3 * popt[temp_arg1])
                    s_err4[j] = np.random.normal(0, eps4 * popt[temp_arg1])
                temp_arg2 = abs(x1[j] - popt[delta_n+1:2*(delta_n+1)]).argsort()[1]
                if abs(x1[j] - popt[delta_n+1+temp_arg2])<=7.5:
                    s_err1[j] = s_err1[j] + np.random.normal(0, eps1 * popt[temp_arg2])
                    s_err2[j] = s_err2[j] + np.random.normal(0, eps2 * popt[temp_arg2])
                    s_err3[j] = s_err3[j] + np.random.normal(0, eps3 * popt[temp_arg2])
                    s_err4[j] = s_err4[j] + np.random.normal(0, eps4 * popt[temp_arg2])
                s_err1[j] = s_err1[j] + np.random.normal(0, np.sqrt(abs(y1[j]) + readout_noise2))
                s_err2[j] = s_err2[j] + np.random.normal(0, np.sqrt(abs(y1[j]) + readout_noise2))
                s_err3[j] = s_err3[j] + np.random.normal(0, np.sqrt(abs(y1[j]) + readout_noise2))
                s_err4[j] = s_err4[j] + np.random.normal(0, np.sqrt(abs(y1[j]) + readout_noise2))

            s_rms1 = rmsd(s_err1, np.zeros(shape=len(x1)))
            s_rms2 = rmsd(s_err2, np.zeros(shape=len(x1)))
            s_rms3 = rmsd(s_err3, np.zeros(shape=len(x1)))
            s_rms4 = rmsd(s_err4, np.zeros(shape=len(x1)))

            zt = np.append(zt, [[order+order_init, i, 0.5*(x1[0]+x1[-1]), rms, s_rms1, s_rms2, s_rms3, s_rms4]], axis=0)
            # 0order, 1chunk_index, 2chunk_center, 3-7rms


            if order+order_init==100 and popt[delta_n+1+1]>2000 and flag==0 and i%2==0:
                norm_const = np.max(y[np.where(x==2000)[0][0]:np.where(x==2250)[0][0]])
                fig = plt.figure(figsize=(16, 4.5))
                ax1 = plt.Axes(fig, [0.07, 0.37, 0.89, 0.59])
                fig.add_axes(ax1)
                ax2 = plt.Axes(fig, [0.07, 0.12, 0.89, 0.25])
                fig.add_axes(ax2)
                flag = 1
                xb = x1[0]

            if order+order_init==100 and popt[delta_n+1+1]>2250 and flag==1 and i%2==0:
                ax1.set_xlim([xb, xe])
                ax1.set_ylim([0, 1.15])
                ax1.set_xticklabels([])
                fig.text(0.02, 0.47, 'normalized CCD counts', va='center', rotation='vertical', fontsize = 14)
                ax1.tick_params(which='major', labelsize = 12)
                ax2.set_xlim([xb, xe])
                ax2.plot([xb, xe], [0, 0], 'k--')
                ax2.set_xlabel('pixel position x', fontsize = 14, verticalalignment='top')
                ax2.set_yticks([-0.02, 0, 0.02])
                ax2.set_ylim([-0.04, 0.04])
                ax2.tick_params(which='major', labelsize = 12)
                fig.savefig('zfit1_cheby.png', dpi=200)
                # plt.show()
                flag = 2

            if flag==1:
                x1temp = np.linspace(x1[0], x1[-1], 300)
                if i%2==0:
                    ax1.fill_between(x1, y1=1.2, y2=-0.05, color='k', alpha=0.2)
                    ax2.fill_between(x1, y1=0.04, y2=-0.04, color='k', alpha=0.2)
                ax1.plot(x1, y1/norm_const, 'ko')
                ax1.plot(x1temp, f_chunk(x1temp, *popt)/norm_const, 'g-')
                ax2.plot(x1, (y1-yf)/norm_const, 'ko')
                xe = x1[-1]
                ax1.text(0.5*(x1[0]+x1[-1]), 1.07, ''.join(['rms = ', f'{rms/norm_const:.5f}']), fontsize=12, ha='center', va='center')
                    
            
            print (order+order_init, 0.5*(x1[0]+x1[-1]), rms/y1.max(), s_rms1/rms, s_rms2/rms, s_rms3/rms, s_rms4/rms)


fig = plt.figure(figsize=(16,4))
ax1 = plt.Axes(fig, [0.06, 0.15, 0.2, 0.82])
fig.add_axes(ax1)
ax2 = plt.Axes(fig, [0.3, 0.15, 0.2, 0.82])
fig.add_axes(ax2)
ax3 = plt.Axes(fig, [0.54, 0.15, 0.2, 0.82])
fig.add_axes(ax3)
ax4 = plt.Axes(fig, [0.78, 0.15, 0.2, 0.82])
fig.add_axes(ax4)

m1 = np.median(zt[:,4]/zt[:,3])
m2 = np.median(zt[:,5]/zt[:,3])
m3 = np.median(zt[:,6]/zt[:,3])
m4 = np.median(zt[:,7]/zt[:,3])

n, bins, patches = ax1.hist(zt[:,4]/zt[:,3], np.linspace(0, np.percentile(zt[:,4]/zt[:,3], 99), 51))
ax1.plot([m1, m1], [0, np.max(n)*1.1], 'k--', lw=2)
ax1.set_ylim([0, np.max(n)*1.1])
ax1.text(0.96, 0.9, r'$\epsilon=0.02$', va='center', ha='right', fontsize=10, transform=ax1.transAxes)
ax1.text(0.96, 0.85, f'Median: {m1:.2f}', va='center', ha='right', fontsize=10, transform=ax1.transAxes)

n, bins, patches = ax2.hist(zt[:,5]/zt[:,3], np.linspace(0, np.percentile(zt[:,5]/zt[:,3], 99), 51))
ax2.plot([m2, m2], [0, np.max(n)*1.1], 'k--', lw=2)
ax2.set_ylim([0, np.max(n)*1.1])
ax2.text(0.96, 0.9, r'$\epsilon=0.017$', va='center', ha='right', fontsize=10, transform=ax2.transAxes)
ax2.text(0.96, 0.85, f'Median: {m2:.2f}', va='center', ha='right', fontsize=10, transform=ax2.transAxes)

n, bins, patches = ax3.hist(zt[:,6]/zt[:,3], np.linspace(0, np.percentile(zt[:,6]/zt[:,3], 99), 51))
ax3.plot([m3, m3], [0, np.max(n)*1.1], 'k--', lw=2)
ax3.set_ylim([0, np.max(n)*1.1])
ax3.text(0.96, 0.9, r'$\epsilon=0.014$', va='center', ha='right', fontsize=10, transform=ax3.transAxes)
ax3.text(0.96, 0.85, f'Median: {m3:.2f}', va='center', ha='right', fontsize=10, transform=ax3.transAxes)

n, bins, patches = ax4.hist(zt[:,7]/zt[:,3], np.linspace(0, np.percentile(zt[:,7]/zt[:,3], 99), 51))
ax4.plot([m4, m4], [0, np.max(n)*1.1], 'k--', lw=2)
ax4.set_ylim([0, np.max(n)*1.1])
ax4.text(0.96, 0.9, r'$\epsilon=0.011$', va='center', ha='right', fontsize=10, transform=ax4.transAxes)
ax4.text(0.96, 0.85, f'Median: {m4:.2f}', va='center', ha='right', fontsize=10, transform=ax4.transAxes)

fig.text(0.5, 0.05, r'$\rm{rms}_{\rm{sim}}/\rm{rms}_{\rm{obs-rc}}$', va='center', fontsize=12)
fig.text(0.02, 0.53, 'frequency (number of chunks)', va='center', rotation='vertical', fontsize=12)
fig.savefig('zse1_cheby.png', dpi=200)


#-------------------------------------------------

orders = np.arange(82, 125)
xc = 2048.0
oc = (np.min(orders) + np.max(orders))/2.0
xd = 416.0
od = (np.max(orders) - np.min(orders))/8.0

fig = plt.figure(figsize=(16,18))
ax1 = plt.Axes(fig, [0.25, 11.1/18.0, 0.55, 6.9/18.0*0.9])
fig.add_axes(ax1)
for i in np.linspace(80, 125, 60):
    ax1.plot([0, 4096], [i, i], 'k-', alpha=0.1)
for i in np.linspace(0, 4095, 84):
    ax1.plot([i, i], [80, 125], 'k-', alpha=0.1)
ax1.add_patch(Rectangle((xc-4*xd, oc-4*od), 8*xd, 8*od, ec='forestgreen', fc='forestgreen', lw=1))
ax1.add_patch(Rectangle((xc-3*xd, oc-3*od), 6*xd, 6*od, ec='y', fc='y', lw=1))
ax1.add_patch(Rectangle((xc-2*xd, oc-2*od), 4*xd, 4*od, ec='b', fc='b', lw=1))
ax1.set_xlim([0, 4095])
ax1.set_ylim([80, 125])
ax1.set_xlabel('pixel position x', fontsize=16, verticalalignment='top')
ax1.set_ylabel('order', fontsize=16, horizontalalignment='center', verticalalignment='bottom')

ax21 = plt.Axes(fig, [0.06, 7.4/18.0, 0.2, 2.8/18.0])
fig.add_axes(ax21)
ax22 = plt.Axes(fig, [0.3, 7.4/18.0, 0.2, 2.8/18.0])
fig.add_axes(ax22)
ax23 = plt.Axes(fig, [0.54, 7.4/18.0, 0.2, 2.8/18.0])
fig.add_axes(ax23)
ax24 = plt.Axes(fig, [0.78, 7.4/18.0, 0.2, 2.8/18.0])
fig.add_axes(ax24)

mask21 = (zt[:,2] > xc - 2*xd) & (zt[:,2] < xc + 2*xd)
mask22 = (zt[:,0] > oc - 2*od) & (zt[:,0] < oc + 2*od)
mask2 = mask21 & mask22

m1 = np.median(zt[mask2,4]/zt[mask2,3])
m2 = np.median(zt[mask2,5]/zt[mask2,3])
m3 = np.median(zt[mask2,6]/zt[mask2,3])
m4 = np.median(zt[mask2,7]/zt[mask2,3])

n, bins, patches = ax21.hist(zt[mask2,4]/zt[mask2,3], np.linspace(0, np.percentile(zt[mask2,4]/zt[mask2,3], 99), 51), color='b')
ax21.plot([m1, m1], [0, np.max(n)*1.1], 'k--', lw=2)
ax21.set_ylim([0, np.max(n)*1.1])
ax21.text(0.96, 0.9, r'$\epsilon=0.02$', va='center', ha='right', fontsize=10, transform=ax21.transAxes)
ax21.text(0.96, 0.85, f'Median: {m1:.2f}', va='center', ha='right', fontsize=10, transform=ax21.transAxes)

n, bins, patches = ax22.hist(zt[mask2,5]/zt[mask2,3], np.linspace(0, np.percentile(zt[mask2,5]/zt[mask2,3], 99), 51), color='b')
ax22.plot([m2, m2], [0, np.max(n)*1.1], 'k--', lw=2)
ax22.set_ylim([0, np.max(n)*1.1])
ax22.text(0.96, 0.9, r'$\epsilon=0.017$', va='center', ha='right', fontsize=10, transform=ax22.transAxes)
ax22.text(0.96, 0.85, f'Median: {m2:.2f}', va='center', ha='right', fontsize=10, transform=ax22.transAxes)

n, bins, patches = ax23.hist(zt[mask2,6]/zt[mask2,3], np.linspace(0, np.percentile(zt[mask2,6]/zt[mask2,3], 99), 51), color='b')
ax23.plot([m3, m3], [0, np.max(n)*1.1], 'k--', lw=2)
ax23.set_ylim([0, np.max(n)*1.1])
ax23.text(0.96, 0.9, r'$\epsilon=0.014$', va='center', ha='right', fontsize=10, transform=ax23.transAxes)
ax23.text(0.96, 0.85, f'Median: {m3:.2f}', va='center', ha='right', fontsize=10, transform=ax23.transAxes)

n, bins, patches = ax24.hist(zt[mask2,7]/zt[mask2,3], np.linspace(0, np.percentile(zt[mask2,7]/zt[mask2,3], 99), 51), color='b')
ax24.plot([m4, m4], [0, np.max(n)*1.1], 'k--', lw=2)
ax24.set_ylim([0, np.max(n)*1.1])
ax24.text(0.96, 0.9, r'$\epsilon=0.011$', va='center', ha='right', fontsize=10, transform=ax24.transAxes)
ax24.text(0.96, 0.85, f'Median: {m4:.2f}', va='center', ha='right', fontsize=10, transform=ax24.transAxes)

ax31 = plt.Axes(fig, [0.06, 4.15/18.0, 0.2, 2.8/18.0])
fig.add_axes(ax31)
ax32 = plt.Axes(fig, [0.3, 4.15/18.0, 0.2, 2.8/18.0])
fig.add_axes(ax32)
ax33 = plt.Axes(fig, [0.54, 4.15/18.0, 0.2, 2.8/18.0])
fig.add_axes(ax33)
ax34 = plt.Axes(fig, [0.78, 4.15/18.0, 0.2, 2.8/18.0])
fig.add_axes(ax34)

mask31 = (zt[:,2] > xc - 3*xd) & (zt[:,2] < xc + 3*xd)
mask32 = (zt[:,0] > oc - 3*od) & (zt[:,0] < oc + 3*od)
mask3 = (mask31 & mask32) & ~mask2

m1 = np.median(zt[mask3,4]/zt[mask3,3])
m2 = np.median(zt[mask3,5]/zt[mask3,3])
m3 = np.median(zt[mask3,6]/zt[mask3,3])
m4 = np.median(zt[mask3,7]/zt[mask3,3])

n, bins, patches = ax31.hist(zt[mask3,4]/zt[mask3,3], np.linspace(0, np.percentile(zt[mask3,4]/zt[mask3,3], 99), 51), color='y')
ax31.plot([m1, m1], [0, np.max(n)*1.1], 'k--', lw=2)
ax31.set_ylim([0, np.max(n)*1.1])
ax31.text(0.96, 0.9, r'$\epsilon=0.02$', va='center', ha='right', fontsize=10, transform=ax31.transAxes)
ax31.text(0.96, 0.85, f'Median: {m1:.2f}', va='center', ha='right', fontsize=10, transform=ax31.transAxes)

n, bins, patches = ax32.hist(zt[mask3,5]/zt[mask3,3], np.linspace(0, np.percentile(zt[mask3,5]/zt[mask3,3], 99), 51), color='y')
ax32.plot([m2, m2], [0, np.max(n)*1.1], 'k--', lw=2)
ax32.set_ylim([0, np.max(n)*1.1])
ax32.text(0.96, 0.9, r'$\epsilon=0.017$', va='center', ha='right', fontsize=10, transform=ax32.transAxes)
ax32.text(0.96, 0.85, f'Median: {m2:.2f}', va='center', ha='right', fontsize=10, transform=ax32.transAxes)

n, bins, patches = ax33.hist(zt[mask3,6]/zt[mask3,3], np.linspace(0, np.percentile(zt[mask3,6]/zt[mask3,3], 99), 51), color='y')
ax33.plot([m3, m3], [0, np.max(n)*1.1], 'k--', lw=2)
ax33.set_ylim([0, np.max(n)*1.1])
ax33.text(0.96, 0.9, r'$\epsilon=0.014$', va='center', ha='right', fontsize=10, transform=ax33.transAxes)
ax33.text(0.96, 0.85, f'Median: {m3:.2f}', va='center', ha='right', fontsize=10, transform=ax33.transAxes)

n, bins, patches = ax34.hist(zt[mask3,7]/zt[mask3,3], np.linspace(0, np.percentile(zt[mask3,7]/zt[mask3,3], 99), 51), color='y')
ax34.plot([m4, m4], [0, np.max(n)*1.1], 'k--', lw=2)
ax34.set_ylim([0, np.max(n)*1.1])
ax34.text(0.96, 0.9, r'$\epsilon=0.011$', va='center', ha='right', fontsize=10, transform=ax34.transAxes)
ax34.text(0.96, 0.85, f'Median: {m4:.2f}', va='center', ha='right', fontsize=10, transform=ax34.transAxes)

ax41 = plt.Axes(fig, [0.06, 0.9/18.0, 0.2, 2.8/18.0])
fig.add_axes(ax41)
ax42 = plt.Axes(fig, [0.3, 0.9/18.0, 0.2, 2.8/18.0])
fig.add_axes(ax42)
ax43 = plt.Axes(fig, [0.54, 0.9/18.0, 0.2, 2.8/18.0])
fig.add_axes(ax43)
ax44 = plt.Axes(fig, [0.78, 0.9/18.0, 0.2, 2.8/18.0])
fig.add_axes(ax44)

mask41 = (zt[:,2] > xc - 4*xd) & (zt[:,2] < xc + 4*xd)
mask42 = (zt[:,0] > oc - 4*od) & (zt[:,0] < oc + 4*od)
mask4 = (mask41 & mask42) & (~mask2 & ~mask3)

m1 = np.median(zt[mask4,4]/zt[mask4,3])
m2 = np.median(zt[mask4,5]/zt[mask4,3])
m3 = np.median(zt[mask4,6]/zt[mask4,3])
m4 = np.median(zt[mask4,7]/zt[mask4,3])

n, bins, patches = ax41.hist(zt[mask4,4]/zt[mask4,3], np.linspace(0, np.percentile(zt[mask4,4]/zt[mask4,3], 99), 51), color='forestgreen')
ax41.plot([m1, m1], [0, np.max(n)*1.1], 'k--', lw=2)
ax41.set_ylim([0, np.max(n)*1.1])
ax41.text(0.96, 0.9, r'$\epsilon=0.02$', va='center', ha='right', fontsize=10, transform=ax41.transAxes)
ax41.text(0.96, 0.85, f'Median: {m1:.2f}', va='center', ha='right', fontsize=10, transform=ax41.transAxes)

n, bins, patches = ax42.hist(zt[mask4,5]/zt[mask4,3], np.linspace(0, np.percentile(zt[mask4,5]/zt[mask4,3], 99), 51), color='forestgreen')
ax42.plot([m2, m2], [0, np.max(n)*1.1], 'k--', lw=2)
ax42.set_ylim([0, np.max(n)*1.1])
ax42.text(0.96, 0.9, r'$\epsilon=0.017$', va='center', ha='right', fontsize=10, transform=ax42.transAxes)
ax42.text(0.96, 0.85, f'Median: {m2:.2f}', va='center', ha='right', fontsize=10, transform=ax42.transAxes)

n, bins, patches = ax43.hist(zt[mask4,6]/zt[mask4,3], np.linspace(0, np.percentile(zt[mask4,6]/zt[mask4,3], 99), 51), color='forestgreen')
ax43.plot([m3, m3], [0, np.max(n)*1.1], 'k--', lw=2)
ax43.set_ylim([0, np.max(n)*1.1])
ax43.text(0.96, 0.9, r'$\epsilon=0.014$', va='center', ha='right', fontsize=10, transform=ax43.transAxes)
ax43.text(0.96, 0.85, f'Median: {m3:.2f}', va='center', ha='right', fontsize=10, transform=ax43.transAxes)

n, bins, patches = ax44.hist(zt[mask4,7]/zt[mask4,3], np.linspace(0, np.percentile(zt[mask4,7]/zt[mask4,3], 99), 51), color='forestgreen')
ax44.plot([m4, m4], [0, np.max(n)*1.1], 'k--', lw=2)
ax44.set_ylim([0, np.max(n)*1.1])
ax44.text(0.96, 0.9, r'$\epsilon=0.011$', va='center', ha='right', fontsize=10, transform=ax44.transAxes)
ax44.text(0.96, 0.85, f'Median: {m4:.2f}', va='center', ha='right', fontsize=10, transform=ax44.transAxes)

ax1.tick_params(which='major', labelsize = 14)
ax21.tick_params(which='major', labelsize = 12)
ax22.tick_params(which='major', labelsize = 12)
ax23.tick_params(which='major', labelsize = 12)
ax24.tick_params(which='major', labelsize = 12)
ax31.tick_params(which='major', labelsize = 12)
ax32.tick_params(which='major', labelsize = 12)
ax33.tick_params(which='major', labelsize = 12)
ax34.tick_params(which='major', labelsize = 12)
ax41.tick_params(which='major', labelsize = 12)
ax42.tick_params(which='major', labelsize = 12)
ax43.tick_params(which='major', labelsize = 12)
ax44.tick_params(which='major', labelsize = 12)

fig.text(0.5, 0.02, r'$\rm{rms}_{\rm{sim}}/\rm{rms}_{\rm{obs-rc}}$', va='center', fontsize=16)
fig.text(0.015, 0.3, 'frequency (number of chunks)', va='center', rotation='vertical', fontsize=16)
fig.savefig('zse2_cheby.png', dpi=200)


#--------------------------------------
x_diff = res_chunk[1,0] - res_chunk[0,0]
order_diff = np.unique(res_chunk[:,1])[1] - np.unique(res_chunk[:,1])[0]
sechunk = np.zeros(shape=(0), dtype=float)
fig5, ax5 = plt.subplots()
ax5.plot([0, 4095], [ws_orders[0], ws_orders[-1]], color='white', zorder=1)
ax5.set_xlim([-0.5, 4095.5])
ax5.set_ylim([ws_orders[0]-0.5, ws_orders[-1]+0.5])
for i in range(len(res_chunk)):
    mask5 = (abs(zt[:,0]-res_chunk[i,1]) < order_diff/2.0+1e-2) & (abs(zt[:,2]-res_chunk[i,0]) <= x_diff/2.0)
    m1 = np.median(zt[mask5,4]/zt[mask5,3])
    m2 = np.median(zt[mask5,5]/zt[mask5,3])
    m3 = np.median(zt[mask5,6]/zt[mask5,3])
    m4 = np.median(zt[mask5,7]/zt[mask5,3])
    m = np.array([m1, m2, m3, m4])
    eps = np.array([eps1, eps2, eps3, eps4])
    sef = interp1d(m, eps, kind='linear', fill_value='extrapolate')
    sechunk = np.append(sechunk, [sef(1.0)], axis=0)
    print (res_chunk[i,1], sef(1.0))
ynum = len(np.unique(res_chunk[:,1]))
xnum = len(np.unique(res_chunk[:,0]))
sechunk = sechunk.reshape(ynum, xnum)

im5 = ax5.imshow(sechunk, origin='lower', interpolation='none', extent=([384, 3712, ws_orders[0], ws_orders[-1]]), zorder=100)
ax5.set_aspect(4096*9.0/16/(ws_orders[-1]-ws_orders[0]+1))
cbar5 = fig5.colorbar(im5, extend='both', shrink=0.85, ax=ax5)
cbar5.ax.set_xlabel('IP error', fontsize=16)
cbar5.ax.tick_params(which='major', labelsize=14)
plt.xlabel('pixel position x', fontsize=16, verticalalignment='top')
plt.ylabel('order number', fontsize=16, horizontalalignment='center', verticalalignment='bottom')
plt.gca().tick_params(which='major', labelsize=14)
fig5.set_size_inches(18.0, 9.0)
fig5.savefig('zsechunk_cheby.png', dpi=200)

