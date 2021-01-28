#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.interpolate import splev
from scipy.optimize import root_scalar, minimize
from scipy.special import hermite
from scipy.integrate import quad
from numpy.polynomial import legendre
import textwrap
import pickle
from functools import reduce
from funcs import cheby_ev

np.set_printoptions(threshold=100000)

def coef(i):
    if i==0 or i==1:
        return 1.0/0.3989422804014329
    else:
        return 1.0/0.3989422804014329 * 1.0/reduce(int.__mul__, range(2*i-2, 0, -4))

def gauss_hermite(x, x0, sig, *An):
    try:
        x = float(x)
        x = np.array([x])
        flag = 1
    except:
        x = np.array(x)
        flag = 2
    
    y = np.zeros(shape=len(x), dtype=float)
    for i in range(len(An)):
        y = y + An[i] * 1.0/(2.0*np.pi*sig) * coef(i) * hermite(i)((x-x0)/sig) * np.exp(-(x-x0)**2/(2*sig**2))
    
    if flag == 1:
        return y[0]
    elif flag == 2:
        return y

def chebyshev(x, x0, p, T):
    
    try:
        x = float(x)
        x = np.array([x])
        flag = 1
    except:
        x = np.array(x)
        flag = 2
    
    y = np.zeros(shape=len(x), dtype=float)
    mask1 = (x-x0 >= p.min()) & (x-x0 <= p.max())
    y[mask1] = cheby_ev(x[mask1]-x0, p, T)
    y[~mask1] = 0.0
    
    if flag == 1:
        return y[0]
    elif flag == 2:
        return y

br = pickle.load(open('IP_20170922_br_3rows.p', "rb"))
normalizing_constant0 = br['normalizing_constant_o']
normalizing_constant1 = br['normalizing_constant_x']
sig_sol2d = br['sig']
beta_sol2d = br['beta']
res_tck_coeffs_array = br['residual']['coeffs']
res_knots = br['residual']['knots']
res_deg = br['residual']['deg']
res_chunk = br['residual']['chunk']

gh = pickle.load(open('IP_20170922_gh.p', "rb"))
sig_gh_sol2d = gh['sig']
An_gh_sol2d = gh['An']

cheby = pickle.load(open('IP_20170922_cheby.p', "rb"))
T_sol2d = cheby['T']

g = pickle.load(open('wavelength_solution_a201709220023.p', 'rb'))
g_line_list = g['line_list']
g_orders = np.unique(g_line_list[:,0]).astype(int)
left_edge = 50
right_edge = 4049
xind = np.arange(left_edge, right_edge+1)


o = [82, 102, 121]
x = [512, 2048, 3584]
x1 = np.linspace(-7.5, 7.5, 1000)
fig = plt.figure(figsize=(15, 10))

for i in range(3):
    for j in range(3):
        ax1 = plt.Axes(fig, [0.14+j*0.3, 0.07+0.05+0.32*i, 0.23, 0.23-0.05])
        fig.add_axes(ax1)
    
        sig = legendre.legval2d(o[i]/normalizing_constant0, x[j]/normalizing_constant1, sig_sol2d)
        beta = legendre.legval2d(o[i]/normalizing_constant0, x[j]/normalizing_constant1, beta_sol2d)
        tck_coeffs1 = np.zeros(shape=len(res_tck_coeffs_array), dtype=float)
        for k in range(len(res_tck_coeffs_array)):
            tck_coeffs1[k] = res_tck_coeffs_array[k](x[j], o[i])
        tck1 = (res_knots, tck_coeffs1, res_deg)
        y1 = np.exp(-(abs(x1)/sig)**beta) + splev(x1, tck1)
        ax1.plot(x1, y1, 'k-', label=f'BR', zorder=10)
    
    
        sig_gh = legendre.legval2d(o[i]/normalizing_constant0, x[j]/normalizing_constant1, sig_gh_sol2d)
        An_gh = np.zeros(shape=(len(An_gh_sol2d)), dtype=float)
        for k in range(len(An_gh)):
            An_gh[k] = legendre.legval2d(o[i]/normalizing_constant0, x[j]/normalizing_constant1, An_gh_sol2d[k])
        y2 = gauss_hermite(x1, *np.concatenate(([0.0, sig_gh], An_gh)))
        ax1.plot(x1, y2, '-', color='b', label=f'GH', zorder=1)
        # print (f'{o[i]}, {x[j]}, gauss_hermite: center of mass: {com2:6f}')
    

        T = np.zeros(shape=(len(T_sol2d)), dtype=float)
        for k in range(len(T)):
            T[k] = legendre.legval2d(o[i]/normalizing_constant0, x[j]/normalizing_constant1, T_sol2d[k])
        y3 = chebyshev(x1, 0.0, cheby['x'], T)
        ax1.plot(x1, y3, '-', color='orange', label=f'DCP', zorder=1)
        # print (f'{o[i]}, {x[j]}, chebyshev: center of mass: {com3:6f}')


        y4 = np.exp(-(abs(x1)/sig)**beta)
        # ax1.plot(x1, y4, 'g:', label=f'SG', zorder=10)
        ax1.set_xlim([-7.0, 7.0])
        ax1.set_ylim([-0.05, 1.1])
        ax1.set_ylabel('normalized intensity', ha='center', fontsize=12)
        ax1.tick_params(axis='both', which='both', bottom=True, labelbottom=False, labelsize=10)
        ax1.legend(loc='upper right', fontsize=9)

        ax2 = plt.Axes(fig, [0.14+j*0.3, 0.07+0.32*i, 0.23, 0.05])
        fig.add_axes(ax2)
        ax2.plot([-7, 7], [0, 0], 'k-', zorder=10)
        ax2.plot(x1, y2-y1, '-', color='b', zorder=1)
        ax2.plot(x1, y3-y1, '-', color='orange', zorder=1)
        # ax2.plot(x1, y4-y1, 'g:', zorder=10)
        ax2.set_xlim([-7.0, 7.0])
        ax2.set_ylim([-0.07, 0.07])
        ax2.set_yticks([-0.05, 0, 0.05])
        ax2.set_xlabel("x' (pixel position relative to the IP center)", va='top', fontsize=12)
        ax2.tick_params(axis='both', which='both', labelsize=10)

        ax2.text(0.2, 0.1, f'SD_GH = {np.std(y2-y1):.3f}', ha='left', fontsize=8, color='b', transform=ax2.transAxes)
        ax2.text(0.8, 0.1, f'SD_DCP = {np.std(y3-y1):.3f}', ha='right', fontsize=8, color='orange', transform=ax2.transAxes)
        

        
fig.text(0.03, 0.07+0.32*0+0.23*0.5, r'$o = 82$', ha='left', va='center', fontsize=14)
fig.text(0.03, 0.07+0.32*1+0.23*0.5, r'$o = 102$', ha='left', va='center', fontsize=14)
fig.text(0.03, 0.07+0.32*2+0.23*0.5, r'$o = 121$', ha='left', va='center', fontsize=14)
fig.text(0.14+0.3*0+0.23*0.5, 0.97, r'$x = 512$', ha='center', va='center', fontsize=14)
fig.text(0.14+0.3*1+0.23*0.5, 0.97, r'$x = 2048$', ha='center', va='center', fontsize=14)
fig.text(0.14+0.3*2+0.23*0.5, 0.97, r'$x = 3584$', ha='center', va='center', fontsize=14)
fig.savefig('zIP_multiple.png', dpi=300)