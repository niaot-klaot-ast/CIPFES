#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar
from scipy.interpolate import splev, interp1d, LSQUnivariateSpline, interp2d
from numpy.polynomial import legendre
from scipy.integrate import quad
from scipy.signal import convolve
from funcs import rmsd, cheby_ev
from copy import deepcopy
from scipy.special import factorial, comb, hermite
import pickle
from funcs import *
from matplotlib import cm
from functools import reduce
from random import uniform

def gaussian(x, inten, x0, sigma, background):
    return inten*np.exp(-(x-x0)**2/(2*sigma**2)) + background

def upside_down_guassian(x, inten, x0, sigma, background):
    return 1.0 - inten*np.exp(-(x-x0)**2/(2*sigma**2)) + background

filename = 'IP_20170922_br_3rows.p'
br = pickle.load(open(filename, "rb"))
line_list = br['line_list']
normalizing_constant0 = br['normalizing_constant_o']
normalizing_constant1 = br['normalizing_constant_x']
sig_br_sol2d = br['sig']
beta_br_sol2d = br['beta']
res_tck_coeffs_array = br['residual']['coeffs']
res_knots = br['residual']['knots']
res_deg = br['residual']['deg']
res_chunk = br['residual']['chunk']

filename = 'IP_20170922_gh.p'
gh = pickle.load(open(filename, "rb"))
sig_gh_sol2d = gh['sig']
An_sol2d = gh['An']

filename = 'IP_20170922_cheby.p'
cheby = pickle.load(open(filename, "rb"))
T_sol2d = cheby['T']

filename = 'wavelength_solution_a201709220026.p'
ws = pickle.load(open(filename, "rb"))
g1, g2 = ws['domain_of_definition_o']
left_edge, right_edge = ws['domain_of_definition_x']
ws2d = ws['solution']
g_orders = np.arange(int(g1), int(g2)+1)


otest = g_orders
xtest_init = np.arange(512, 3584+1, 64)
aFWHM = np.array([1, 10, 20, 30])  # in the unit of km/s
# abr = np.zeros(shape=(len(xtest), len(aFWHM), 2), dtype=float)
# agh = np.zeros(shape=(len(xtest), len(aFWHM), 2), dtype=float)
# c1 = np.zeros(shape=(len(xtest)), dtype=float)
# c2 = np.zeros(shape=(len(xtest)), dtype=float)
# c3 = np.zeros(shape=(len(xtest)), dtype=float)
# rms1 = np.zeros(shape=(len(xtest)), dtype=float)
# rms2 = np.zeros(shape=(len(xtest)), dtype=float)
# rms3 = np.zeros(shape=(len(xtest)), dtype=float)
bias_br = np.zeros(shape=(0,4), dtype=float)
bias_gh = np.zeros(shape=(0,4), dtype=float)
bias_cheby = np.zeros(shape=(0,4), dtype=float)
# c1 = np.zeros(shape=(len(otest)), dtype=float)
# c2 = np.zeros(shape=(len(otest)), dtype=float)
# c3 = np.zeros(shape=(len(otest)), dtype=float)
# rms1 = np.zeros(shape=(len(otest)), dtype=float)
# rms2 = np.zeros(shape=(len(otest)), dtype=float)
# rms3 = np.zeros(shape=(len(otest)), dtype=float)


# obtain the non-repetative range of each order
non_rep_range = np.zeros(shape=(len(g_orders), 3), dtype=float)
for i in range(len(g_orders)-1):
    right_wav1 = legendre.legval2d(g_orders[i]/normalizing_constant0, 4095.0/normalizing_constant1, ws2d)/(g_orders[i]/normalizing_constant0)
    left_wav2 = legendre.legval2d((g_orders[i+1])/normalizing_constant0, 0.0/normalizing_constant1, ws2d)/((g_orders[i+1])/normalizing_constant0)
    cross_wav = (right_wav1 + left_wav2) / 2.0
    def wav2pos(x, o, w):
        return legendre.legval2d(o/normalizing_constant0, x/normalizing_constant1, ws2d)/(o/normalizing_constant0) - w
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


for i in range(len(otest)):
    mask_temp = (xtest_init > non_rep_range[i,1]) & (xtest_init < non_rep_range[i,2])
    xtest = xtest_init[mask_temp]

    for j in range(len(xtest)):

        # backbone residual model
        sig = legendre.legval2d(otest[i]/normalizing_constant0, xtest[j]/normalizing_constant1, sig_br_sol2d)
        beta = legendre.legval2d(otest[i]/normalizing_constant0, xtest[j]/normalizing_constant1, beta_br_sol2d)
        tck_coeffs1 = np.zeros(shape=len(res_tck_coeffs_array), dtype=float)
        for k in range(len(res_tck_coeffs_array)):
            tck_coeffs1[k] = res_tck_coeffs_array[k](xtest[j], otest[i])
        tck1 = (res_knots, tck_coeffs1, res_deg)
                    
        def backbone_residual(x, inten, x0):
            
            try:
                x = float(x)
                x = np.array([x])
                flag = 1
            except:
                x = np.array(x)
                flag = 2
            mask1 = (x-x0 >= np.min(res_knots)) & (x-x0 <= np.max(res_knots))
            
            y = np.zeros(shape=len(x), dtype=float)
            try:
                y[mask1] = np.exp(-(abs(x[mask1]-x0)/sig)**beta) + splev(x[mask1]-x0, tck1)
            except:
                pass
            
            try:
                y[~mask1] = np.exp(-(abs(x[~mask1]-x0)/sig)**beta)
            except:
                pass
            
            if flag == 1:
                return inten * y[0]
            elif flag == 2:
                return inten * y
        
        
        # super gaussian model
        def super_gaussian(x, inten, x0):
            return inten * np.exp(-(abs(x-x0)/sig)**beta)
        

        # gauss-hermite model
        sig_gh = legendre.legval2d(otest[i]/normalizing_constant0, xtest[j]/normalizing_constant1, sig_gh_sol2d)
        An = []
        for k in range(len(An_sol2d)):
            An.append(legendre.legval2d(otest[i]/normalizing_constant0, xtest[j]/normalizing_constant1, An_sol2d[k]))
        
        def coef(i):
            if i==0 or i==1:
                return 1.0
            else:
                return 1.0/reduce(int.__mul__, range(2*i-2, 0, -4))
        
        def gauss_hermite(x, inten, x0):
            try:
                x = float(x)
                x = np.array([x])
                flag = 1
            except:
                x = np.array(x)
                flag = 2
            
            y = np.zeros(shape=len(x), dtype=float)
            for k in range(len(An)):
                y = y + An[k] * 1.0/(np.sqrt(2.0*np.pi)*sig_gh) * coef(k) * hermite(k)((x-x0)/sig_gh) * np.exp(-(x-x0)**2/(2*sig_gh**2))

            if flag == 1:
                return inten * y[0]
            elif flag == 2:
                return inten * y
    
    
        # chebyshev model
        T = []
        for k in range(len(T_sol2d)):
            T.append(legendre.legval2d(otest[i]/normalizing_constant0, xtest[j]/normalizing_constant1, T_sol2d[k]))

        def chebyshev(x, inten, x0):
            
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
                return inten* y


        inten1 = quad(backbone_residual, np.min(res_knots), np.max(res_knots), args=(1.0, 0.0))[0]
        integrand1 = lambda x: x * backbone_residual(x, 1.0, 0.0) / inten1
        com_br = quad(integrand1, np.min(res_knots), np.max(res_knots))[0]
        inten2 = quad(gauss_hermite, np.min(res_knots), np.max(res_knots), args=(1.0, 0.0))[0]
        integrand2 = lambda x: x * gauss_hermite(x, 1.0, 0.0) / inten2
        com_gh = quad(integrand2, np.min(res_knots), np.max(res_knots))[0]
        inten3 = quad(chebyshev, np.min(res_knots), np.max(res_knots), args=(1.0, 0.0))[0]
        integrand3 = lambda x: x * chebyshev(x, 1.0, 0.0) / inten3
        com_cheby = quad(integrand3, np.min(res_knots), np.max(res_knots))[0]
        print (otest[i], xtest[j], com_br, com_gh, com_cheby)

        # com_br = 0.0
        # com_gh = 0.0
        # com_cheby = 0.0
        # print (otest[i], xtest[j])

        wav1 = legendre.legval2d(otest[i]/normalizing_constant0, xtest[j]/normalizing_constant1, ws2d)/(otest[i]/normalizing_constant0)

        x1 = np.linspace(-10, 10, 21) + uniform(-0.5, 0.5)
        y1 = backbone_residual(x1, 1.0, -com_br)
        par0 = [1.0, 1.0, 2.0, 0.0]
        popt, pcov = curve_fit(gaussian, x1, y1, p0=par0)
        bias_br = np.append(bias_br, [[wav1, otest[i], xtest[j], l(popt[1])]], axis=0)
        # rms_br[i] = rmsd(y1, gaussian(x1, *popt))

        # x1 = np.linspace(-10, 10, 21)
        # y1 = backbone_residual(x1, 1.0/inten1, -com_br)
        # par0 = [1.0/inten1, -com_br]
        # popt, pcov = curve_fit(backbone_residual, x1, y1, p0=par0)
        # c2[i] = popt[1] + com_br
        # rms2[i] = rmsd(y1, backbone_residual(x1, *popt))
        # plt.figure()
        # plt.plot(x1, y1, 'ks')
        # plt.plot(np.linspace(-10, 10, 2001), gauss_hermite_ix(np.linspace(-10, 10, 2001), *popt), '--')
        # plt.show

        # x1 = np.linspace(-10, 10, 21)
        # y1 = backbone_residual(x1, 1.0/inten1, -com_br)
        # par0 = [1.0/inten1, 0.0]
        # popt, pcov = curve_fit(super_gaussian, x1, y1, p0=par0)
        # c3[i] = popt[1]
        # rms3[i] = rmsd(y1, super_gaussian(x1, *popt))
    
        x1 = np.linspace(-10, 10, 21) + uniform(-0.5, 0.5)
        y1 = gauss_hermite(x1, 1.0, -com_gh)
        par0 = [1.0, 1.0, 2.0, 0.0]
        popt, pcov = curve_fit(gaussian, x1, y1, p0=par0)
        bias_gh = np.append(bias_gh, [[wav1, otest[i], xtest[j], ll(popt[1])]], axis=0)
        # rms2[i] = rmsd(y1, gaussian(x1, *popt))

        x1 = np.linspace(-10, 10, 21) + uniform(-0.5, 0.5)
        y1 = chebyshev(x1, 1.0, -com_cheby)
        par0 = [1.0, 1.0, 2.0, 0.0]
        popt, pcov = curve_fit(gaussian, x1, y1, p0=par0)
        bias_cheby = np.append(bias_cheby, [[wav1, otest[i], xtest[j], l(popt[1])]], axis=0)
        # rms3[i] = rmsd(y1, gaussian(x1, *popt))


    # for j in range(len(aFWHM)):

    #     x1 = np.linspace(-10, 10, 2001)
    #     ip = backbone_residual(x1, 1.0/inten1, -com_br) / 100.0
    #     x2 = np.linspace(-100, 100, 20001)
    #     y2 = np.ones(shape=len(x2), dtype=float) - gaussian(x2, 0.5, 0.0, aFWHM[j]/1.2/2.355, 0.0)
    #     xs = x2[len(x1)//2:len(x2)-len(x1)//2]
    #     s = convolve(y2, ip, mode='valid')
    #     fs = interp1d(xs, s, kind='cubic')
    #     xf = np.linspace(-80, 80, 161)
    #     yf = fs(xf)
    #     par0 = [0.1, 1.0, aFWHM[j]/1.2/2.355, 1.0]
    #     popt, pcov = curve_fit(upside_down_guassian, xf, yf, p0=par0)

    #     # plt.figure()
    #     # plt.plot(xf, yf, 'ks')
    #     # plt.plot(np.linspace(-80, 80, 1601), upside_down_guassian(np.linspace(-80, 80, 1601), *popt), 'g-')
    #     # plt.show()

    #     abr[i,j,0] = aFWHM[j]
    #     abr[i,j,1] = popt[1]


        # x1 = np.linspace(-10, 10, 2001)
        # ip = gauss_hermite(x1, 1.0, -com_gh) / 100.0
        # x2 = np.linspace(-100, 100, 20001)
        # y2 = np.ones(shape=len(x2), dtype=float) - gaussian(x2, 0.5, 0.0, aFWHM[j]/1.2/2.355, 0.0)
        # xs = x2[len(x1)//2:len(x2)-len(x1)//2]
        # s = convolve(y2, ip, mode='valid')
        # fs = interp1d(xs, s, kind='cubic')
        # xf = np.linspace(-80, 80, 161)
        # yf = fs(xf)
        # par0 = [0.5, 0.0, aFWHM[j]/1.2/2.355, 1.0]
        # popt, pcov = curve_fit(upside_down_guassian, xf, yf, p0=par0)

        # # plt.figure()
        # # plt.plot(xf, yf, 'ks')
        # # plt.plot(np.linspace(-80, 80, 1601), upside_down_guassian(np.linspace(-80, 80, 1601), *popt), 'g-')
        # # plt.show()

        # agh[i,j,0] = aFWHM[j]
        # agh[i,j,1] = popt[1]

        # bias = random.uniform(-0.5, 0.5)
        # x1 = bias + np.linspace(-10, 10, 21)
        # y1 = gauss_hermite(x1, -com_gh)

        # 
        # popt, pcov = curve_fit(gaussian, x1, y1, p0=par0)
        # print (bias, com_gh, popt[1])


# plt.figure()
# plt.plot(abr[0,:,0], abr[0,:,1], 'b-')
# plt.plot(abr[-1,:,0], abr[-1,:,1], 'g-')

# plt.figure()
# plt.plot(agh[0,:,0], agh[0,:,1], 'b-')
# plt.plot(agh[-1,:,0], agh[-1,:,1], 'g-')

# plt.figure()
# plt.plot(xtest, c1, 'b')
# # plt.plot(xtest, c2, 'g')
# # plt.plot(xtest, c3, 'r')
# plt.plot(xtest, abr[:,0,1], 'g')

# plt.figure()
# plt.plot(xtest, c1, 'b')
# plt.plot(xtest, abr[:,9,1], 'g')

# plt.figure()
# plt.plot(xtest, c1, 'b')
# plt.plot(xtest, abr[:,19,1], 'g')

# plt.figure()
# plt.plot(xtest, c1, 'b')
# plt.plot(xtest, abr[:,29,1], 'g')

# plt.figure()
# plt.plot(xtest, rms1, 'b')
# plt.plot(xtest, rms2, 'g')
# plt.plot(xtest, rms3, 'r')


# w = np.array([0, 9, 19, 29])
# ax = list(range(4))
# f, ((ax[0], ax[1]), (ax[2], ax[3])) = plt.subplots(2, 2, figsize=(10,5.625))
# plt.subplots_adjust(hspace=0.25, wspace=0.03)
# for i in range(len(w)):
       
#     # ax[i].plot(xtest, c1, 'b')
#     # ax[i].plot(xtest, abr[:,w[i],1], 'g')
#     ax[i].plot(otest, c1, 'b')
#     ax[i].plot(otest, abr[:,w[i],1], 'g')

#     if i%2==0:
#         ax[i].set_yticks([-0.02, -0.01, 0, 0.01])
#         ax[i].tick_params(which='major', labelsize=10)
#     else:
#         ax[i].set_yticks([-0.02, -0.01, 0, 0.01])
#         ax[i].set_yticklabels([])
#         ax[i].tick_params(which='major', labelsize=10)
#         ax_twin = ax[i].twinx()
#         ax_twin.set_ylim(ax[i].get_ylim())
#         ax_twin.set_yticks(ax[i].yaxis.get_ticklocs())
#         ax_twin.set_yticklabels([-24, -12, 0, 12])
#         ax_twin.tick_params(which='major', labelsize=10)

#     ax[i].text(0.95, 0.93, f"AL's FWHM = {aFWHM[w[i]]} km/s", ha='right', va='center', fontsize=10, transform=ax[i].transAxes)

# f.text(0.5, 0.04, 'pixel position x', ha='center', fontsize=12)
# f.text(0.05, 0.47, 'Gaussian-fit line-centre bias [pixel]', va='center', rotation='vertical', fontsize=12)
# f.text(0.96, 0.47, 'Gaussian-fit line-centre bias [m/s]', va='center', rotation='vertical', fontsize=12)
# f.savefig('zsystematic.png', dpi=200)

fig = plt.figure(figsize=(8,6))
ax = plt.Axes(fig, [0.12, 0.12, 0.78, 0.83])
fig.add_axes(ax)
# plt.plot(xtest, c1, 'b')
# plt.plot(xtest, c2-0.02, 'g')
# plt.plot(xtest, c3, 'r')
ax.plot(bias_gh[:,0], bias_gh[:,3], color='olive', lw=0.8)
ax.plot(bias_cheby[:,0], bias_cheby[:,3], color='red', lw=0.8)
ax.plot(bias_br[:,0], bias_br[:,3], color='blue', lw=1.2)
ax.set_yticks([-0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06])
ax.tick_params(which='major', labelsize=12)
ax_twin = ax.twinx()
ax_twin.set_ylim(ax.get_ylim())
ax_twin.set_yticks(ax.yaxis.get_ticklocs())
ax_twin.set_yticklabels([-104, -78, -52, -26, 0, 26, 52, 78])
ax_twin.tick_params(which='major', labelsize=12)
ax.set_xlabel(''.join(['wavelength ', r'$[\rm{\AA}]$']), fontsize=14, va='top')
ax.set_ylabel('Gaussian-fit line-centre bias [pixel]', fontsize=14, ha='center', va='bottom')
ax_twin.set_ylabel('Gaussian-fit line-centre bias [m/s]', fontsize=14, ha='center', va='top')
fig.set_size_inches(8.0, 6.0)
fig.savefig('zdistortion.png', dpi=300)