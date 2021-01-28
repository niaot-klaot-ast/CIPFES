#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.interpolate import splev, interp1d
from scipy.optimize import curve_fit, root_scalar
from numpy.polynomial import legendre
from scipy.fftpack import fft
from refractive_index import refractive_index_moist_air
import pickle

np.set_printoptions(threshold=100000)

def gaussian(x, height, x0, sigma):
    return height*np.exp(-(x-x0)**2/(2*sigma**2))


wavelength_solution_filename = 'wavelength_solution_a201709220023.p'
ws = pickle.load(open(wavelength_solution_filename, 'rb'))
ws_orders = np.unique(ws['line_list'][:,0]).astype(int)
normalizing_constant0 = ws['normalizing_constant_o']
normalizing_constant1 = ws['normalizing_constant_x']
wav_sol2d_1 = ws['solution']
dodo = ws['domain_of_definition_o']
dodx = ws['domain_of_definition_x']
xind = np.arange(dodx[0], dodx[1]+1)

psf_filename = 'IP_20170922_br_3rows.p'
psf = pickle.load(open(psf_filename, "rb"))
line_list = psf['line_list']
normalizing_constant0 = psf['normalizing_constant_o']
normalizing_constant1 = psf['normalizing_constant_x']
sig_sol2d = psf['sig']
beta_sol2d = psf['beta']
res_tck_coeffs_array = psf['residual']['coeffs']
res_knots = psf['residual']['knots']
res_deg = psf['residual']['deg']
res_chunk = psf['residual']['chunk']

solar_spec = pf.open('Kurucz_1984.fits')[1].data
thar_spec = pf.getdata('a201709220023.fits', header=False)

order_init = 64

fig = plt.figure(figsize=(8,11))


o = 100
x = 2048

sig = legendre.legval2d(o/normalizing_constant0, x/normalizing_constant1, sig_sol2d)
beta = legendre.legval2d(o/normalizing_constant0, x/normalizing_constant1, beta_sol2d)
tck_coeffs1 = np.zeros(shape=len(res_tck_coeffs_array), dtype=float)
for k in range(len(res_tck_coeffs_array)):
    tck_coeffs1[k] = res_tck_coeffs_array[k](x, o)
tck1 = (res_knots, tck_coeffs1, res_deg)

N1 = 201
x1 = np.linspace(-50, 50, N1)
T1 = x1[1] - x1[0]
y1 = np.zeros(shape=len(x1), dtype=float)
for k in range(len(x1)):
    if abs(x1[k])<=7.5:
        y1[k] = np.exp(-(abs(x1[k])/sig)**beta) + splev(x1[k], tck1)

y1f = fft(y1)
x1f = np.linspace(0.0, 1.0/(2.0*T1), N1//2)

ax = []
ax1 = plt.Axes(fig, [0.15, 0.87, 0.35, 0.11])
ax2 = plt.Axes(fig, [0.6, 0.87, 0.35, 0.11])
ax.append([ax1, ax2])
fig.add_axes(ax[0][0])
fig.add_axes(ax[0][1])
ax[0][0].plot(x1, y1)
ax[0][1].plot(x1f[:N1//2], 2.0/N1*np.abs(y1f[:N1//2]))
ax[0][0].text(0.99, 0.9, f'(a1)', ha='right', va='center', transform=ax[0][0].transAxes)
ax[0][1].text(0.99, 0.9, f'(a2)', ha='right', va='center', transform=ax[0][1].transAxes)


os = np.array([80, 90, 100, 110, 120], dtype=int)
xb = 2000
xe = 2100
N2 = 201
letter = ['b', 'c', 'd', 'e', 'f']
for i in range(len(os)):

    wav1 = legendre.legval2d(os[i]/normalizing_constant0, xe/normalizing_constant1, wav_sol2d_1)/(os[i]/normalizing_constant0)
    wav2 = legendre.legval2d(os[i]/normalizing_constant0, xb/normalizing_constant1, wav_sol2d_1)/(os[i]/normalizing_constant0)
    i1 = np.searchsorted(solar_spec['wavelength'], wav1) - 1
    i2 = np.searchsorted(solar_spec['wavelength'], wav2)
    xx = solar_spec['wavelength'][i1:i2+1]
    yy = solar_spec['flux'][i1:i2+1]
    f2 = interp1d(xx, yy, kind='cubic')

    x2 = np.linspace(xb, xe, N2)
    T2 = x2[1] - x2[0]
    y2 = f2(legendre.legval2d(os[i]/normalizing_constant0, x2/normalizing_constant1, wav_sol2d_1)/(os[i]/normalizing_constant0))
    y2 = y2 / y2.max()

    y2f = fft(y2)
    x2f = np.linspace(0.0, 1.0/(2.0*T2), N2//2)

    ax1 = plt.Axes(fig, [0.15, 0.87-0.135*(i+1), 0.35, 0.11])
    ax2 = plt.Axes(fig, [0.6, 0.87-0.135*(i+1), 0.35, 0.11])
    ax.append([ax1, ax2])
    fig.add_axes(ax[i+1][0])
    fig.add_axes(ax[i+1][1])
    ax[i+1][0].plot(x2, y2)
    ax[i+1][1].plot(x2f[:N2//2], 2.0/N2*np.abs(y2f[:N2//2]))
    ax[i+1][0].set_xticks([2000, 2020, 2040, 2060, 2080, 2100])
    ax[i+1][0].set_xticklabels([0, 20, 40, 60, 80, 100])
    ax[i+1][0].text(0.99, 0.9, f'({letter[i]}1)', ha='right', va='center', transform=ax[i+1][0].transAxes)
    ax[i+1][1].text(0.99, 0.9, f'({letter[i]}2)', ha='right', va='center', transform=ax[i+1][1].transAxes)


speed_of_light = 299792458.0
f_0 = 9.52                            # GHz
f_rep = 25.0                          # GHz
N3 = 201

left_wav = legendre.legval2d(o/normalizing_constant0, xb/normalizing_constant1, wav_sol2d_1)/(o/normalizing_constant0)
right_wav = legendre.legval2d(o/normalizing_constant0, xe/normalizing_constant1, wav_sol2d_1)/(o/normalizing_constant0)
low_n = int(np.floor((speed_of_light/left_wav*10.0 - f_0) / f_rep))
high_n = int(np.ceil((speed_of_light/right_wav*10.0 - f_0) / f_rep))
yy = speed_of_light / (f_0 + f_rep*np.arange(low_n, high_n+1)) * 10.0
def wav2pos(x, o, w):
    return legendre.legval2d(o/normalizing_constant0, x/normalizing_constant1, wav_sol2d_1)/(o/normalizing_constant0) - w
b = np.zeros(shape=len(yy), dtype=float)
for i in range(len(yy)):
    b[i] = root_scalar(wav2pos, args=(o, yy[i]), bracket=[0, 4095], method='ridder').root

x3 = np.linspace(xb, xe, N3)
T3 = x3[1] - x3[0]
y3 = np.zeros(shape=len(x3), dtype=float)
for i in range(len(b)):
    y3[abs(x3 - b[i]).argmin()] = 1.0
y3[0] = 0.0
y3[-1] = 0.0

N4 = 20001
x4 = np.linspace(xb, xe, N4)
y4 = np.zeros(shape=len(x4), dtype=float)
for i in range(len(b)):
    y4[abs(x4 - b[i]).argmin()] = 1.0
y4[0] = 0.0
y4[-1] = 0.0

y3f = fft(y3)
x3f = np.linspace(0.0, 1.0/(2.0*T3), N3//2)

ax1 = plt.Axes(fig, [0.15, 0.87-0.135*6, 0.35, 0.11])
ax2 = plt.Axes(fig, [0.6, 0.87-0.135*6, 0.35, 0.11])
ax.append([ax1, ax2])
fig.add_axes(ax[6][0])
fig.add_axes(ax[6][1])
ax[6][0].plot(x4, y4)
ax[6][1].plot(x3f[x3f<=1], 2.0/N3*np.abs(y3f[:N3//2][x3f<=1]))
ax[6][0].set_xticks([2000, 2020, 2040, 2060, 2080, 2100])
ax[6][0].set_xticklabels([0, 20, 40, 60, 80, 100])
ax[6][0].text(0.99, 0.9, f'(g1)', ha='right', va='center', transform=ax[6][0].transAxes)
ax[6][1].text(0.99, 0.9, f'(g2)', ha='right', va='center', transform=ax[6][1].transAxes)

fig.text(0.325, 0.02, 'pixel position', fontsize=12, ha='center', va='center')
fig.text(0.775, 0.02, r'$\rm{frequency \ [pixel} ^{-1} \rm{]}$', fontsize=12, ha='center', va='center')
fig.text(0.05, 0.522, 'normalized intensity', fontsize=12, rotation='vertical', ha='center', va='center')
fig.text(0.55, 0.522, 'amplitude', fontsize=12, rotation='vertical', ha='center', va='center')
fig.savefig(f'zFFT.png', dpi=200)