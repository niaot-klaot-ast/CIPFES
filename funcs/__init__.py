#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
from scipy.special import comb


def gaussian_fit(x, y1, sigma1):
    x, y1, sigma1=np.array(x, dtype=float), np.array(y1, dtype=float), np.array(sigma1, dtype=float)
    if min(y1)<0:
        print ('there are negative elements in y.')
        assert 0
    for j in range(len(y1)):
       if y1[j]==0:
           y1[j]=min(y1[y1!=0])*1e-6
    y=np.log(abs(y1))
    sigma=sigma1/abs(y1)
    weight=1.0/sigma**2
    
    alpha=np.zeros(shape=(3,3),dtype=float)
    for j in range(3):
        for k in range(3):
            alpha[j,k]=(x**j * x**k * weight).sum()
             
    beta=np.zeros(shape=(3),dtype=float)
    for j in range(3):
        beta[j]=(x**j * y * weight).sum()
    
    epsilon=np.linalg.inv(alpha)
    
    temp=np.dot(beta,epsilon)    
    sig=np.sqrt(-1.0/(2*temp[2]))
    mu=temp[1]*sig**2
    h=np.exp(temp[0]+mu**2/(2*sig**2))
        
    return [h, mu, sig]



def bernstein_poly(k, n, u):    
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, k) * u**k * (1-u)**(n-k)


def bezier_curve(xPoints, yPoints, nTimes=1000):   
    """
       Given a set of control points, return the bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    if len(xPoints)!=len(yPoints):
        assert 0

    nPoints = len(xPoints)
    u = np.linspace(0.0, 1.0, nTimes)
    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, u) for i in range(0, nPoints) ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def OnlyNum(s, oth='', e=True):
    s2 = s.lower()
    if e==True:
        format = 'e0123456789.-+'
    else:
        format = '0123456789.-+'
    for c in s2:
        if not c in format:
            s = s.replace(c,'');
    return s


# Root-mean-square deviation
def rmsd(f, o, **kw):
    if len(kw.keys())==0:
        return np.sqrt(((f - o)**2).sum() / len(f))


# CDG fit
def CDG_fit(x, y, **kw):

    from scipy.optimize import curve_fit
    def gaussian(p,x):
        a, x0, sig = p
        return a * np.exp(-(x-x0)**2/(2.0*sig**2))
    
    def CDG(x, mc, dc, a1, sig1, a2, sig2, background):
        x01 = (mc + dc)/2.0
        x02 = (mc - dc)/2.0
        p1 = [a1, x01, sig1]
        p2 = [a2, x02, sig2]
        return gaussian(p1, x) + gaussian(p2, x) + background
        
    cg = np.average(x, weights=y)
    stddev = np.sqrt(np.average((x-cg)**2, weights=y))
    K = 1.107
    ehfwhm = np.sqrt(2.0*np.log(2.0)) * K * stddev     # estimated half FWHM
    
    FWHM = 5.55
    mc = 2.0*cg;  dc = ehfwhm
    a1 = y.max()/2.0;  sig1 = FWHM/2.355
    a2 = y.max()/2.0;  sig2 = FWHM/2.355
    background = 0.0
    
    par0 = [mc, dc, a1, sig1, a2, sig2, background]
    bounds0 = ([(2.0*cg-ehfwhm, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 
                (2.0*cg+ehfwhm, 2.0*ehfwhm, np.inf, np.inf, np.inf, np.inf, np.inf)])
    
    if 'sigma' in kw:
        popt, pcov = curve_fit(CDG, x, y, p0=par0, bounds=bounds0, sigma=kw['sigma'])
    else:
        popt, pcov = curve_fit(CDG, x, y, p0=par0, bounds=bounds0, sigma=np.ones(len(x))*np.sqrt(np.average(y)))
    
    return popt, pcov


# return the line-center, height and FWHM of CDG function
def CDG_param1(popt):

    from scipy.optimize import root_scalar
    def gaussian(p,x):
        a, x0, sig = p
        return a * np.exp(-(x-x0)**2/(2.0*sig**2))
    def CDG(x, mc, dc, a1, sig1, a2, sig2, background):
        x01 = (mc + dc)/2.0
        x02 = (mc - dc)/2.0
        p1 = [a1, x01, sig1]
        p2 = [a2, x02, sig2]
        return gaussian(p1, x) + gaussian(p2, x) + background
        
    mc, dc, a1, sig1, a2, sig2, background = popt
    x01 = (mc + dc)/2.0
    x02 = (mc - dc)/2.0
    p1 = [a1, x01, sig1]
    p2 = [a2, x02, sig2]
    
    func1 = lambda x: -(x-x01)/(sig1**2)*gaussian(p1, x) - (x-x02)/(sig2**2)*gaussian(p2, x)
    xm = root_scalar(func1, bracket=[min(x01,x02), max(x01,x02)], method='ridder').root
    height = CDG(xm, *popt) - background
    
    func2 = lambda x: CDG(x, *popt) - background - 0.5*height
    xL = root_scalar(func2, bracket=[xm-10.0, xm], method='ridder').root
    xR = root_scalar(func2, bracket=[xm, xm+10.0], method='ridder').root
    FWHM = xR - xL
    
    return xm, height, FWHM
    
    
def moffat_fit(x, y, **kw):

    from scipy.optimize import curve_fit
    def moffat(x, inten, x0, theta, beta, delta, background):
        return inten * (1.0 + ((x-x0)/theta)**2)**(-beta * ((x-x0)/delta)**2) + background
        
    inten = y.max()-y.min();  x0 = np.median(x);  theta = 1.0
    beta = 1.0;  delta = 1.0;  background = 0.0
    par0 = [inten, x0, theta, beta, delta, background]
    bounds0 = ([(0.0, x[0], 0.0, 0.0, 0.0, 0.0), 
                (np.inf, x[-1], np.inf, np.inf, np.inf, np.inf)])
    
    if 'sigma' in kw:
        popt, pcov = curve_fit(moffat, x, y, p0=par0, bounds=bounds0, sigma=kw['sigma'])
    else:
        popt, pcov = curve_fit(moffat, x, y, p0=par0, bounds=bounds0, sigma=np.ones(len(x))*np.sqrt(np.average(y)))

    return popt, pcov
    
    
def moffat_param1(popt):

    from scipy.optimize import root_scalar
    def moffat(x, inten, x0, theta, beta, delta, background):
        return inten * (1.0 + ((x-x0)/theta)**2)**(-beta * ((x-x0)/delta)**2) + background
        
    inten, x0, theta, beta, delta, background = popt
    height = inten
    func1 = lambda x: moffat(x, *popt) - background - 0.5*height
    xL = root_scalar(func1, bracket=[x0-10.0, x0], method='ridder').root
    xR = root_scalar(func1, bracket=[x0, x0+10.0], method='ridder').root
    FWHM = xR - xL

    return x0, height, FWHM
    
    
def super_gaussian_fit(x, y, **kw):

    from scipy.optimize import curve_fit
    def super_gaussian(x, inten, x0, sig, beta, background):
        return inten * np.exp(-(abs(x-x0)/sig)**beta) + background
        
    inten = y.max()-y.min();  x0 = np.median(x);  FWHM = 5.55
    sig = FWHM/2.355;  beta = 2.0;  background = 0.0
    par0 = [inten, x0, sig, beta, background]
    bounds0 = ([(0.0, x[0], 0.0, 0.0, 0.0), 
                (np.inf, x[-1], np.inf, np.inf, np.inf)])
                
    if 'sigma' in kw:
        popt, pcov = curve_fit(super_gaussian, x, y, p0=par0, bounds=bounds0, sigma=kw['sigma'])
    else:
        popt, pcov = curve_fit(super_gaussian, x, y, p0=par0, bounds=bounds0, sigma=np.ones(len(x))*np.sqrt(np.average(y)))
        
    return popt, pcov
    
    
def super_gaussian_param1(popt, pcov):

    from scipy.optimize import root_scalar
    from scipy.misc import derivative
    def super_gaussian(x, inten, x0, sig, beta, background):
        return inten * np.exp(-(abs(x-x0)/sig)**beta) + background
        
    inten, x0, sig, beta, background = popt
    height = inten
    func1 = lambda x: super_gaussian(x, *popt) - background - 0.5*height
    xL = root_scalar(func1, bracket=[x0-10.0, x0], method='ridder').root
    xR = root_scalar(func1, bracket=[x0, x0+10.0], method='ridder').root
    FWHM = xR - xL
    
    funcx = lambda t: super_gaussian(t, inten, x0, sig, beta, background)
    
    funcxL1 = lambda t: super_gaussian(xL, t, x0, sig, beta, background)
    funcxL2 = lambda t: super_gaussian(xL, inten, t, sig, beta, background)
    funcxL3 = lambda t: super_gaussian(xL, inten, x0, t, beta, background)
    funcxL4 = lambda t: super_gaussian(xL, inten, x0, sig, t, background)
    funcxL5 = lambda t: super_gaussian(xL, inten, x0, sig, beta, t)
    
    funcxR1 = lambda t: super_gaussian(xR, t, x0, sig, beta, background)
    funcxR2 = lambda t: super_gaussian(xR, inten, t, sig, beta, background)
    funcxR3 = lambda t: super_gaussian(xR, inten, x0, t, beta, background)
    funcxR4 = lambda t: super_gaussian(xR, inten, x0, sig, t, background)
    funcxR5 = lambda t: super_gaussian(xR, inten, x0, sig, beta, t)
    
    dfxL1 = derivative(funcxL1, inten, dx=inten*1e-4, n=1)
    dfxL2 = derivative(funcxL2, x0, dx=1e-4, n=1)
    dfxL3 = derivative(funcxL3, sig, dx=sig*1e-4, n=1)
    dfxL4 = derivative(funcxL4, beta, dx=beta*1e-4, n=1)
    dfxL5 = derivative(funcxL5, background, dx=background*1e-4, n=1)
    
    dfxR1 = derivative(funcxR1, inten, dx=inten*1e-4, n=1)
    dfxR2 = derivative(funcxR2, x0, dx=1e-4, n=1)
    dfxR3 = derivative(funcxR3, sig, dx=sig*1e-4, n=1)
    dfxR4 = derivative(funcxR4, beta, dx=beta*1e-4, n=1)
    dfxR5 = derivative(funcxR5, background, dx=background*1e-4, n=1)
    
    sigfxL = np.sqrt(pcov[0,0]*dfxL1**2 + pcov[1,1]*dfxL2**2 + pcov[2,2]*dfxL3**2 + pcov[3,3]*dfxL4**2 + pcov[4,4]*dfxL5**2)
    sigfxR = np.sqrt(pcov[0,0]*dfxR1**2 + pcov[1,1]*dfxR2**2 + pcov[2,2]*dfxR3**2 + pcov[3,3]*dfxR4**2 + pcov[4,4]*dfxR5**2)
    sigxL = abs(sigfxL/derivative(funcx, xL, dx=1e-4, n=1))
    sigxR = abs(sigfxR/derivative(funcx, xR, dx=1e-4, n=1))
    sigFWHM = np.sqrt(sigxL**2 + sigxR**2)

    return x0, np.sqrt(pcov[1,1]), height, np.sqrt(pcov[0,0]), FWHM, sigFWHM
    

def gauss_hermite_fit(x, y, n, **kw):
    
    from scipy.optimize import curve_fit
    from scipy.special import hermite
    from functools import reduce

    def coef(i):
        if i==0 or i==1:
            return 1.0
        else:
            return 1.0/reduce(int.__mul__, range(2*i-2, 0, -4))

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
            y = y + An[i] * 1.0/(np.sqrt(2.0*np.pi)*sig) * coef(i) * hermite(i)(x/sig) * np.exp(-x**2/(2*sig**2))
        # y = y + background
        
        if flag == 1:
            return y[0]
        elif flag == 2:
            return y

    FWHM = 5.55; sig = FWHM/2.355; An = np.zeros(n+1)
    par0 = np.concatenate(([sig], An))

    if 'sigma' in kw:
        popt, pcov = curve_fit(gauss_hermite, x, y, p0=par0, sigma=kw['sigma'])
    else:
        popt, pcov = curve_fit(gauss_hermite, x, y, p0=par0, sigma=np.ones(len(x))*np.sqrt(np.average(y)))

    return popt, pcov


def cheby(x, p, N):
    """
    This function is defined according to the non-iterative version defined in Chapter 4 of Mukundan et al. (2000)
    """
    from scipy.special import factorial, comb

    try:
        x = float(x)
        x = np.array([x])
        flag = 1
    except:
        x = np.array(x)
        flag = 2
    
    p = int(p)
    N = int(N)
        
    if p>=0 and p<=N-1 and N>=1:
        temp1 = 0.0
        for k in range(p+1):
            if k==0:
                temp2 = np.ones(shape=(len(x)), dtype=float)
            else:
                temp2 = 1.0
                for j in range(k):
                    temp2 = temp2 * (x-j)
            temp1 = temp1 + (-1)**(p-k) * factorial(p)/factorial(k) * comb(N-1-k, p-k) * comb(p+k, p) * temp2
        return temp1
    else:
        print ('The p and N are not in the reasonable parameter range.')
        assert 0

        
def cheby_norm(p, N):
    return np.sqrt((cheby(np.arange(N), p, N) * cheby(np.arange(N), p, N)).sum())

def cheby_nd(x, p, N):
    """
    This function is the normalized discrete Chebyshev polynomial
    """
    return cheby(x, p, N) / cheby_norm(p, N)

def cheby_fit(x, y, p):
    q = np.arange(len(x))
    T = np.zeros(shape=(p+1), dtype=float)
    for j in range(p+1):
        T[j] = (cheby_nd(q, j, len(q)) * y).sum()
    return T 

def cheby_ev(x, q, T):
    try:
        x = float(x)
        x = np.array([x])
        flag = 1
    except:
        x = np.array(x)
        flag = 2
    x = x - min(q)
    y = np.zeros(shape=len(x), dtype=float)
    for j in range(len(T)):
        y = y + T[j] * cheby_nd(x, j, len(q))
    if flag == 1:
        return y[0]
    elif flag == 2:
        return y


def leg2d_fit(x, y, z, m, n, sigma_clipping=False, **kw):
    
    from copy import deepcopy
    from numpy.polynomial import legendre

    m = m+1
    n = n+1
    if 'sigma' in kw:
        s = np.array(kw['sigma'])
    else:
        s = np.ones(shape=len(x), dtype=float)

    if sigma_clipping==True:
        x1 = deepcopy(x)
        y1 = deepcopy(y)
        z1 = deepcopy(z)
        s1 = deepcopy(s)
        while 1:
            X = np.zeros(shape=(len(z1), m*n), dtype=float)
            for i in range(len(z1)):
                for j in range(m):
                    for k in range(n):
                        c = np.zeros(shape=(m,n), dtype=float)
                        c[j,k] = 1.0
                        X[i,j*n+k] = legendre.legval2d(x1[i], y1[i], c)
                X[i] = X[i]/s1[i]**2
            
            sol1 = np.linalg.lstsq(X, z1/s1**2, rcond=-1)
            sol2d1 = sol1[0].reshape(m,n)
            
            fit1 = legendre.legval2d(x1, y1, sol2d1)
            res1 = z1 - fit1
            rms1 = rmsd(z1, fit1)
            mask1 = abs(res1) > 3.0*rms1
            if len(mask1[mask1==True]) == 0:
                mask = np.array([i in z1 for i in z])
                break
            else:
                x1 = x1[~mask1]; y1 = y1[~mask1]; z1 = z1[~mask1]; s1 = s1[~mask1]

    else:
        X = np.zeros(shape=(len(z), m*n), dtype=float)
        for i in range(len(z)):
            for j in range(m):
                for k in range(n):
                    c = np.zeros(shape=(m,n), dtype=float)
                    c[j,k] = 1.0
                    X[i,j*n+k] = legendre.legval2d(x[i], y[i], c)
            X[i] = X[i]/s[i]**2
        
        sol1 = np.linalg.lstsq(X, z/s**2, rcond=-1)
        sol2d1 = sol1[0].reshape(m,n)
        mask = np.ones(shape=(z), dtype=bool)

    return sol2d1, mask