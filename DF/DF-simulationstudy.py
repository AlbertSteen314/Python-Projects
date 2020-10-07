#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:35:24 2020

@author: ezravanderstelt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import random
import statsmodels.api as sm
from math import sqrt
from mylib.bieb import AR
from mylib.bieb import RdmWalk
import time

"""
Sources: Dickey/Fuller 1979 paper on distribution of DF test, simulation study
file:///Users/ezravanderstelt/Desktop/literatuur/adv%20ectri/DFdistribution1979.pdf
https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=7266&context=rtd
file:///Users/ezravanderstelt/Desktop/literatuur/FinancialEconometrics/Brooks(2008).pdf p588

DF test tests whether there is a unit root, so this is the H0, if we reject the test
we conclude the series is stationary or trend-stationary. I will restrict myself to
deriving tables for the no constant random walk DGP. 

Aim: derive critical value DF distribution table for varying sample size n.
Also try to do the power comparisons as in the table of the main DF 1979 paper
"""

# =============================================================================
# HERE WE DERIVE CRITICAL VALUES DF DISTRIBUTION
# =============================================================================

n = [25, 50, 100, 500, 5000] # different sample sizes
N = 50000 # number of simulations
t0 = 0
t_stats = np.zeros((len(n),N))
df_cv = pd.DataFrame(0,index = n, columns = ['0.01', '0.025','0.05','0.10','0.90','0.95', '0.975','0.99'])

start = time.time()
for n_loop,size in enumerate(n):
        for sim in range(N):
            y = AR(p=1, n=size, start = t0, mu = 0, sigma = 1)
            dy = np.diff(y, n=1)
            ylagged = y[:-1]
            y=y[1:]
            
            lr = sm.OLS(y,ylagged).fit()
            beta = lr.params
            se = lr.bse
            t_stat = (beta-1)/se
    
            t_stats[n_loop,sim] = t_stat

elapsed_time = (time.time() - start)

t_stats = t_stats.T
#np.save(file = '/Users/ezravanderstelt/Desktop/python/ectri/dfcv.npy', arr = t_stats)
#t_stats = np.load('/Users/ezravanderstelt/Desktop/python/ectri/dfcv.npy')

# 10 simulations took 14.5 seconds, will 4000 simulations take 400*14.5 = 5800sec?
# this is 97 minutes. Result; 5532 sec, approximate linear relation.
# 50000 simulations took 8545 sec, no different roo's

plt.hist(t_stats[:,4], bins=100)
plt.title('Histogram t-stats')

percentiles = [1, 2.5, 5, 10, 90, 95, 97.5, 99]

for count,size in enumerate(n):
    for col,per in enumerate(percentiles):
        cv = np.percentile(t_stats[:,count],per)
        df_cv.iloc[count,col] = cv    

# =============================================================================
# SIMULATE POWER EXPERIMENT AS IN DF 1979, RECREATE RESULTS
# ALSO CHECK RESULTS WHEN YOU WOULD DO 1-SIDED 5% TEST
# AND CHECK WHAT WOULD HAPPEN WHAT WOULD HAPPEN IF YOU USED REGULAR 2SIDED TEST
# =============================================================================

N = 4000
roo = [0.8,0.9,0.95,0.99,1,1.02,1.05]
results_onesided = pd.DataFrame(0,index = n, columns = ['0.8', '0.9','0.95',
                                               '0.99','1','1.02', '1.05'])
results_twosided = pd.DataFrame(0,index = n, columns = ['0.8', '0.9','0.95',
                                               '0.99','1','1.02', '1.05'])
    
results_regular = pd.DataFrame(0,index = n, columns = ['0.8', '0.9','0.95',
                                               '0.99','1','1.02', '1.05'])
    
t_stats_powertest = np.zeros((len(n),N,len(roo)))
    
start = time.time()
for n_loop,size in enumerate(n):
    for roo_loop,p in enumerate(roo):
        for sim in range(N):
            y = AR(p=p, n=size, start = t0, mu = 0, sigma = 1)
            dy = np.diff(y, n=1)
            ylagged = y[:-1]
            y=y[1:]
            
            lr = sm.OLS(y,ylagged).fit()
            beta = lr.params
            se = lr.bse
            t_stat = (beta-1)/se
    
            t_stats_powertest[n_loop,sim,roo_loop] = t_stat

elapsed_time = (time.time() - start)
#np.save(file = '/Users/ezravanderstelt/Desktop/python/ectri/dfpower.npy', arr = t_stats_powertest)
t_stats_powertest = np.load('/Users/ezravanderstelt/Desktop/python/ectri/dfpower.npy')

# time: 4800sec

for count,size in enumerate(n):
    for col,p in enumerate(roo):
        critl = df_cv.iloc[count,1]
        crith = df_cv.iloc[count,-2]
        filt = np.logical_or(t_stats_powertest[count,:,col]<critl,
                             t_stats_powertest[count,:,col]>crith)
        reject = round(np.sum(filt)/t_stats_powertest.shape[1],3)
        results_twosided.iloc[count,col] = reject    

for count,size in enumerate(n):
    for col,p in enumerate(roo):
        critl = df_cv.iloc[count,2]
        filt = np.logical_or(t_stats_powertest[count,:,col]<critl,
                             t_stats_powertest[count,:,col]>crith)
        reject = round(np.sum(filt)/t_stats_powertest.shape[1],2)
        results_onesided.iloc[count,col] = reject    

t_cv = pd.DataFrame(np.array([[-2.06, 2.06],[-2.01, 2.01],[-1.984,1.984],[-1.96,1.96],[-1.96,1.96]]),
                    index = n, columns = ['0.025','0.975'])

for count,size in enumerate(n):
    for col,p in enumerate(roo):
        critl = t_cv.iloc[count,0]
        crith = t_cv.iloc[count,1]
        filt = np.logical_or(t_stats_powertest[count,:,col]<critl,
                             t_stats_powertest[count,:,col]>crith)
        reject = round(np.sum(filt)/t_stats_powertest.shape[1],3)
        results_regular.iloc[count,col] = reject    