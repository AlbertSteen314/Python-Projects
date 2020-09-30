#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:25:57 2020

@author: ezravanderstelt
"""

from sklearn.metrics import mean_squared_error
import scipy.stats as stats
import pandas as pd
import numpy as np
from mylib.bieb import RdmWalk
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from statsmodels.tsa.vector_ar import util
import scipy

"""
Aim: - unit root test
    - cointegration model (VECM?)
    - cointegration model estimation
    - forecast
    - evaluate forecast

Source data: treasury yield curves, yields of similar bonds for different maturities (annual)
https://home.treasury.gov/data/treasury-coupon-issues-and-corporate-bond-yield-curves/treasury-coupon-issues

Sources theory:
file:///Users/ezravanderstelt/Desktop/literatuur/FinancialEconometrics/Brooks(2008).pdf
file:///Users/ezravanderstelt/Desktop/literatuur/FinancialEconometrics/B.%20Bhaskara%20Rao%20(eds.)%20-%20Cointegration_%20for%20the%20Applied%20Economist-Palgrave%20Macmillan%20UK%20(1994).pdf
https://www.jstor.org/stable/pdf/2331330.pdf (***) replicate these results
file:///Users/ezravanderstelt/Desktop/literatuur/adv%20ectri/Richard%20I.%20D.%20Harris%20-%20Using%20Cointegration%20Analysis%20in%20Econometric%20Modelling-Prentice%20Hall%20(1995).pdf
slides ectr3 course

Richard Harris book chapter 5 and slides especially usefull
"""

# Read in and structure data
path1 = '/Users/ezravanderstelt/Desktop/python/ectri/cointegration/tnc_03_07.xls'
path2 = '/Users/ezravanderstelt/Desktop/python/ectri/cointegration/tnc_08_12.xls'
path3 = '/Users/ezravanderstelt/Desktop/python/ectri/cointegration/tnc_13_17.xls'
data1 = pd.read_excel(path1, skiprows = [0,1,2], usecols = 'A,C:BJ').T
data2 = pd.read_excel(path2, skiprows = [0,1,2], usecols = 'A,C:BJ').T
data3 = pd.read_excel(path3, skiprows = [0,1,2], usecols = 'A,C:BJ').T

data = [data1, data2, data3]
for d in data:
    d.columns = d.iloc[0]
    
data1 = data1.drop(data1.columns[1], axis=1)[1:]
data2 = data2.drop(data2.columns[1], axis=1)[1:]
data3 = data3.drop(data3.columns[1], axis=1)[1:]

merged = pd.concat([data1,data2,data3], axis=0) 
time_rng = pd.date_range('2003-01-01 10:15', periods = 180, freq = 'M')
data = merged.set_index(time_rng).drop('Maturity', axis=1)
cols = [0.5,1,3,5,7,10,30]
data = data[cols]

del data1, data2, data3, d, path1, path2, path3, merged
# Check if it is a balanced panel, make plot over time of series
data.isnull().values.any()
data.isnull().sum().sum()

#data.plot(title='Interest rates over time', figsize=(5,5)).set(xlabel="date", ylabel="interest rate")
#plt.savefig(fname='/Users/ezravanderstelt/Desktop/python/ectri/cointegration/interestrates.png')

# Check for unit roots

for col in cols:
    serie = data[col]
    adf_p = adfuller(serie)[1]
    kpss_p = kpss(serie, nlags = 'auto')[1]
    if adf_p>0.05:
        print('Series of %s' %(col) + 'has a unit root according to adf')
    else:
        print('Series of %s' %(col) + 'has no unit root according to adf')
    if kpss_p>0.05:
        print('Series of %s' %(col) + 'has no unit root according to kpss')
    else:
        print('Series of %s' %(col) + 'has a unit root according to kpss')
            
diff = data.diff()[1:]        
for col in cols:
    serie = diff[col]
    adf_p = adfuller(serie)[1]
    kpss_p = kpss(serie, nlags = 'auto')[1]
    if adf_p>0.05:
        print('Series of %s' %(col) + 'has a unit root according to adf')
    else:
        print('Series of %s' %(col) + 'has no unit root according to adf')
    if kpss_p>0.05:
        print('Series of %s' %(col) + 'has no unit root according to kpss')
    else:
        print('Series of %s' %(col) + 'has a unit root according to kpss')
        
del adf_p, col, kpss_p, serie
# Theoretically perfect outcome, all series seem to be exactly I(1)
        
yt = data[1:].copy()
Ylag = data[:-1].copy()
ydiff = data.diff(axis = 0,periods=1)[1:]

# For now assume VAR(1) process. Afterwards do lag order selection specification tests
# and rerun analysis. Notation for estimation p70 Lutk or slide 45 ectr2 class

def YZ_VAR_MATRIX(data, p):
    """Takes in dataframe data, VAR order p, and sample length T and spews out
    corresponding (p*k+1)x(T-p) stacked Z VAR matrix and Y matrix"""
    Y = np.array(data[p:].T.values.astype('float64'))
    
    T = data.shape[0]-p
    Z = []
    for i in range(T):
        z = [[1]]
        col = data.iloc[i:i+p,:].values[::-1].flatten().tolist()
        z.append(col)
        flat_z = [item for sublist in z for item in sublist]
        Z.append(flat_z)
    Zarray = np.array(Z).T
    return Y, Zarray

# Multivariate least squares estimator of VAR(1) ignoring instability and/or cointegration.

p = 1
k = data.shape[1]
T = data.shape[0]-p
Y,Z = YZ_VAR_MATRIX(data = data, p=p)    
B = (Y@Z.T)@(np.linalg.inv(Z@Z.T))
U = Y-B@Z
unbiased_Sigma = (T)/(T - B.shape[1])*(U@U.T)*T**-1
ZZ = (Z@Z.T)/T
covB = np.kron(np.linalg.inv(ZZ), unbiased_Sigma)
A1 = B[:,1:]
V = B[:,0]
#PIimplied = A-np.eye(A.shape[0])-V

# Now estimate VECM using slides and luthkepol p270, since I suspect cointegrated series.
# Apply Johansen procedure for finding cointegration rank and the corresponding cointegration
# relations. First estimate OLS B estimator, than for ML alpha, beta, sigma estimators.

D = np.ones(shape=(1,T))
dY = np.diff(data.T.astype('float64'), axis = 1)[:,p-1:]
Ylag = Z[1:,]

# LS estimation
num = np.hstack((dY@Ylag.T,dY@D.T))
denu = np.hstack((Ylag@Ylag.T,Ylag@D.T))
denl = np.hstack((D@Ylag.T,D@D.T))
den = np.vstack((denu,denl))
Bvecm = num@np.linalg.inv(den)

PI = Bvecm[:,:-1]
LSgamma = Bvecm[:,-1].reshape((k,1))

Uvecm = dY-PI@Ylag-LSgamma@D
Su = (1/T)*(Uvecm@Uvecm.T)
unbiased_Su = T/(T - 1- k*p)*(Su)

del den, denl, denu, num, ZZ

# ML p294, aim is r and ab'
M = np.eye(T) - D.T@(np.linalg.inv(D@D.T))@D
R0 = dY@M
R1 = Ylag@M

S11 = (1/T)*R1@R1.T
S10 = (1/T)*R1@R0.T
S00 = (1/T)*R0@R0.T
S01 = (1/T)*R0@R1.T

monster = fractional_matrix_power(S11, -0.5)@S10@fractional_matrix_power(S00, -1)@S01@fractional_matrix_power(S11, -0.5)
lambdas, eigvec = np.linalg.eig(monster)

# Sort lambdas and vectors from high to low
idx = lambdas.argsort()[::-1]
lambdas = lambdas[idx]
eigvec = eigvec[:,idx]

# Check cointegration rank r with trace and maxeig test, crit vals in Johansen 1995 table 15

LRtrace = -T*(sum(np.log(1-lambdas)))
LRmax = -T*(np.log(1-lambdas)[0])

LRtrace = -T*(sum(np.log(1-lambdas[1:])))
LRmax = -T*(np.log(1-lambdas[1]))

LRtrace = -T*(sum(np.log(1-lambdas[2:]))) 
LRmax = -T*(np.log(1-lambdas[2]))

LRtrace = -T*(sum(np.log(1-lambdas[3:]))) 
LRmax = -T*(np.log(1-lambdas[3]))

r=3

# Proceed given r=3, deconstruct PI in ab'

b = fractional_matrix_power(S11, -0.5)@eigvec[:,:r]
a = S01@b@np.linalg.inv(b.T@S11@b)
ab = a@b.T

# Apply "identification" strategy ab', Lutk p297 remark3

Ir = np.eye(r)
b_r = b[:r]
b_kr = (b@np.linalg.inv(b_r))[r:]
b_norm = np.vstack([Ir,b_kr])
a_norm = S01@b_norm@np.linalg.inv(b_norm.T@S11@b_norm)
ab = a_norm@b_norm.T

del LRtrace, LRmax, idx, a,b,b_r,b_kr, Ir, monster

gamma =(dY-ab@Ylag)@D.T@np.linalg.inv(D@D.T)
dYhat = ab@Ylag + gamma@D
Uhat = dY - dYhat

SS = Uhat@Uhat.T
SSR = np.sum(np.diag(SS))

# Notice differences in PI and ab, and gamma and LSgamma

Su = (1/T)*(Uhat@Uhat.T)
unbiased_Su = T/(T - 1- k*p)*(Su)

# VAR representation of VECM parameters, these are used for "coefs" parameter from now on

A1_rep = ab+np.eye(k)
V_rep = gamma.copy()
coefs = [V_rep, A1_rep]

# Model Diagnostics, autocorrelation, non normality, structural change 
# Lutk chapter 8.4 and statsmodels source code

def ACF(Uhat, lags=1):
# Uhat = kxT, note this acf returns orders 0,1,..,lags. Explosive loss of df as lags>>
    result = []
    for h in range(0,lags+1):
        if h==0:
            covh=Uhat@Uhat.T
            result.append(covh)
        else:
            covh = (Uhat[:,h:]@Uhat[:,:-h].T)
            result.append(covh)
    
    return np.hstack(result)/Uhat.shape[1]

def PortmanteauStatistic(Uhat, lags=1):
# Calculate Portmanteau test statistic for residual autocorrelation
    k=Uhat.shape[0]
    T=Uhat.shape[1]
    acf = ACF(Uhat = Uhat, lags = lags)
    C0_inv = np.linalg.inv(acf[:,0:k])
    Q=0
    for h in range(1,lags+1):
        Ci = acf[:,k*h:k*(h+1)]
        score = np.trace(Ci@C0_inv@Ci.T@C0_inv)
        Q+=score
    Q*=T
    
    return Q

Q = PortmanteauStatistic(Uhat=Uhat, lags=p)
df = p*k**2-k**2*(p-1)-k*r
dist = stats.chi2(df)
cv = dist.ppf(1-0.05)
if Q>cv:
    print('H_0: there is no autocorrelation up to lag 2, Q>cv so reject H_0')

# Jarque-Bera Normality test Chapter4 and p 175 of Lutk
# https://www.statsmodels.org/dev/_modules/statsmodels/tsa/vector_ar/var_model.html#VARResults.test_normality    

Ucentered = Uhat.T-np.mean(Uhat.T, axis=0)
Scentered = Ucentered.T@Ucentered/T
Pinv = np.linalg.inv(np.linalg.cholesky(Scentered))
w = Pinv@Ucentered.T

b1 = (w**3).sum(1) [:, None] / T
b2 = (w**4).sum(1)[:, None] / T - 3
lam_skew = (T/6) * (b1.T@b1)
lam_kurt = (T/24) * (b2.T@b2)

lam_joint = float(lam_skew + lam_kurt)
joint_dist = stats.chi2(k * 2)
joint_cv = joint_dist.ppf(1 - 0.05)

if lam_joint>joint_cv:
    print('H_0: residuals normal, stat>cv so reject H_0')

del b1,b2,lam_skew,lam_kurt,lam_joint,joint_dist,joint_cv,cv,w,dist,df,Q,Ucentered,Scentered,Pinv
# Test for absence of structural change, i.g.time invariance of DGP
# Sources of non stationarity due to trends and/or seasonallities are corrected for,
# however, big events like war, price shocks or change of legislation can also change
# parameters of economic system. 
    
"""
TO DO: STRUCTURAL TEST

Now that we have a model try forecasting and structural analysis like granger-causality tests, 
and impulse response analysis.

"""

def ForecastVAR(yt, p, coefs, h=5):
    # return 1,..,h step ahead forecasts of VAR(1), coefs=[C A1] given p initial values in yt
    # coefs must be 1+kp.
    k = yt.shape[1]
    pred = np.vstack([yt ,np.zeros((h,k))])
    for i in range(p,p+h):
        c = coefs[0].T
        A = np.zeros(k)
        for l in range(1,p+1):
            A= A + coefs[l]@pred[i-l].T
        yhat = c+A
        pred[i]=yhat
        
    return pred[p:]

def MA_Phi(coefs, p, h):
    """
    Use alternative computation Phi, slide40 ectr2 part1 2016 slides or Lutk p 23 
    equation 2.1.22. h is forecast horizon and Phi a MA coefficient matrix that is a power
    of the VAR coefficient matrix A in stacked/companion form. Code from statsmodels ma_rep()
    """
    k = coefs.shape[1]
    coefs = coefs.reshape((p,k,k))
    phis = np.zeros((h, k, k))
    phis[0] = np.eye(k)
    for i in range(1, h):
        for j in range(1, i+1):
            if j > p:
            # if j>i: ??
                break

            phis[i] += np.dot(phis[i-j], coefs[j-1])
    return phis

def ConfidenceInterval(pred, coefs, sigma, alpha=0.05):
    # Assume gaussian errors, make confidence bounds for preds.
    # slides 97/98 ectr2 and chapter 2.2.3 Lutk p.39
    # returns lower,upper confidence bounds
    
    k=pred.shape[1]
    h= pred.shape[0]
    A = np.concatenate(coefs[1:], axis=0)
    phis = MA_Phi(coefs = A, p=(len(coefs)-1), h= h)
    mse = np.zeros((h, k, k))
    prior = np.zeros((k, k))
    for h_index in range(h):
        # Sigma(h) = Sigma(h-1) + Phi Sig_u Phi'
        phi = phis[h_index]
        var = phi @ sigma @ phi.T
        mse[h_index] = prior = prior + var
    
    q = util.norm_signif_level(alpha)
    lower = np.zeros((h, k))
    upper = np.zeros((h, k))
    for h_index in range(h):
        sd = np.sqrt(np.diag(mse[h_index]))
        lower[h_index] = pred[h_index] - q*sd
        upper[h_index] = pred[h_index] + q*sd
        
    return lower, upper
        
h=12
Yt = Y.T[-p:]
pred = ForecastVAR(yt = Yt, p=p, coefs = coefs, h=h)
lower, upper = ConfidenceInterval(pred=pred, coefs=coefs, sigma=Su, alpha=0.05)

pd.DataFrame(data = pred, index=pd.date_range(time_rng[-1], periods=(h+1),
             freq='M')[1:], columns=cols).plot()
pd.DataFrame(data = np.column_stack([lower[:,0],pred[:,0],upper[:,0]]), index=pd.date_range(time_rng[-1], periods=(h+1),
             freq='M')[1:], columns=['lower','pred','upper']).plot()

# Structural analysis 
# Granger-causality test; H0 no GC between x and z comes down to testing whether
# certain elements in the coefficient matrices are 0.
# But skip, focus on impulse response analysis. Lutk p.51 and p.321 or slides77 part1.
# The shocks can be considered as 1 step ahead forecast errors. The MA psi_j matrix 
# represents the j-th time effect on the system. IRF can shed light on Granger-causality.

def IRF(colindex, system, coefs, p, mag=1, h=5):
    """Compute irf of col on system (Txk) for certain a certain number of periods.
    Unit shock in col at time 0"""
    k=system.shape[1]
    shock = np.zeros((1,k))
    shock[:,colindex] = mag    
    phis = MA_Phi(coefs=coefs, p=p, h=h)
    
    irf = np.zeros((h,k))
    for step in range(h):
        pulse = shock@phis[step]
        irf[step] = pulse
    
    return irf

h=50    
impulse_col_index = 0
irf = IRF(impulse_col_index ,data,coefs,p,h=h)

pre = np.zeros((round(h/2),k))
irf = np.vstack([pre,irf])
time = list(range(-round(h/2),h))

fig,ax =  plt.subplots(k, figsize=(7,20), sharex=True, sharey=True)
plt.xlabel("time", size='small', weight='semibold')
plt.ylabel("effect", size='small', va='center', weight='semibold')
fig.patch.set_facecolor('grey')
fig.patch.set_alpha(0.4)
for i,col in enumerate(cols):
    ax[i].plot(time, irf[:,i])
    ax[i].set_title(col, fontsize = 7)
    ax[i].set_facecolor('salmon')
    ax[i].patch.set_alpha(0.4)
fig.tight_layout(h_pad = 0.5) 
# https://stackoverflow.com/questions/14088687/how-to-change-plot-background-color

lags = [0,1,2,3,4,5]
aic_full = pd.DataFrame(np.zeros(len(lags)), index =lags ,dtype=float)

for i,p in enumerate(lags):
    Y,Z = YZ_VAR_MATRIX(data = data, p=p)
    B = (Y@Z.T)@(np.linalg.inv(Z@Z.T))    
    U = Y-B@Z
    T = 180-p
    params = p*(k**2)
    sigma = 1/T*(U@U.T)
    AIC = np.log(np.linalg.det(sigma))+(2/T)*params
    aic_full.iloc[i,0] = AIC
   
""" 
According to AIC lag order VAR should be 3, redo analysis see if everything works
and compare model diagnostics.
"""

def dYX_VECM_MATRIX(data, p):
    """Takes in dataframe data, VAR order p, and spews out
    corresponding (p*k+1)x(T-p) stacked Z VAR matrix and Y matrix"""
    d=data.diff()
    dY = np.array(d[p:].T.values.astype('float64'))
    
    T = data.shape[0]-p
    Z = []
    for i in range(T):
        z = [[1]]
        col = d.iloc[i+1:i+p,:].values[::-1].flatten().tolist()
        z.append(col)
        flat_z = [item for sublist in z for item in sublist]
        Z.append(flat_z)
    dX = np.array(Z).T
    return dY, dX

p = 3
k = data.shape[1]
T = data.shape[0]-p
Y,Z = YZ_VAR_MATRIX(data = data, p=p)    
Ylag = Z[1:k+1,:].copy()
Ylag2 = Z[k+1:2*k+1,:].copy()
Ylag3 = Z[2*k+1:3*k+1,:].copy()
D=Z[1,:]
dY, dX = dYX_VECM_MATRIX(data= data, p=p)

# LS estimation
num = np.hstack((dY@Ylag.T,dY@dX.T))
denu = np.hstack((Ylag@Ylag.T,Ylag@dX.T))
denl = np.hstack((dX@Ylag.T,dX@dX.T))
den = np.vstack((denu,denl))
Bvecm = num@np.linalg.inv(den)

PI = Bvecm[:,:k]

del den, denl, denu, num

# ML p294, aim is r and ab'
M = np.eye(T) - dX.T@(np.linalg.inv(dX@dX.T))@dX
R0 = dY@M
R1 = Ylag@M

S11 = (1/T)*R1@R1.T
S10 = (1/T)*R1@R0.T
S00 = (1/T)*R0@R0.T
S01 = (1/T)*R0@R1.T

monster = fractional_matrix_power(S11, -0.5)@S10@fractional_matrix_power(S00, -1)@S01@fractional_matrix_power(S11, -0.5)
lambdas, eigvec = np.linalg.eig(monster)

# Sort lambdas and vectors from high to low, note this time we need to switch order of data
idx = lambdas.argsort()[::-1]
lambdas = lambdas[idx]
eigvec = eigvec[:,idx]

# Check cointegration rank r with trace and maxeig test, crit vals in Johansen 1995 table 15
from statsmodels.tsa.coint_tables import c_sja, c_sjt
LRtrace = -T*(sum(np.log(1-lambdas)))
LRmax = -T*(np.log(1-lambdas)[0])
c_sjt(7,0)
c_sja(7,0)

LRtrace = -T*(sum(np.log(1-lambdas[1:])))
LRmax = -T*(np.log(1-lambdas[1]))
c_sjt(6,0)
c_sja(6,0)

LRtrace = -T*(sum(np.log(1-lambdas[2:]))) 
LRmax = -T*(np.log(1-lambdas[2]))
c_sjt(5,0)
c_sja(5,0)

r=2

# Proceed given r=2, deconstruct PI in ab'

b = fractional_matrix_power(S11, -0.5)@eigvec[:,:r]
a = S01@b@np.linalg.inv(b.T@S11@b)
ab = a@b.T

# Apply "identification" strategy ab', Lutk p297 remark3

Ir = np.eye(r)
b_r = b[:r]
b_kr = (b@np.linalg.inv(b_r))[r:]
b_norm = np.vstack([Ir,b_kr])
a_norm = S01@b_norm@np.linalg.inv(b_norm.T@S11@b_norm)
ab = a_norm@b_norm.T

del LRtrace, LRmax, idx, a,b,b_r,b_kr, Ir, monster

gamma =(dY-ab@Ylag)@dX.T@np.linalg.inv(dX@dX.T)
V=gamma[:,0]
gamma1=gamma[:,1:k+1]
gamma2=gamma[:,k+1:]
dYhat = ab@Ylag + gamma@dX
Uhat = dY - dYhat

# Notice differences in PI and ab, and gamma and LSgamma

Su = (1/T)*(Uhat@Uhat.T)
unbiased_Su = T/(T - 1- k*p)*(Su)

def SigmaParameters(Ylag, dX, Su, b_norm):
    # Based on chapter7 Lutk p.287 and statsmodels source code. Question about sigma on:
    #https://stats.stackexchange.com/questions/488829/covariance-matrix-computation-vecm-lutkepohl-2005-p-287
    T = Ylag.shape[1]
    k= Ylag.shape[0]
    p= int(((dX.shape[0]-1)/k)+1)
    
    b_id = scipy.linalg.block_diag(b_norm, np.eye(k*(p-1)+1))
    S00 = dX@Ylag.T@b_norm
    S10 = dX@dX.T
    S11 = b_norm.T@Ylag@dX.T
    S01 = b_norm.T@Ylag@Ylag.T@b_norm
    S = np.vstack([np.hstack([S01,S11]),np.hstack([S00,S10])])
    
    # alternative:
#    S00 = dX@Ylag.T
#    S10 = dX@dX.T
#    S11 = Ylag@dX.T
#    S01 = Ylag@Ylag.T
#    S = T*np.linalg.inv(np.vstack([np.hstack([S01,S11]),np.hstack([S00,S10])]))
    
    Sco = np.kron(b_id@np.linalg.inv(S)@b_id.T, Su)
#    Scoo = np.kron(S, Su)
    return Sco

# Slight differences between Su and vecm.sigma_u from 10th decimal,
# has small implications in later calculations like here with Sco and vecm.cov_params_default
# Though it seems like same numbers, but different order, weird

Sco = SigmaParameters(Ylag = Ylag, dX = dX, Su=Su, b_norm = b_norm)
Se_params = [np.diag(Sco[:k**2,:k**2]),np.diag(Sco[k**2:k**2+k,k**2:k**2+k]),
                              np.diag(Sco[k**2+k:2*k**2+k,k**2+k:2*k**2+k]),
                              np.diag(Sco[2*k**2+k:3*k**2+k,2*k**2+k:3*k**2+k])]

se_ab = np.sqrt(Se_params[0]).reshape((k,k)).T
se_v = np.sqrt(Se_params[1]).T
se_g1 = np.sqrt(Se_params[2]).reshape((k,k)).T
se_g2 = np.sqrt(Se_params[3]).reshape((k,k)).T

# VAR representation of VECM parameters (p), these are used for "coefs" parameter from now on

A1_rep = ab+np.eye(k)+gamma1
A2_rep = gamma2-gamma1
A3_rep = -gamma2
V_rep = V.copy()
coefs = [V_rep, A1_rep, A2_rep, A3_rep]
Yhat = coefs[0].reshape((k,1))@D.reshape((1,T)) + coefs[1]@Ylag + coefs[2]@Ylag2 + coefs[3]@Ylag3

lags = 5
Q = PortmanteauStatistic(Uhat=Uhat, lags=lags)
df = lags*k**2-k**2*(p-1)-k*r
dist = stats.chi2(df)
cv = dist.ppf(1-0.05)
if Q>cv:
    print('H_0: there is no autocorrelation up to lag 2, Q>cv so reject H_0')

Ucentered = Uhat.T-np.mean(Uhat.T, axis=0)
Scentered = Ucentered.T@Ucentered/T
Pinv = np.linalg.inv(np.linalg.cholesky(Scentered))
w = Pinv@Ucentered.T

b1 = (w**3).sum(1) [:, None] / T
b2 = (w**4).sum(1)[:, None] / T - 3
lam_skew = (T/6) * (b1.T@b1)
lam_kurt = (T/24) * (b2.T@b2)

lam_joint = float(lam_skew + lam_kurt)
joint_dist = stats.chi2(k * 2)
joint_cv = joint_dist.ppf(1 - 0.05)

if lam_joint>joint_cv:
    print('H_0: residuals normal, stat>cv so reject H_0')

del b1,b2,lam_skew,lam_kurt,lam_joint,joint_dist,joint_cv,cv,w,dist,df,Q,Ucentered,Scentered,Pinv
    
SS = Uhat@Uhat.T
SSR = np.sum(np.diag(SS))

# Contrast models by forecasting y0.5 till today.

def RollingOneStepForecast(yt, future, p, coefs, sigma, h=5, col = 0):
    k = yt.shape[1]
    rolling_lower = np.zeros((h,k))
    rolling_upper = np.zeros((h,k))
    rolling_pred = np.zeros((h,k))
    dat = np.vstack([yt,future[:h]])
    for i in range(h):
        pred = ForecastVAR(yt=dat[i:i+p], p=p, coefs=coefs, h=1)    
        l,h= ConfidenceInterval(pred= pred, coefs=coefs, sigma=Su, alpha=.05)
        rolling_pred[i] = pred
        rolling_lower[i] = l
        rolling_upper[i] = h
    roll = np.column_stack([rolling_lower[:,col], rolling_pred[:,col], rolling_upper[:,col]]).astype(float)
    return roll

forecast_performance = pd.DataFrame(data= np.zeros((3,3)), index=['VECM1','VECM3','VAR3'], columns = ['MSE','MAE','Sign'])

path = '/Users/ezravanderstelt/Desktop/python/ectri/cointegration/tnc_18_22.xls'
data1 = pd.read_excel(path, skiprows = [0,1,2], usecols = 'A,C:BJ').T
data1.columns = data1.iloc[0]
data1 = data1.drop(data1.columns[1], axis=1)[1:33]
time_rng_fc = pd.date_range('2018-01-01 10:15', periods = 32, freq = 'M')
data1 = data1.set_index(time_rng_fc).drop('Maturity', axis=1)
data1 = data1[cols]
del path

p = 1
r=3
k = data.shape[1]
T = data.shape[0]-p
Y,Z = YZ_VAR_MATRIX(data = data, p=p)    
D = np.ones(shape=(1,T))
dY = np.diff(data.T.astype('float64'), axis = 1)[:,p-1:]
Ylag = Z[1:,]
M = np.eye(T) - D.T@(np.linalg.inv(D@D.T))@D
R0 = dY@M
R1 = Ylag@M
S11 = (1/T)*R1@R1.T
S10 = (1/T)*R1@R0.T
S00 = (1/T)*R0@R0.T
S01 = (1/T)*R0@R1.T
monster = fractional_matrix_power(S11, -0.5)@S10@fractional_matrix_power(S00, -1)@S01@fractional_matrix_power(S11, -0.5)
lambdas, eigvec = np.linalg.eig(monster)
idx = lambdas.argsort()[::-1]
lambdas = lambdas[idx]
eigvec = eigvec[:,idx]
b = fractional_matrix_power(S11, -0.5)@eigvec[:,:r]
a = S01@b@np.linalg.inv(b.T@S11@b)
ab = a@b.T
Ir = np.eye(r)
b_r = b[:r]
b_kr = (b@np.linalg.inv(b_r))[r:]
b_norm = np.vstack([Ir,b_kr])
a_norm = S01@b_norm@np.linalg.inv(b_norm.T@S11@b_norm)
ab = a_norm@b_norm.T
gamma =(dY-ab@Ylag)@D.T@np.linalg.inv(D@D.T)
dYhat = ab@Ylag + gamma@D
Uhat = dY - dYhat

A1_rep = ab+np.eye(k)
V_rep = gamma.copy()
coefs = [V_rep, A1_rep]
Su = (1/T)*(Uhat@Uhat.T)
Yt = Y.T[-p:]

h=data1.shape[0]
cols_fc = ['low','fc','top','real']
rolling_fc1 = RollingOneStepForecast(yt = Yt, future=data1, p=p, coefs = coefs, sigma=Su, h=h)
fc1= pd.DataFrame(np.column_stack([rolling_fc1, data1.iloc[:,0].values]), index=time_rng_fc[:h], columns=cols_fc).astype(float)

mse = ((fc1.real - fc1.fc)**2).mean()
mae = abs((fc1.real - fc1.fc)).mean()
helper = np.concatenate([[Yt[-1,0]], data1.iloc[:,0].values]).astype('float')
psignfc = (fc1.fc-helper[:-1])>0
psignreal = np.diff(helper)>0
sign = sum(psignfc==psignreal)/h
forecast_performance.iloc[0,0] = mse
forecast_performance.iloc[0,1] = mae
forecast_performance.iloc[0,2] = sign

p = 3
r=2
T = data.shape[0]-p
Y,Z = YZ_VAR_MATRIX(data = data, p=p)    
dY, dX = dYX_VECM_MATRIX(data= data, p=p)
Ylag = Z[1:k+1,:].copy()
Ylag2 = Z[k+1:2*k+1,:].copy()
Ylag3 = Z[2*k+1:3*k+1,:].copy()
D=Z[1,:]
M = np.eye(T) - dX.T@(np.linalg.inv(dX@dX.T))@dX
R0 = dY@M
R1 = Ylag@M
S11 = (1/T)*R1@R1.T
S10 = (1/T)*R1@R0.T
S00 = (1/T)*R0@R0.T
S01 = (1/T)*R0@R1.T
monster = fractional_matrix_power(S11, -0.5)@S10@fractional_matrix_power(S00, -1)@S01@fractional_matrix_power(S11, -0.5)
lambdas, eigvec = np.linalg.eig(monster)
idx = lambdas.argsort()[::-1]
lambdas = lambdas[idx]
eigvec = eigvec[:,idx]
b = fractional_matrix_power(S11, -0.5)@eigvec[:,:r]
a = S01@b@np.linalg.inv(b.T@S11@b)
ab = a@b.T
Ir = np.eye(r)
b_r = b[:r]
b_kr = (b@np.linalg.inv(b_r))[r:]
b_norm = np.vstack([Ir,b_kr])
a_norm = S01@b_norm@np.linalg.inv(b_norm.T@S11@b_norm)
ab = a_norm@b_norm.T
gamma =(dY-ab@Ylag)@dX.T@np.linalg.inv(dX@dX.T)
V=gamma[:,0]
gamma1=gamma[:,1:k+1]
gamma2=gamma[:,k+1:]
dYhat = ab@Ylag + gamma@dX
Uhat = dY - dYhat

A1_rep = ab+np.eye(k)+gamma1
A2_rep = gamma2-gamma1
A3_rep = -gamma2
V_rep = V.copy()
coefs = [V_rep, A1_rep, A2_rep, A3_rep]
Yhat = coefs[0].reshape((k,1))@D.reshape((1,T)) + coefs[1]@Ylag + coefs[2]@Ylag2 + coefs[3]@Ylag3
Yt = Y.T[-p:]

rolling_fc3 = RollingOneStepForecast(yt = Yt, future=data1, p=p, coefs = coefs, sigma=Su, h=h)
fc3= pd.DataFrame(np.column_stack([rolling_fc3, data1.iloc[:,0].values]), index=time_rng_fc[:h], columns=cols_fc).astype(float)

mse = ((fc3.real - fc3.fc)**2).mean()
mae = abs((fc3.real - fc3.fc)).mean()
helper = np.concatenate([[Yt[-1,0]], data1.iloc[:,0].values]).astype('float')
psignfc = (fc3.fc-helper[:-1])>0
psignreal = np.diff(helper)>0
sign = sum(psignfc==psignreal)/h
forecast_performance.iloc[1,0] = mse
forecast_performance.iloc[1,1] = mae
forecast_performance.iloc[1,2] = sign

# VAR3 ignoring cointegration
p = 3
T = data.shape[0]-p
Y,Z = YZ_VAR_MATRIX(data = data, p=p)    
dY, dX = dYX_VECM_MATRIX(data= data, p=p)
dYlag = dX[1:k+1,:].copy()
dYlag2 = dX[k+1:2*k+1,:].copy()
D=dX[1,:]
B = (dY@dX.T)@(np.linalg.inv(dX@dX.T))
U = dY-B@dX
Su = (U@U.T)*T**-1
V = B[:,1]
A1_rep = B[:,1:k+1]+np.eye(k)
A2_rep = B[:,1:k+1] - B[:,k+1:2*k+1]
A3_rep = - B[:,k+1:2*k+1]
coefs = [V, A1_rep, A2_rep, A3_rep]
Yt = Y.T[-p:]

rolling_fc2 = RollingOneStepForecast(yt = Yt, future=data1, p=p, coefs = coefs, sigma=Su, h=h)
fc2= pd.DataFrame(np.column_stack([rolling_fc2, data1.iloc[:,0].values]), index=time_rng_fc[:h], columns=cols_fc).astype(float)

mse = ((fc2.real - fc2.fc)**2).mean()
mae = abs((fc2.real - fc2.fc)).mean()
helper = np.concatenate([[Yt[-1,0]], data1.iloc[:,0].values]).astype('float')
psignfc = (fc2.fc-helper[:-1])>0
psignreal = np.diff(helper)>0
sign = sum(psignfc==psignreal)/h
forecast_performance.iloc[2,0] = mse
forecast_performance.iloc[2,1] = mae
forecast_performance.iloc[2,2] = sign

pd.DataFrame(data = fc1, index=time_rng_fc, columns=cols_fc).plot().fill_between(x = time_rng_fc[:h], y1 = fc1.iloc[:,0],y2 = fc1.iloc[:,2], alpha=0.2)
pd.DataFrame(data = fc3, index=time_rng_fc, columns=cols_fc).plot().fill_between(x = time_rng_fc[:h], y1 = fc3.iloc[:,0],y2 = fc3.iloc[:,2], alpha=0.2)




"""
compare forecast error of vecm3, vecm1, var3, var1 for mse and mae
"""


"""
Theory; If variables are I(0) you can set up a VAR. If they are I(1) and have no
cointegrating relation, set up a VAR with differenced data. If I(1) and cointegrated
you should set up a VECM. The b'Y term in a VECM represents a long run equilibrium
relation and its corresponding a the long run effect, the other terms are the 
short term effects.

First step is to determine the optimal lag length p.

When you suspect cointegration, the Johansen method proceeds by setting up a VAR, 
transforming it to a VECM, and then examine the coefficients matrix PI rank by 
constructing maximum likelihood like tests, the trace and max eigenvalue test.

The cointegration tests are based on the (kxk) matrix Skk^-1S10S00^-1S01Skk^-1/2,
NOT on the least squares estimate of PI (why?).

For point forecasts no need of distributional assumptions, for confidence bounds we do,
typically assume gaussian errors. 
    
"""

from statsmodels.tsa.api import VAR
var = VAR(data.values.astype('float64')).fit(maxlags = 1, ic='aic', trend='c')
var.k_ar
var.intercept_longrun()
var = VAR(data.values.astype('float64')).select_order(5, trend='c')
print(var.summary())
var.params
var.coefs
pred = var.forecast(y=last_observations, steps=h, exog_future=exog_future)

from statsmodels.tsa.vector_ar.vecm import select_coint_rank
from statsmodels.tsa.coint_tables import c_sja, c_sjt
coint = select_coint_rank(data.values.astype('float64'), det_order = 0, k_ar_diff = 2, method= 'maxeig')
coint.rank
coint.test_stats
coint.crit_vals
c_sja(7,0)

from statsmodels.tsa.api import VECM
vecm = VECM(data.values.astype('float64'), k_ar_diff = 2, deterministic = 'co', coint_rank=2).fit()
vecm.k_ar
vecm.coint_rank
vecm.coefs
vecmSu = vecm.sigma_u
print(vecm.summary())
vecma = vecm.alpha
vecmb = vecm.beta
vecmg = vecm.gamma
vecmdet = vecm.cov_params_wo_det
vecms = vecm.cov_params_default
vecmstd = vecm.stderr_params
stdab = vecmstd[:k**2].reshape((k,k)).T
stda = vecm.stderr_alpha
stdb = vecm.stderr_beta
stdg = vecm.stderr_gamma
stdv = vecm.stderr_det_coef
resid = vecm.resid
vecm.test_whiteness(nlags = 2, signif=0.05).test_statistic
vecm.test_whiteness(nlags = 2, signif=0.05).crit_value
print(vecm.test_whiteness(nlags = 10, signif=0.05).summary())
pred0, pred1, pred2 = vecm.predict(steps=12, alpha=0.05)
vecm.plot_forecast(steps = 20, alpha=0.05)

irf1 = vecm.irf(periods = h)
irf1.plot(impulse = 0)
abc = irf1.irfs
abcd = irf1.cum_effects

pd.DataFrame(np.column_stack((dY.T[:,-1],dYhat.T[:,-1])), index=time_rng[1:], columns=['dY','dYhat']).plot()
pd.DataFrame((ab@Ylag).T, index=time_rng[1:], columns = cols).plot()
pd.DataFrame((gamma@D).T, index=time_rng[1:], columns = cols).plot()
pd.DataFrame(dYhat.T[:,:3], index=time_rng[1:], columns = cols[:3]).plot()
pd.DataFrame(dYhat.T[:,3:], index=time_rng[1:], columns = cols[3:]).plot()
pd.DataFrame((b_norm.T@Ylag).T, index=time_rng[1:]).plot()
pd.DataFrame(dY.T, index=time_rng[1:], columns = cols).plot()
pd.DataFrame(Uhat.T, index=time_rng[p:], columns = cols).plot()

