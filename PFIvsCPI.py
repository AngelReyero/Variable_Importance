#%%
import argparse
import pickle
import time
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BBI_package.src.BBI import BlockBasedImportance
from joblib import Parallel, delayed
from scipy.linalg import cholesky
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
import vimpy
#from .utils import compute_loco
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
seed=2024


#%%

##y=B1 x1+ B2 x2+ 0*x3+ eps with x3 highly correlated 


snr=4
p=3
n=15000
intra_cor=0.8
cor_mat=np.zeros((p,p))
cor_mat[0:p,0:p]=intra_cor
np.fill_diagonal(cor_mat, 1)
x = norm.rvs(size=(p, n), random_state=seed)
c = cholesky(cor_mat, lower=True)
data = pd.DataFrame(np.dot(c, x).T, columns=[str(i) for i in np.arange(p)])
data_enc = data.copy()

rng = np.random.RandomState(seed)
n_signal=2
data_enc_a = data_enc.iloc[:, np.arange(n_signal)]
tmp_comb = data_enc_a.shape[1]

    # Determine beta coefficients
effectset = [-0.5, -1, -2, -3, 0.5, 1, 2, 3]
beta = rng.choice(effectset, size=(tmp_comb), replace=True)

    # Generate response
    ## The product of the signal predictors with the beta coefficients
prod_signal = np.dot(data_enc_a, beta)

sigma_noise = np.linalg.norm(prod_signal, ord=2) / (
            snr * np.sqrt(data_enc_a.shape[0])
        )
y = prod_signal + sigma_noise * rng.normal(size=prod_signal.shape[0])


# %%
inter=[100,200, 350, 500, 750, 1000, 2000, 3500, 5000, 7500,9000, 11000, 13000, 15000]
imp=np.zeros((3, len(inter), 3))
pval=np.zeros((3, len(inter), 3))
for (i,n) in enumerate(inter): 
    print("With n="+str(n))
    #Conditional
    bbi_model = BlockBasedImportance(
            estimator=None,
            do_hyper=True,
            importance_estimator="Mod_RF",
            dict_hyper=None,
            conditional=True,
            group_stacking=False,
            n_perm=100,
            n_jobs=10,
            prob_type="regression",
            k_fold=2,
        )
    bbi_model.fit(data_enc[0:n], y[0:n])
    res_CPI = bbi_model.compute_importance()
    imp[0,i]=res_CPI["importance"].reshape((3,))
    pval[0,i]=res_CPI["pval"].reshape((3,))
    #PFI
    bbi_model2 = BlockBasedImportance(
            estimator=None,
            do_hyper=True,
            importance_estimator="Mod_RF",
            dict_hyper=None,
            conditional=False,
            group_stacking=False,
            n_perm=100,
            n_jobs=10,
            prob_type="regression",
            k_fold=2,
        )
    bbi_model2.fit(data_enc[0:n], y[0:n])
    res_PFI = bbi_model2.compute_importance()
    imp[1,i]=res_PFI["importance"].reshape((3,))
    pval[1,i]=res_PFI["pval"].reshape((3,))
    #LOCO
    ntrees = np.arange(100, 500, 100)
    lr = np.arange(.01, .1, .05)
    param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
    ## set up cv objects
    cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 1), param_grid = param_grid, cv = 5, n_jobs=10)
    for j in range(3):
        print("covariate: "+str(j))
        vimp = vimpy.vim(y = y[0:n], x = data_enc.values[0:n], s = j, pred_func = cv_full, measure_type = "r_squared")
        vimp.get_point_est()
        vimp.get_influence_function()
        vimp.get_se()
        vimp.get_ci()
        vimp.hypothesis_test(alpha = 0.05, delta = 0)
        imp[2,i,j]=vimp.vimp_
        pval[2,i, j]=vimp.p_value_

#%%
f_res={}
f_res = pd.DataFrame(f_res)
for i in range(3):#CPI, PFI, LOCO
    for j in range(len(inter)):
        f_res1={}
        if i==0:
            f_res1["method"] = ["CPI"]
        elif i==1:
            f_res1["method"]=["PFI"]
        else: 
            f_res1["method"]=["LOCO"]
        f_res1["n"]=inter[j]
        for k in range(len(list(data.columns))):
            f_res1["imp_V"+str(k)]=imp[i, j, k]
            f_res1["pval_V"+str(k)]=pval[i, j, k]
        f_res1=pd.DataFrame(f_res1)
        f_res=pd.concat([f_res, f_res1], ignore_index=True)
f_res.to_csv(
    f"results/results_csv_Angel/simulation_PFIvsCPIvsLOCO_conv-rates.csv",
    index=False,
) 


#%%

# Visualization from the csv file

res_path = pathlib.Path('results/results_csv_Angel')
list(res_path.glob('*.csv'))

df = pd.read_csv(res_path/"simulation_PFIvsCPIvsLOCO_conv-rates.csv")

p=3# Number of covariates
fig, axs = plt.subplots(2,p)
fig.suptitle("Convergence rates of CPI vs PFI vs LOCO",fontsize=16)
for i in range(p):
    for method, group in df.groupby('method'):
        axs[0, i].plot(group['n'], group['imp_V'+str(i)], label=method)
        axs[1, i].plot(group['n'], -np.log10(group['pval_V'+str(i)]+1e-10), label=method)
    axs[0, i].set_title(r'Importance $x$'+str(i), fontsize=14)
    axs[1,i].axhline(y=-np.log10(0.05), color='r', linestyle='--', linewidth=1)
    axs[1, i].set_title(r'-log10(p_value) $x$'+str(i), fontsize=14)
    axs[0,i].legend()
    axs[1,i].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
fig.text(0.53, 0, r'$n$', ha='center', va='center')
#fig.text(0.53, -0.05, " ", ha='center', va='center')
fig.savefig("visualization/plots_Angel/CPIvsPFIvsLOCO_conv-rates.pdf", bbox_inches="tight")



# %%

fig, axs = plt.subplots(2,p)
fig.suptitle("Convergence rates of CPI vs PFI",fontsize=16)
for i in range(p):
    axs[0, i].plot(inter, imp[0,:,i], label="CPI", linestyle="dashed")
    axs[0, i].plot(inter, imp[1,:,i], label="PFI")
    axs[0, i].set_title(r'Importance $x$'+str(i), fontsize=14)
    axs[1, i].plot(inter, -np.log10(pval[0,:,i]+1e-10), label="CPI", linestyle="dashed")
    axs[1, i].plot(inter, -np.log10(pval[1,:,i]+1e-10), label="PFI")
    axs[1,i].axhline(y=-np.log10(0.05), color='r', linestyle='-', label=r"$-\mathrm{log}_{10}(\alpha)$")
    axs[1, i].set_title(r'-log10(p_value) $x$'+str(i), fontsize=14)
    axs[0,i].legend()
    axs[1,i].legend()

fig.text(0.53, 0, r'$n$', ha='center', va='center')
plt.tight_layout(rect=[0, 0, 1, 0.95]) 
fig.savefig("CPIvsPFI_conv-rates.pdf")


#%%

## Different correlations

snr=4
p=3
n=1000
x = norm.rvs(size=(p, n), random_state=seed)
intra_cor=[0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.65, 0.8, 0.9]
imp2=np.zeros((3, len(intra_cor), 3))
pval2=np.zeros((3, len(intra_cor), 3))
 # Determine beta coefficients
rng = np.random.RandomState(seed)
n_signal=2
effectset = [-0.5, -1, -2, -3, 0.5, 1, 2, 3]
beta = rng.choice(effectset, size=(n_signal), replace=True)


for (i,cor) in enumerate(intra_cor):
    print("With correlation="+str(cor))
    #First we construct the sample with the third useless covariate with correlation=cor
    cor_mat=np.zeros((p,p))
    cor_mat[0:p,0:p]=cor
    np.fill_diagonal(cor_mat, 1)

    c = cholesky(cor_mat, lower=True)
    data = pd.DataFrame(np.dot(c, x).T, columns=[str(i) for i in np.arange(p)])
    data_enc = data.copy()
    data_enc_a = data_enc.iloc[:, np.arange(n_signal)]

       

    # Generate response
    ## The product of the signal predictors with the beta coefficients
    prod_signal = np.dot(data_enc_a, beta)

    sigma_noise = np.linalg.norm(prod_signal, ord=2) / (
                snr * np.sqrt(data_enc_a.shape[0])
            )
    y = prod_signal + sigma_noise * rng.normal(size=prod_signal.shape[0]) 
    
    #Conditional
    bbi_model = BlockBasedImportance(
            estimator=None,
            do_hyper=True,
            importance_estimator="Mod_RF",
            dict_hyper=None,
            conditional=True,
            group_stacking=False,
            n_perm=100,
            n_jobs=10,
            prob_type="regression",
            k_fold=2,
        )
    bbi_model.fit(data_enc, y)
    res_CPI = bbi_model.compute_importance()
    imp2[0,i]=res_CPI["importance"].reshape((3,))
    pval2[0,i]=res_CPI["pval"].reshape((3,))
    #PFI
    bbi_model2 = BlockBasedImportance(
            estimator=None,
            do_hyper=True,
            importance_estimator="Mod_RF",
            dict_hyper=None,
            conditional=False,
            group_stacking=False,
            n_perm=100,
            n_jobs=10,
            prob_type="regression",
            k_fold=2,
        )
    bbi_model2.fit(data_enc, y)
    res_PFI = bbi_model2.compute_importance()
    imp2[1,i]=res_PFI["importance"].reshape((3,))
    pval2[1,i]=res_PFI["pval"].reshape((3,))
    #LOCO
    ntrees = np.arange(100, 500, 100)
    lr = np.arange(.01, .1, .05)
    param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
    ## set up cv objects
    cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 1), param_grid = param_grid, cv = 5, n_jobs=10)
    for j in range(3):
        print("covariate: "+str(j))
        vimp = vimpy.vim(y = y, x = data_enc.values, s = j, pred_func = cv_full, measure_type = "r_squared")
        vimp.get_point_est()
        vimp.get_influence_function()
        vimp.get_se()
        vimp.get_ci()
        vimp.hypothesis_test(alpha = 0.05, delta = 0)
        imp2[2,i,j]=vimp.vimp_
        pval2[2,i, j]=vimp.p_value_


#%%
f_res={}
f_res = pd.DataFrame(f_res)
for i in range(3):#CPI, PFI, LOCO
    for j in range(len(intra_cor)):
        f_res1={}
        if i==0:
            f_res1["method"] = ["CPI"]
        elif i==1:
            f_res1["method"]=["PFI"]
        else: 
            f_res1["method"]=["LOCO"]
        f_res1["intra_cor"]=intra_cor[j]
        for k in range(len(list(data.columns))):
            f_res1["imp_V"+str(k)]=imp2[i, j, k]
            f_res1["pval_V"+str(k)]=pval2[i, j, k]
        f_res1=pd.DataFrame(f_res1)
        f_res=pd.concat([f_res, f_res1], ignore_index=True)
f_res.to_csv(
    f"results/results_csv_Angel/simulation_PFIvsCPIvsLOCO_diff-corr.csv",
    index=False,
) 


#%%

# Visualization from the csv file

res_path = pathlib.Path('results/results_csv_Angel')
list(res_path.glob('*.csv'))

df = pd.read_csv(res_path/"simulation_PFIvsCPIvsLOCO_diff-corr.csv")

p=3# Number of covariates
fig, axs = plt.subplots(2,p)
fig.suptitle("Different correlations with CPI vs PFI vs LOCO",fontsize=16)
for i in range(p):
    for method, group in df.groupby('method'):
        axs[0, i].plot(group['intra_cor'], group['imp_V'+str(i)], label=method)
        axs[1, i].plot(group['intra_cor'], -np.log10(group['pval_V'+str(i)]+1e-10), label=method)
    axs[0, i].set_title(r'Importance $x$'+str(i), fontsize=14)
    axs[1,i].axhline(y=-np.log10(0.05), color='r', linestyle='--', linewidth=1)
    axs[1, i].set_title(r'-log10(p_value) $x$'+str(i), fontsize=14)
    axs[0,i].legend()
    axs[1,i].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
fig.text(0.53, 0, r'$\rho$', ha='center', va='center')

fig.savefig("visualization/plots_Angel/CPIvsPFIvsLOCO_diff-corr.pdf", bbox_inches="tight")

# %%

# Plots
fig, axs = plt.subplots(2,p)
fig.suptitle("Correlation effects on CPI vs PFI",fontsize=16)
for i in range(p):
    axs[0, i].plot(intra_cor, imp2[0,:,i], label="CPI", linestyle="dashed")
    axs[0, i].plot(intra_cor, imp2[1,:,i], label="PFI")
    axs[0, i].set_title(r'Importance $x$'+str(i), fontsize=14)
    axs[1, i].plot(intra_cor, -np.log10(pval2[0,:,i]+1e-10), label="CPI", linestyle="dashed")
    axs[1, i].plot(intra_cor, -np.log10(pval2[1,:,i]+1e-10), label="PFI")
    axs[1,i].axhline(y=-np.log10(0.05), color='r', linestyle='-',  label=r"$-\mathrm{log}_{10}(\alpha)$")
    axs[1, i].set_title(r'-log10(p_value) $x$'+str(i), fontsize=14)
    axs[0,i].legend()
    axs[1,i].legend()

fig.text(0.53, 0, r'$\rho$', ha='center', va='center')

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
fig.savefig("CPIvsPFI_corr.pdf")



# %%
# With Noise/innovation
## y=B1 X1+ B2 X2 + B3 (X1*X2+noise*eps1)+eps2

snr=4
p=3
n=10000
x = norm.rvs(size=(p, n), random_state=seed)
noise_arr=[0,0.02, 0.05, 0.1,0.2, 0.35, 0.5, 0.75, 1, 2]

imp3=np.zeros((3, len(noise_arr), 3))
pval3=np.zeros((3, len(noise_arr), 3))
 # Determine beta coefficients
rng = np.random.RandomState(seed)
n_signal=3
effectset = [-0.5, -1, -2, -3, 0.5, 1, 2, 3]
beta = rng.choice(effectset, size=(n_signal), replace=True)


for (i,noise) in enumerate(noise_arr):
    print("With noise="+str(noise))
    x_2=x.copy()
    x_2[2]=x[0]*x[1]+noise*x[2]
    #First we construct the sample with the third covariate X_1*X_2+noise*X_3
    data = pd.DataFrame(x_2.T, columns=[str(i) for i in np.arange(p)])
    data_enc = data.copy()
    data_enc_a = data_enc.iloc[:, np.arange(n_signal)]

    # Generate response
    ## The product of the signal predictors with the beta coefficients
    prod_signal = np.dot(data_enc_a, beta)

    sigma_noise = np.linalg.norm(prod_signal, ord=2) / (
                snr * np.sqrt(data_enc_a.shape[0])
            )
    y = prod_signal + sigma_noise * rng.normal(size=prod_signal.shape[0]) 
    
    #Conditional
    bbi_model = BlockBasedImportance(
            estimator=None,
            do_hyper=True,
            importance_estimator="Mod_RF",
            dict_hyper=None,
            conditional=True,
            group_stacking=False,
            n_perm=100,
            n_jobs=10,
            prob_type="regression",
            k_fold=2,
        )
    bbi_model.fit(data_enc, y)
    res_CPI = bbi_model.compute_importance()
    imp3[0,i]=res_CPI["importance"].reshape((3,))
    pval3[0,i]=res_CPI["pval"].reshape((3,))
    #PFI
    bbi_model2 = BlockBasedImportance(
            estimator=None,
            do_hyper=True,
            importance_estimator="Mod_RF",
            dict_hyper=None,
            conditional=False,
            group_stacking=False,
            n_perm=100,
            n_jobs=10,
            prob_type="regression",
            k_fold=2,
        )
    bbi_model2.fit(data_enc, y)
    res_PFI = bbi_model2.compute_importance()
    imp3[1,i]=res_PFI["importance"].reshape((3,))
    pval3[1,i]=res_PFI["pval"].reshape((3,))

    #LOCO
    ntrees = np.arange(100, 500, 100)
    lr = np.arange(.01, .1, .05)
    param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
    ## set up cv objects
    cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 1), param_grid = param_grid, cv = 5)
    for j in range(3):
        print("covariate: "+str(j))
        vimp = vimpy.vim(y = y, x = data_enc.values, s = j, pred_func = cv_full, measure_type = "r_squared")
        vimp.get_point_est()
        vimp.get_influence_function()
        vimp.get_se()
        vimp.get_ci()
        vimp.hypothesis_test(alpha = 0.05, delta = 0)
        imp3[2,i,j]=vimp.vimp_
        pval3[2,i, j]=vimp.p_value_
#%%
f_res={}
f_res = pd.DataFrame(f_res)
for i in range(3):#CPI, PFI, LOCO
    for j in range(len(noise_arr)):
        f_res1={}
        if i==0:
            f_res1["method"] = ["CPI"]
        elif i==1:
            f_res1["method"]=["PFI"]
        else: 
            f_res1["method"]=["LOCO"]
        f_res1["noise"]=noise_arr[j]
        for k in range(len(list(data.columns))):
            f_res1["imp_V"+str(k)]=imp3[i, j, k]
            f_res1["pval_V"+str(k)]=pval3[i, j, k]
        f_res1=pd.DataFrame(f_res1)
        f_res=pd.concat([f_res, f_res1], ignore_index=True)
f_res.to_csv(
    f"results/results_csv_Angel/simulation_PFIvsCPIvsLOCO_cond_null_hyp.csv",
    index=False,
) 


#%%

# Visualization from the csv file

res_path = pathlib.Path('results/results_csv_Angel')
list(res_path.glob('*.csv'))

df = pd.read_csv(res_path/"simulation_PFIvsCPIvsLOCO_cond_null_hyp.csv")

p=3# Number of covariates
fig, axs = plt.subplots(2,p)
fig.suptitle("Conditionally independent covariate with CPI vs PFI vs LOCO",fontsize=16)
for i in range(p):
    for method, group in df.groupby('method'):
        axs[0, i].plot(group['noise'], group['imp_V'+str(i)], label=method)
        axs[1, i].plot(group['noise'], -np.log10(group['pval_V'+str(i)]+1e-10), label=method)
    axs[0, i].set_title(r'Importance $x$'+str(i), fontsize=14)
    axs[1,i].axhline(y=-np.log10(0.05), color='r', linestyle='--', linewidth=1)
    axs[1, i].set_title(r'-log10(p_value) $x$'+str(i), fontsize=14)
    axs[0,i].legend()
    axs[1,i].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
fig.text(0.53, 0, r'$\lambda$', ha='center', va='center')

fig.savefig("visualization/plots_Angel/CPIvsPFIvsLOCO_cond-indep.pdf", bbox_inches="tight")

# %%

# Plots
fig, axs = plt.subplots(2,p)
fig.suptitle("Conditionally independent covariate with CPI vs PFI",fontsize=16)
for i in range(p):
    axs[0, i].plot(noise_arr, imp3[0,:,i], label="CPI", linestyle="dashed")
    axs[0, i].plot(noise_arr, imp3[1,:,i], label="PFI")
    axs[0, i].plot(noise_arr, imp3[2,:,i], label="LOCO")
    axs[0, i].set_title(r'Importance $x$'+str(i), fontsize=14)
    axs[1, i].plot(noise_arr, -np.log10(pval3[0,:,i]+1e-10), label="CPI", linestyle="dashed")
    axs[1, i].plot(noise_arr, -np.log10(pval3[1,:,i]+1e-10), label="PFI")
    axs[1, i].plot(noise_arr, -np.log10(pval3[2,:,i]+1e-10), label="LOCO")
    axs[1,i].axhline(y=-np.log10(0.05), color='r', linestyle='-')
    axs[1, i].set_title(r'-log10(p_value) $x$'+str(i), fontsize=14)
    axs[0,i].legend()
    axs[1,i].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
fig.text(0.53, 0, r'$\lambda$', ha='center', va='center')

fig.savefig("CPIvsPFIvsLOCO_cond-indep.pdf", bbox_inches="tight")

# %%

# Plots
fig, axs = plt.subplots(2,p)
fig.suptitle("Conditionally independent covariate with CPI vs PFI",fontsize=16)
for i in range(p):
    axs[0, i].plot(noise_arr[0:7], imp3[0,0:7,i], label="CPI", linestyle="dashed")
    axs[0, i].plot(noise_arr[0:7], imp3[1,0:7,i], label="PFI")
    axs[0, i].set_title(r'Importance $x$'+str(i), fontsize=14)
    axs[1, i].plot(noise_arr[0:7], -np.log10(pval3[0,0:7,i]+1e-10), label="CPI", linestyle="dashed")
    axs[1, i].plot(noise_arr[0:7], -np.log10(pval3[1,0:7,i]+1e-10), label="PFI")
    axs[1,i].axhline(y=-np.log10(0.05), color='r', linestyle='-', label=r"$-\mathrm{log}_{10}(\alpha)$")
    axs[1, i].set_title(r'-log10(p_value) $x$'+str(i), fontsize=14)
    axs[0,i].legend()
    axs[1,i].legend()

fig.text(0.53, 0, r'$\lambda$', ha='center', va='center')
#fig.text(0.06, 0.5, 'Common Y Label', ha='center', va='center', rotation='vertical')
plt.tight_layout(rect=[0, 0, 1, 0.95]) 

fig.savefig("CPIvsPFI_cond-indep_2.pdf")


# %%
# Example of how to use LOCO and a higher dimensional setting
#LOCO

## -------------------------------------------------------------
## problem setup
## -------------------------------------------------------------
## define a function for the conditional mean of Y given X
def cond_mean(x = None):
    f1 = np.where(np.logical_and(-2 <= x[:, 0], x[:, 0] < 2), np.floor(x[:, 0]), 0)
    f2 = np.where(x[:, 1] <= 0, 1, 0)
    f3 = np.where(x[:, 2] > 0, 1, 0)
    f6 = np.absolute(x[:, 5]/4) ** 3
    f7 = np.absolute(x[:, 6]/4) ** 5
    f11 = (7./3)*np.cos(x[:, 10]/2)
    ret = f1 + f2 + f3 + f6 + f7 + f11
    return ret

## create data
np.random.seed(seed)
n = 10000
p = 15
x = np.zeros((n, p))
for i in range(0, x.shape[1]) :
    x[:,i] = np.random.normal(0, 2, n)

y = cond_mean(x) + np.random.normal(0, 1, n)
ntrees = np.arange(100, 500, 100)
lr = np.arange(.01, .1, .05)

param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]

## set up cv objects
cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 1), param_grid = param_grid, cv = 5)
cv_small = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 1), param_grid = param_grid, cv = 5)

## -------------------------------------------------------------
## get variable importance estimates
## -------------------------------------------------------------
# set seed
np.random.seed(seed)

# %%
LOCO_imp=np.zeros(15)
LOCO_p_val=np.zeros(15)
for j in range(15):
    print("covariate: "+str(j))
    vimp = vimpy.vim(y = y, x = x, s = j, pred_func = cv_full, measure_type = "r_squared")
    vimp.get_point_est()
    vimp.get_influence_function()
    vimp.get_se()
    vimp.get_ci()
    vimp.hypothesis_test(alpha = 0.05, delta = 0)
    LOCO_imp[j]=vimp.vimp_
    LOCO_p_val[j]=vimp.p_value_


# %%
LOCO_p_val<0.05


# %%
