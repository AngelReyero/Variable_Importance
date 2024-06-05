#%%
import argparse
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BBI_package.src.BBI import BlockBasedImportance
from joblib import Parallel, delayed
from scipy.linalg import cholesky
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
#from .utils import compute_loco
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

#%% 
data_enc.cov()


# %%
inter=[100,200, 350, 500, 750, 1000, 2000, 3500, 5000, 7500,9000, 11000, 13000, 15000]
imp=np.zeros((2, len(inter), 3))
pval=np.zeros((2, len(inter), 3))
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
imp2=np.zeros((2, len(intra_cor), 3))
pval2=np.zeros((2, len(intra_cor), 3))
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

## y=B1 X1+ B2 X2 + B3 (X1*X2+noise*eps1)+eps2

snr=4
p=3
n=1000
x = norm.rvs(size=(p, n), random_state=seed)
noise_arr=[0, 0.05, 0.1,0.2, 0.35, 0.5, 0.75, 1, 1.5, 2,3.5, 5]

imp3=np.zeros((2, len(noise_arr), 3))
pval3=np.zeros((2, len(noise_arr), 3))
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




# %%

# Plots
fig, axs = plt.subplots(2,p)
fig.suptitle("Conditionally independent covariate with CPI vs PFI",fontsize=16)
for i in range(p):
    axs[0, i].plot(noise_arr, imp3[0,:,i], label="CPI", linestyle="dashed")
    axs[0, i].plot(noise_arr, imp3[1,:,i], label="PFI")
    axs[0, i].set_title(r'Importance $x$'+str(i), fontsize=14)
    axs[1, i].plot(noise_arr, -np.log10(pval3[0,:,i]+1e-10), label="CPI", linestyle="dashed")
    axs[1, i].plot(noise_arr, -np.log10(pval3[1,:,i]+1e-10), label="PFI")
    axs[1,i].axhline(y=-np.log10(0.05), color='r', linestyle='-')
    axs[1, i].set_title(r'-log10(p_value) $x$'+str(i), fontsize=14)
    axs[0,i].legend()
    axs[1,i].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
fig.savefig("CPIvsPFI_cond-indep.pdf")
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

#LOCO

