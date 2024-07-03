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
from utils.utils_py import compute_loco
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
seed=2024

#%%
num_rep=1
snr=4
p=2
n=10000
x = norm.rvs(size=(p, n), random_state=seed)
intra_cor=[0,0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.65, 0.8, 0.9]
imp2=np.zeros((4, len(intra_cor), 2))
pval2=np.zeros((4, len(intra_cor), 2))
 # Determine beta coefficients
rng = np.random.RandomState(seed)
n_signal=2
#effectset = [-0.5, -1, -2, -3, 0.5, 1, 2, 3]
#beta = rng.choice(effectset, size=(n_signal), replace=True)
beta=np.array([2,1])
for i in range(num_rep):
    print("Experience: "+str(i))
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
                importance_estimator=None,
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
        imp2[0,i]+=1/(2*num_rep)*res_CPI["importance"].reshape((2,))
        pval2[0,i]+=1/(2*num_rep)*res_CPI["pval"].reshape((2,))
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
        imp2[1,i]+=1/num_rep*res_PFI["importance"].reshape((2,))
        pval2[1,i]+=1/num_rep*res_PFI["pval"].reshape((2,))
        #LOCO
        ntrees = np.arange(100, 500, 100)
        lr = np.arange(.01, .1, .05)
        param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
        ## set up cv objects
        cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 1), param_grid = param_grid, cv = 5, n_jobs=10)
        for j in range(2):
            print("covariate: "+str(j))
            vimp = vimpy.vim(y = y, x = data_enc.values, s = j, pred_func = cv_full, measure_type = "r_squared")
            vimp.get_point_est()
            vimp.get_influence_function()
            vimp.get_se()
            vimp.get_ci()
            vimp.hypothesis_test(alpha = 0.05, delta = 0)
            imp2[2,i,j]+=1/num_rep*vimp.vimp_*np.var(y)
            pval2[2,i, j]+=1/num_rep*vimp.p_value_
        #LOCO Ahmad
        res_LOCO=compute_loco(data_enc, y)
        imp2[3, i]+=1/num_rep*np.array(res_LOCO["val_imp"], dtype=float)
        pval2[3, i]+=1/num_rep*np.array(res_LOCO["p_value"], dtype=float)


#%%
#Save the results
f_res={}
f_res = pd.DataFrame(f_res)
for i in range(4):#CPI, PFI, LOCO_W, LOCO_AC
    for j in range(len(intra_cor)):
        f_res1={}
        if i==0:
            f_res1["method"] = ["0.5*CPI"]
        elif i==1:
            f_res1["method"]=["PFI"]
        elif i==2: 
            f_res1["method"]=["LOCO"]
        else:
            f_res1["method"]=["LOCO-AC"]
        f_res1["intra_cor"]=intra_cor[j]
        for k in range(len(list(data.columns))):
            f_res1["imp_V"+str(k)]=imp2[i, j, k]
            f_res1["pval_V"+str(k)]=pval2[i, j, k]
        f_res1=pd.DataFrame(f_res1)
        f_res=pd.concat([f_res, f_res1], ignore_index=True)
f_res.to_csv(
    f"results/results_csv_Angel/simulation_CPI-LOCO-Bias-diff_corr.csv",
    index=False,
) 


#%%

# Visualization from the csv file
beta=np.array([2,1])
res_path = pathlib.Path('results/results_csv_Angel')
list(res_path.glob('*.csv'))

df = pd.read_csv(res_path/"simulation_CPI-LOCO-Bias-diff_corr.csv")

p=2# Number of covariates
fig, axs = plt.subplots(2,p)
fig.suptitle("Different correlations with CPI vs PFI vs LOCO",fontsize=16)
for i in range(p):
    for method, group in df.groupby('method'):
        axs[0, i].plot(group['intra_cor'], group['imp_V'+str(i)], label=method)
        axs[1, i].plot(group['intra_cor'], -np.log10(group['pval_V'+str(i)]+1e-10), label=method)
    axs[0, i].plot(np.linspace(0,0.9, 50), beta[i]**2*(1-np.linspace(0,0.9, 50)**2), label=r"$\beta^2_j(1-\rho^2)$",linestyle='--', linewidth=1)
    axs[0, i].set_title(r'Importance $x$'+str(i), fontsize=14)
    axs[1,i].axhline(y=-np.log10(0.05), color='r', linestyle='--', linewidth=1)
    axs[1, i].set_title(r'-log10(p_value) $x$'+str(i), fontsize=14)
    axs[0,i].legend()
    axs[1,i].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
fig.text(0.53, 0, r'$\rho$', ha='center', va='center')

fig.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-corr.pdf", bbox_inches="tight")


# %%
