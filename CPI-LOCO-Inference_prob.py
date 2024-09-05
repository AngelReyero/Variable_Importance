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
import seaborn as sns
seed=2024

#%%

#FIRST EXPERIMENT: 
#DATA
num_rep=5
snr=4
p=2
n=100
x = norm.rvs(size=(p, n), random_state=seed)
intra_cor=[0,0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.65, 0.8, 0.9]
imp2=np.zeros((5,num_rep, len(intra_cor), 2))# 5 because there is 5 methods
pval2=np.zeros((5, len(intra_cor), 2))
 # Determine beta coefficients
rng = np.random.RandomState(seed)
n_signal=2
beta=np.array([2,1])

#%%
for l in range(num_rep):
    print("Experiment: "+str(l))
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
        

        #LOCO robust
        n_cal=100
        bbi_model3 = BlockBasedImportance(
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
                robust=True,
                n_cal=n_cal,
            )
        bbi_model3.fit(data_enc, y)
        res_CPI_Rob = bbi_model3.compute_importance()
        imp2[4,l,i]=res_CPI_Rob["importance"].reshape((2,))*n_cal/(n_cal+1)
        pval2[4,i]+=1/num_rep*res_CPI_Rob["pval"].reshape((2,))


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
        imp2[0,l,i]=1/2*res_CPI["importance"].reshape((2,))
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
        imp2[1,l,i]=res_PFI["importance"].reshape((2,))
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
            imp2[2,l,i,j]+=vimp.vimp_*np.var(y)
            pval2[2,i, j]+=1/num_rep*vimp.p_value_
        #LOCO Ahmad
        res_LOCO=compute_loco(data_enc, y, dnn=True)#TO CHANGE (dnn=True for the correct LOCO)
        imp2[3, l,i]=np.array(res_LOCO["val_imp"], dtype=float)
        pval2[3, i]+=1/num_rep*np.array(res_LOCO["p_value"], dtype=float)

        


#%% Lineplot
#Save the results
f_res={}
f_res = pd.DataFrame(f_res)
for l in range(num_rep):
    for i in range(5):#CPI, PFI, LOCO_W, LOCO_AC, Robust-Loco
        for j in range(len(intra_cor)):
            f_res1={}
            if i==0:
                f_res1["method"] = ["0.5*CPI"]
            elif i==1:
                f_res1["method"]=["PFI"]
            elif i==2: 
                f_res1["method"]=["LOCO"]
            elif i==3:
                f_res1["method"]=["LOCO-AC"]
            else:
                f_res1["method"]=["Robust-CPI"]
            f_res1["intra_cor"]=intra_cor[j]
            for k in range(len(list(data.columns))):
                f_res1["imp_V"+str(k)]=imp2[i,l, j, k]
                f_res1["pval_V"+str(k)]=pval2[i, j, k]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)
f_res.to_csv(
    f"results/results_csv_Angel/simulation_CPI-LOCO-Bias-diff_corr_lineplt.csv",
    index=False,
) 
print(f_res.head())

#%%


df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-Bias-diff_corr_lineplt.csv")

# Display the first few rows of the DataFrame
print(df.head())


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='intra_cor',y='imp_V0',hue='method')#,palette=palette,style='Regressor',markers=markers, dashes=dashes)
plt.plot(np.linspace(0,0.9, 50), beta[0]**2*(1-np.linspace(0,0.9, 50)**2), label=r"$\beta^2_j(1-\rho^2)$",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)

plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_0$')
plt.xlabel(r'Correlation')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-corr-lineplt0.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-Bias-diff_corr_lineplt.csv")

# Display the first few rows of the DataFrame
print(df.head())


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='intra_cor',y='imp_V1',hue='method')#,palette=palette,style='Regressor',markers=markers, dashes=dashes)
plt.plot(np.linspace(0,0.9, 50), beta[1]**2*(1-np.linspace(0,0.9, 50)**2), label=r"$\beta^2_j(1-\rho^2)$",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_1$')
plt.xlabel(r'Correlation')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-corr-lineplt0.pdf", bbox_inches="tight")
plt.show()



#%% Second experiment

#SECOND EXPERIMENT: 
#DATA
num_rep=3
snr=4
p=2
cor=0.6
n_samples=[30, 50, 100, 200, 300, 700]
imp2=np.zeros((5,num_rep, len(n_samples), 2))# 5 because there is 5 methods
pval2=np.zeros((5, len(n_samples), 2))
 # Determine beta coefficients
rng = np.random.RandomState(seed)
n_signal=2
beta=np.array([2,1])

#%%
for l in range(num_rep):
    print("Experiment: "+str(l))
    for (i,n) in enumerate(n_samples):
        print("With n="+str(n))
        cor_mat=np.zeros((p,p))
        cor_mat[0:p,0:p]=cor
        np.fill_diagonal(cor_mat, 1)
        x = norm.rvs(size=(p, n), random_state=seed)
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
        

        #LOCO robust
        n_cal=100
        bbi_model3 = BlockBasedImportance(
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
                robust=True,
                n_cal=n_cal,
            )
        bbi_model3.fit(data_enc, y)
        res_CPI_Rob = bbi_model3.compute_importance()
        imp2[4,l,i]=res_CPI_Rob["importance"].reshape((2,))*n_cal/(n_cal+1)
        pval2[4,i]+=1/num_rep*res_CPI_Rob["pval"].reshape((2,))


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
        imp2[0,l,i]=1/2*res_CPI["importance"].reshape((2,))
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
        imp2[1,l,i]=res_PFI["importance"].reshape((2,))
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
            imp2[2,l,i,j]+=vimp.vimp_*np.var(y)
            pval2[2,i, j]+=1/num_rep*vimp.p_value_
        #LOCO Ahmad
        res_LOCO=compute_loco(data_enc, y, dnn=True)#TO CHANGE (dnn=True for the correct LOCO)
        imp2[3, l,i]=np.array(res_LOCO["val_imp"], dtype=float)
        pval2[3, i]+=1/num_rep*np.array(res_LOCO["p_value"], dtype=float)

        


#%% Lineplot
#Save the results
f_res={}
f_res = pd.DataFrame(f_res)
for l in range(num_rep):
    for i in range(5):#CPI, PFI, LOCO_W, LOCO_AC, Robust-Loco
        for j in range(len(n_samples)):
            f_res1={}
            if i==0:
                f_res1["method"] = ["0.5*CPI"]
            elif i==1:
                f_res1["method"]=["PFI"]
            elif i==2: 
                f_res1["method"]=["LOCO"]
            elif i==3:
                f_res1["method"]=["LOCO-AC"]
            else:
                f_res1["method"]=["Robust-CPI"]
            f_res1["n_samples"]=n_samples[j]
            for k in range(len(list(data.columns))):
                f_res1["imp_V"+str(k)]=imp2[i,l, j, k]
                f_res1["pval_V"+str(k)]=pval2[i, j, k]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)
f_res.to_csv(
    f"results/results_csv_Angel/simulation_CPI-LOCO-Bias-diff_n_lineplt.csv",
    index=False,
) 
print(f_res.head())

#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-Bias-diff_n_lineplt.csv")

# Display the first few rows of the DataFrame
print(df.head())


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n_samples',y='imp_V0',hue='method')#,palette=palette,style='Regressor',markers=markers, dashes=dashes)
plt.plot(n_samples, beta[0]**2*(1-cor**2), label=r"$\beta^2_j(1-\rho^2)$",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)

plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_0$')
plt.xlabel(r'Number of samples')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-n-lineplt0.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-Bias-diff_n_lineplt.csv")

# Display the first few rows of the DataFrame
print(df.head())


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n_samples',y='imp_V1',hue='method')#,palette=palette,style='Regressor',markers=markers, dashes=dashes)
plt.plot(n_samples, beta[0]**2*(1-cor**2), label=r"$\beta^2_j(1-\rho^2)$",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_1$')
plt.xlabel(r'Number of samples')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-n-lineplt0.pdf", bbox_inches="tight")
plt.show()









#%% Higher dimension experiment
# covariance matrice 
def ind(i,j,k):
    # separates &,n into k blocks
    return int(i//k==j//k)

# One Toeplitz matrix  
def toep (d, rho=0.6):
  return np.array([[ (rho)**abs(i-j) for i in range(d)]for j in range(d)])

def GenToysDataset(n=1000, d=10, cor='toep', y_method="imp1", k=2, mu=None, rho_toep=0.6):
    X = np.zeros((n,d))
    y = np.zeros(n)
    if mu is None:
        mu=np.ones(d)
    if cor =='iso': 
        # Generate a simple MCAR distribution, with isotrope observation 
        X= np.random.normal(size=(n,d))
    elif cor =='cor': 
        # Generate a simple MCAR distribution, with anisotropic observations and Sigma=U
        U= np.array([[ind(i,j,k) for j in range(d)] for i in range(d)])/np.sqrt(k)
        X= np.random.normal(size=(n,d))@U+mu
    elif cor =='toep': 
        # Generate un simpler MCAR distribution, with anisotropic observations and Sigma=Toepliz
        X= np.random.multivariate_normal(mu,toep(d, rho_toep),size=n)
    else :
        print("WARNING: key word")
    
    if y_method == "imp1":
        y=X[:,0]*X[:,1]*(X[:,2]>0)+2*X[:,3]*X[:,4]*(0>X[:,2])
    elif y_method == "lin":
        y=X[:,0]-X[:,1]+2*X[:, 2]+ X[:,3]-3*X[:,4]
    else :
        print("WARNING: key word")
    return X, y
#%%

#Third EXPERIMENT: 
#DATA
num_rep=3
snr=4
p=100
cor=0.6
n_samples=[30, 50, 100, 200, 300, 700]
imp2=np.zeros((4,num_rep, len(n_samples), 2))# 4 because there is 4 methods
pval2=np.zeros((4, len(n_samples), 2))
 # Determine beta coefficients
rng = np.random.RandomState(seed)
n_cal=100

interest_coord=[0, 1, 6, 7]
X,y=GenToysDataset(n=100000, d=p, cor='toep', y_method="imp1", k=2, mu=None, rho_toep=cor)
#LOCO asymptotically 
ntrees = np.arange(100, 500, 100)
lr = np.arange(.01, .1, .05)
param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
## set up cv objects
cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 1), param_grid = param_grid, cv = 5, n_jobs=-1)
asymp_df={}
asymp_df=pd.DataFrame(asymp_df)
for j in range(len(interest_coord)):
    print("covariate: "+str(interest_coord[j]))
    asymp1={}
    vimp = vimpy.vim(y = y, x = X, s = j, pred_func = cv_full, measure_type = "r_squared")
    vimp.get_point_est()
    vimp.get_influence_function()
    vimp.get_se()
    vimp.get_ci()
    vimp.hypothesis_test(alpha = 0.05, delta = 0)
    asymp1["LOCO"]=vimp.vimp_*np.var(y)
    asymp1["p_value"]=vimp.p_value_
    asymp1["coord"]=interest_coord[j]
    asymp1=pd.DataFrame(asymp1)
    asymp_df=pd.concat([asymp_df, asymp1], ignore_index=True)


asymp_df.to_csv(
    f"results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt.csv",
    index=False,
) 


#%%
for l in range(num_rep):
    print("Experiment: "+str(l))
    for (i,n) in enumerate(n_samples):
        print("With n="+str(n))
        X,y=GenToysDataset(n=n, d=p, cor='toep', y_method="imp1", k=2, mu=None, rho_toep=cor)

        

        #LOCO robust
        
        bbi_model3 = BlockBasedImportance(
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
                robust=True,
                n_cal=n_cal,
            )
        bbi_model3.fit(X, y)
        res_CPI_Rob = bbi_model3.compute_importance()
        imp2[3,l,i]=res_CPI_Rob["importance"].reshape((2,))*n_cal/(n_cal+1)
        pval2[3,i]+=1/num_rep*res_CPI_Rob["pval"].reshape((2,))


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
        bbi_model.fit(X, y)
        res_CPI = bbi_model.compute_importance()
        imp2[0,l,i]=1/2*res_CPI["importance"].reshape((2,))
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
        bbi_model2.fit(X, y)
        res_PFI = bbi_model2.compute_importance()
        imp2[1,l,i]=res_PFI["importance"].reshape((2,))
        pval2[1,i]+=1/num_rep*res_PFI["pval"].reshape((2,))
        #LOCO
        ntrees = np.arange(100, 500, 100)
        lr = np.arange(.01, .1, .05)
        param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
        ## set up cv objects
        cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 1), param_grid = param_grid, cv = 5, n_jobs=10)
        for j in range(p):
            print("covariate: "+str(j))
            vimp = vimpy.vim(y = y, x = X, s = j, pred_func = cv_full, measure_type = "r_squared")
            vimp.get_point_est()
            vimp.get_influence_function()
            vimp.get_se()
            vimp.get_ci()
            vimp.hypothesis_test(alpha = 0.05, delta = 0)
            imp2[2,l,i,j]+=vimp.vimp_*np.var(y)
            pval2[2,i, j]+=1/num_rep*vimp.p_value_

        


#%% Lineplot
#Save the results
f_res={}
f_res = pd.DataFrame(f_res)
for l in range(num_rep):
    for i in range(4):#CPI, PFI, LOCO_W, Robust-Loco
        for j in range(len(n_samples)):
            f_res1={}
            if i==0:
                f_res1["method"] = ["0.5*CPI"]
            elif i==1:
                f_res1["method"]=["PFI"]
            elif i==2: 
                f_res1["method"]=["LOCO"]
            else:
                f_res1["method"]=["Robust-CPI"]
            f_res1["n_samples"]=n_samples[j]
            for k in range(len(list(data.columns))):
                f_res1["imp_V"+str(k)]=imp2[i,l, j, k]
                f_res1["pval_V"+str(k)]=pval2[i, j, k]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)
f_res.to_csv(
    f"results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_n_lineplt.csv",
    index=False,
) 
print(f_res.head())

#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_n_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt.csv")
# Display the first few rows of the DataFrame
print(df.head())


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n_samples',y='imp_V0',hue='method')#,palette=palette,style='Regressor',markers=markers, dashes=dashes)
plt.plot(n_samples, asymp[0,0], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)

plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_0$')
plt.xlabel(r'Number of samples')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-HighDim-diff-n-lineplt0.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_n_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt.csv")

# Display the first few rows of the DataFrame
print(df.head())


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n_samples',y='imp_V1',hue='method')#,palette=palette,style='Regressor',markers=markers, dashes=dashes)
plt.plot(n_samples, asymp[1,0], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_1$')
plt.xlabel(r'Number of samples')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-n-lineplt1.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_n_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt.csv")

# Display the first few rows of the DataFrame
print(df.head())


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n_samples',y='imp_V5',hue='method')#,palette=palette,style='Regressor',markers=markers, dashes=dashes)
plt.plot(n_samples, asymp[2,0], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_5$')
plt.xlabel(r'Number of samples')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-n-lineplt5.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_n_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt.csv")

# Display the first few rows of the DataFrame
print(df.head())


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n_samples',y='imp_V6',hue='method')#,palette=palette,style='Regressor',markers=markers, dashes=dashes)
plt.plot(n_samples, asymp[3,0], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_6$')
plt.xlabel(r'Number of samples')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-n-lineplt6.pdf", bbox_inches="tight")
plt.show()







#Fourth EXPERIMENT: 
#DATA
num_rep=3
snr=4
p=50
n=300
intra_cor=[0.05, 0.1, 0.3, 0.5, 0.8]
imp2=np.zeros((4,num_rep, len(intra_cor), 2))# 4 because there is 4 methods
pval2=np.zeros((4, len(intra_cor), 2))
 # Determine beta coefficients
rng = np.random.RandomState(seed)
n_cal=100

interest_coord=[0, 1, 6, 7]

#LOCO asymptotically 
ntrees = np.arange(100, 500, 100)
lr = np.arange(.01, .1, .05)
param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
## set up cv objects
cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 1), param_grid = param_grid, cv = 5, n_jobs=-1)
asymp_df={}
asymp_df=pd.DataFrame(asymp_df)
for i_cor in range(len(intra_cor)):
    for j in range(len(interest_coord)):
        print("covariate: "+str(interest_coord[j]))
        X,y=GenToysDataset(n=100000, d=p, cor='toep', y_method="imp1", k=2, mu=None, rho_toep=intra_cor[i_cor])
        asymp1={}
        vimp = vimpy.vim(y = y, x = X, s = j, pred_func = cv_full, measure_type = "r_squared")
        vimp.get_point_est()
        vimp.get_influence_function()
        vimp.get_se()
        vimp.get_ci()
        vimp.hypothesis_test(alpha = 0.05, delta = 0)
        asymp1["LOCO"]=vimp.vimp_*np.var(y)
        asymp1["p_value"]=vimp.p_value_
        asymp1["coord"]=interest_coord[j]
        asymp1["intra_cor"]=intra_cor[i_cor]
        asymp1=pd.DataFrame(asymp1)
        asymp_df=pd.concat([asymp_df, asymp1], ignore_index=True)


asymp_df.to_csv(
    f"results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_cor.csv",
    index=False,
) 


#%%
for l in range(num_rep):
    print("Experiment: "+str(l))
    for (i,cor) in enumerate(intra_cor):
        print("With n="+str(n))
        X,y=GenToysDataset(n=n, d=p, cor='toep', y_method="imp1", k=2, mu=None, rho_toep=cor)

        

        #LOCO robust
        
        bbi_model3 = BlockBasedImportance(
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
                robust=True,
                n_cal=n_cal,
            )
        bbi_model3.fit(X, y)
        res_CPI_Rob = bbi_model3.compute_importance()
        imp2[3,l,i]=res_CPI_Rob["importance"].reshape((2,))*n_cal/(n_cal+1)
        pval2[3,i]+=1/num_rep*res_CPI_Rob["pval"].reshape((2,))


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
        bbi_model.fit(X, y)
        res_CPI = bbi_model.compute_importance()
        imp2[0,l,i]=1/2*res_CPI["importance"].reshape((2,))
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
        bbi_model2.fit(X, y)
        res_PFI = bbi_model2.compute_importance()
        imp2[1,l,i]=res_PFI["importance"].reshape((2,))
        pval2[1,i]+=1/num_rep*res_PFI["pval"].reshape((2,))
        #LOCO
        ntrees = np.arange(100, 500, 100)
        lr = np.arange(.01, .1, .05)
        param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
        ## set up cv objects
        cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 1), param_grid = param_grid, cv = 5, n_jobs=10)
        for j in range(p):
            print("covariate: "+str(j))
            vimp = vimpy.vim(y = y, x = X, s = j, pred_func = cv_full, measure_type = "r_squared")
            vimp.get_point_est()
            vimp.get_influence_function()
            vimp.get_se()
            vimp.get_ci()
            vimp.hypothesis_test(alpha = 0.05, delta = 0)
            imp2[2,l,i,j]+=vimp.vimp_*np.var(y)
            pval2[2,i, j]+=1/num_rep*vimp.p_value_

        


#%% Lineplot
#Save the results
f_res={}
f_res = pd.DataFrame(f_res)
for l in range(num_rep):
    for i in range(4):#CPI, PFI, LOCO_W, Robust-Loco
        for j in range(len(intra_cor)):
            f_res1={}
            if i==0:
                f_res1["method"] = ["0.5*CPI"]
            elif i==1:
                f_res1["method"]=["PFI"]
            elif i==2: 
                f_res1["method"]=["LOCO"]
            else:
                f_res1["method"]=["Robust-CPI"]
            f_res1["cor"]=intra_cor[j]
            for k in range(len(list(data.columns))):
                f_res1["imp_V"+str(k)]=imp2[i,l, j, k]
                f_res1["pval_V"+str(k)]=pval2[i, j, k]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)
f_res.to_csv(
    f"results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_cor_lineplt.csv",
    index=False,
) 
print(f_res.head())

#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_cor_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_cor.csv")
# Display the first few rows of the DataFrame
print(df.head())


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='cor',y='imp_V0',hue='method')#,palette=palette,style='Regressor',markers=markers, dashes=dashes)
plt.plot(intra_cor, asymp[:,0,0], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)

plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_0$')
plt.xlabel(r'Correlation')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-HighDim-diff-cor-lineplt0.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_cor_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_cor.csv")

# Display the first few rows of the DataFrame
print(df.head())


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='cor',y='imp_V1',hue='method')#,palette=palette,style='Regressor',markers=markers, dashes=dashes)
plt.plot(intra_cor, asymp[:,1,0], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_1$')
plt.xlabel(r'Correlation')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-cor-lineplt1.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_cor_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_cor.csv")

# Display the first few rows of the DataFrame
print(df.head())


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='cor',y='imp_V6',hue='method')#,palette=palette,style='Regressor',markers=markers, dashes=dashes)
plt.plot(intra_cor, asymp[:,2,0], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_5$')
plt.xlabel(r'Correlation')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-cor-lineplt5.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_cor_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_cor.csv")

# Display the first few rows of the DataFrame
print(df.head())


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='cor',y='imp_V6',hue='method')#,palette=palette,style='Regressor',markers=markers, dashes=dashes)
plt.plot(intra_cor, asymp[:,3,0], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_6$')
plt.xlabel(r'Correlation')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-cor-lineplt6.pdf", bbox_inches="tight")
plt.show()









