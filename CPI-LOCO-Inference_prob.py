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
intra_cor=[0,0.05, 0.1, 0.2, 0.3, 0.5, 0.65, 0.85]
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

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}

sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='intra_cor',y='imp_V0',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
plt.plot(np.linspace(0,0.85, 50), beta[0]**2*(1-np.linspace(0,0.85, 50)**2), label=r"$\beta^2_j(1-\rho^2)$",linestyle='--', linewidth=1, color="black")

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

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}

sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='intra_cor',y='imp_V1',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
plt.plot(np.linspace(0,0.85, 50), beta[1]**2*(1-np.linspace(0,0.85, 50)**2), label=r"$\beta^2_j(1-\rho^2)$",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_1$')
plt.xlabel(r'Correlation')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-corr-lineplt1.pdf", bbox_inches="tight")
plt.show()



#%% Second experiment

#SECOND EXPERIMENT: 
#DATA
num_rep=3
snr=4
p=2
cor=0.6
n_samples=[30, 50, 100, 250, 500, 1000, 2000]
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

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}

sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n_samples',y='imp_V0',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
plt.plot(n_samples, [beta[0]**2*(1-cor**2) for i in range(len(n_samples))], label=r"$\beta^2_j(1-\rho^2)$",linestyle='--', linewidth=1, color="black")

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

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}

sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n_samples',y='imp_V1',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
plt.plot(n_samples, [beta[1]**2*(1-cor**2) for i in range(len(n_samples))], label=r"$\beta^2_j(1-\rho^2)$",linestyle='--', linewidth=1, color="black")

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
p=50
cor=0.6
n_samples=[50, 100, 200, 500, 1000, 2000]
imp2=np.zeros((4,num_rep, len(n_samples), p))# 4 because there is 4 methods
pval2=np.zeros((4, len(n_samples), p))
 # Determine beta coefficients
rng = np.random.RandomState(seed)
n_cal=100

interest_coord=[0, 1, 6, 7]
#%%
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
    asymp1=pd.DataFrame([asymp1])
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
        imp2[3,l,i]=res_CPI_Rob["importance"].reshape((p,))*n_cal/(n_cal+1)
        pval2[3,i]+=1/num_rep*res_CPI_Rob["pval"].reshape((p,))


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
        imp2[0,l,i]=1/2*res_CPI["importance"].reshape((p,))
        pval2[0,i]+=1/(2*num_rep)*res_CPI["pval"].reshape((p,))
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
        imp2[1,l,i]=res_PFI["importance"].reshape((p,))
        pval2[1,i]+=1/num_rep*res_PFI["pval"].reshape((p,))
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
            for k in range(p):
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

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}

sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n_samples',y='imp_V0',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
asymp=asymp[asymp["coord"]==0]
plt.plot(n_samples, [asymp["LOCO"] for i in range(len(n_samples))], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

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
palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n_samples',y='imp_V1',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
asymp=asymp[asymp["coord"]==1]
plt.plot(n_samples, [asymp["LOCO"] for i in range(len(n_samples))], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_1$')
plt.xlabel(r'Number of samples')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-HighDim-Bias-diff-n-lineplt1.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_n_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt.csv")

# Display the first few rows of the DataFrame
print(df.head())
palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}


sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n_samples',y='imp_V5',hue='method',palette=palette) #,style='Regressor',markers=markers, dashes=dashes)
asymp=asymp[asymp["coord"]==5]
plt.plot(n_samples, [asymp["LOCO"] for i in range(len(n_samples))], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_5$')
plt.xlabel(r'Number of samples')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-HighDim-Bias-diff-n-lineplt5.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_n_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt.csv")

# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='n_samples',y='imp_V6',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
asymp=asymp[asymp["coord"]==6]
plt.plot(n_samples, [asymp["LOCO"] for i in range(len(n_samples))], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_6$')
plt.xlabel(r'Number of samples')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-HighDim-Bias-diff-n-lineplt6.pdf", bbox_inches="tight")
plt.show()






#%%
#Fourth EXPERIMENT: 
#DATA
num_rep=3
snr=4
p=50
n=300
intra_cor=[0.05, 0.1, 0.3, 0.5, 0.8]
imp2=np.zeros((4,num_rep, len(intra_cor), p))# 4 because there is 4 methods
pval2=np.zeros((4, len(intra_cor), p))
 # Determine beta coefficients
rng = np.random.RandomState(seed)
n_cal=100

interest_coord=[0, 1, 6, 7]
#%%
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
        X,y=GenToysDataset(n=10000, d=p, cor='toep', y_method="imp1", k=2, mu=np.zeros(p), rho_toep=intra_cor[i_cor])
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
        asymp1=pd.DataFrame([asymp1])
        asymp_df=pd.concat([asymp_df, asymp1], ignore_index=True)


asymp_df.to_csv(
    f"results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_cor_cent.csv",
    index=False,
) 


#%%
for l in range(num_rep):
    print("Experiment: "+str(l))
    for (i,cor) in enumerate(intra_cor):
        print("With cor="+str(cor))
        X,y=GenToysDataset(n=n, d=p, cor='toep', y_method="imp1", k=2, mu=np.zeros(p), rho_toep=cor)

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
        imp2[3,l,i]=res_CPI_Rob["importance"].reshape((p,))*n_cal/(n_cal+1)
        pval2[3,i]+=1/num_rep*res_CPI_Rob["pval"].reshape((p,))


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
        imp2[0,l,i]=1/2*res_CPI["importance"].reshape((p,))
        pval2[0,i]+=1/(2*num_rep)*res_CPI["pval"].reshape((p,))
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
        imp2[1,l,i]=res_PFI["importance"].reshape((p,))
        pval2[1,i]+=1/num_rep*res_PFI["pval"].reshape((p,))
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
            for k in range(p):
                f_res1["imp_V"+str(k)]=imp2[i,l, j, k]
                f_res1["pval_V"+str(k)]=pval2[i, j, k]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)
f_res.to_csv(
    f"results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_cor_lineplt_cent.csv",
    index=False,
) 
print(f_res.head())

#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_cor_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_cor.csv")
# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='cor',y='imp_V0',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
asymp=asymp[asymp["coord"]==0]
plt.plot(asymp["intra_cor"], asymp["LOCO"], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

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

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='cor',y='imp_V1',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
asymp=asymp[asymp["coord"]==1]
plt.plot(asymp["intra_cor"], asymp["LOCO"], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

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

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='cor',y='imp_V5',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
asymp=asymp[asymp["coord"]==5]
plt.plot(asymp["intra_cor"], asymp["LOCO"], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

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

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='cor',y='imp_V6',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
asymp=asymp[asymp["coord"]==6]
plt.plot(asymp["intra_cor"], asymp["LOCO"], label=r"Asymptotic",linestyle='--', linewidth=1, color="black")

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

#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_cor_lineplt_cent.csv")
# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='cor',y='imp_V0',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)


plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)

plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_0$')
plt.xlabel(r'Correlation')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-HighDim-diff-cor-lineplt0_cent.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_cor_lineplt_cent.csv")

# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='cor',y='imp_V1',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_1$')
plt.xlabel(r'Correlation')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-cor-lineplt1_cent.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_cor_lineplt_cent.csv")

# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='cor',y='imp_V5',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_5$')
plt.xlabel(r'Correlation')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-cor-lineplt5_cent.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_cor_lineplt_cent.csv")

# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='cor',y='imp_V6',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_6$')
plt.xlabel(r'Correlation')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-cor-lineplt6_cent.pdf", bbox_inches="tight")
plt.show()

#%%
#Fifth EXPERIMENT: 
#DATA
num_rep=3
snr=4
dim=[10, 20, 35, 50, 100]
min_p=10
n=300
cor=0.6
imp2=np.zeros((4,num_rep, len(dim), min_p))# 4 because there is 4 methods
pval2=np.zeros((4, len(dim), min_p))
 # Determine beta coefficients
rng = np.random.RandomState(seed)
n_cal=100

interest_coord=[0, 1, 6, 7]
#%%

#LOCO asymptotically 
ntrees = np.arange(100, 500, 100)
lr = np.arange(.01, .1, .05)
param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
## set up cv objects
cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 1), param_grid = param_grid, cv = 5, n_jobs=-1)
asymp_df={}
asymp_df=pd.DataFrame(asymp_df)
for i_p in range(len(dim)):
    for j in range(len(interest_coord)):
        print("covariate: "+str(interest_coord[j]))
        X,y=GenToysDataset(n=100000, d=dim[i_p], cor='toep', y_method="imp1", k=2, mu=np.zeros(dim[i_p]), rho_toep=cor)
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
        asymp1["d"]=dim[i_p]
        asymp1=pd.DataFrame([asymp1])
        asymp_df=pd.concat([asymp_df, asymp1], ignore_index=True)


asymp_df.to_csv(
    f"results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d.csv",
    index=False,
) 


#%%
for l in range(num_rep):
    print("Experiment: "+str(l))
    for (i,p) in enumerate(dim):
        print("With d="+str(p))
        X,y=GenToysDataset(n=n, d=p, cor='toep', y_method="imp1", k=2, mu=np.zeros(p), rho_toep=cor)

        

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
        intermediate_imp=res_CPI_Rob["importance"].reshape((p,))*n_cal/(n_cal+1)
        imp2[3,l,i]=intermediate_imp[:min_p]
        intermediate_pval=1/num_rep*res_CPI_Rob["pval"].reshape((p,))
        pval2[3,i]+=intermediate_pval[:min_p]


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
        intermed_imp=1/2*res_CPI["importance"].reshape((p,))
        imp2[0,l,i]=intermed_imp[:min_p]
        intermed_pval=1/(2*num_rep)*res_CPI["pval"].reshape((p,))
        pval2[0,i]+=intermed_pval[:min_p]
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
        intermed_imp=res_PFI["importance"].reshape((p,))
        imp2[1,l,i]=intermed_imp[:min_p]
        intermed_pval=1/num_rep*res_PFI["pval"].reshape((p,))
        pval2[1,i]+=intermed_pval[:min_p]
        #LOCO
        ntrees = np.arange(100, 500, 100)
        lr = np.arange(.01, .1, .05)
        param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
        ## set up cv objects
        cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 1), param_grid = param_grid, cv = 5, n_jobs=10)
        for j in range(min_p):
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
        for j in range(len(dim)):
            f_res1={}
            if i==0:
                f_res1["method"] = ["0.5*CPI"]
            elif i==1:
                f_res1["method"]=["PFI"]
            elif i==2: 
                f_res1["method"]=["LOCO"]
            else:
                f_res1["method"]=["Robust-CPI"]
            f_res1["d"]=dim[j]
            for k in range(min_p):
                f_res1["imp_V"+str(k)]=imp2[i,l, j, k]
                f_res1["pval_V"+str(k)]=pval2[i, j, k]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)
f_res.to_csv(
    f"results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_d_lineplt.csv",
    index=False,
) 
print(f_res.head())

#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_d_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d.csv")
asymp_rob=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d_robust.csv")
asymp_cpi=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d_05CPI.csv")
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}

sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='d',y='imp_V0',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
asymp=asymp[asymp["coord"]==0]
plt.plot(asymp["d"], asymp["LOCO"], label=r"AsymptoticLOCO",linestyle='--', linewidth=1, color="green")
asymp_rob=asymp_rob[asymp_rob["coord"]==0]

plt.plot(asymp_rob["d"], asymp_rob["LOCO"], label=r"AsymptoticRob",linestyle='--', linewidth=1, color="purple")
asymp_cpi=asymp_cpi[asymp_cpi["coord"]==0]
plt.plot(asymp_cpi["d"], asymp_cpi["LOCO"], label=r"AsymptoticCPI",linestyle='--', linewidth=1, color="blue")

plt.plot(asymp["d"],[(1-cor**2)/2 for i in range(len(asymp["d"]))], label=r"Theoretical",linestyle='--', linewidth=2, color="gray")

#plt.ylim((1e-2,1e3))
#plt.legend()

plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)

plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_0$')
plt.xlabel(r'Dimension')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-HighDim-diff-d-lineplt0.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_d_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d.csv")
asymp_rob=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d_robust.csv")
asymp_cpi=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d_05CPI.csv")

# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='d',y='imp_V1',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
asymp=asymp[asymp["coord"]==1]
plt.plot(asymp["d"], asymp["LOCO"], label=r"Asymptotic",linestyle='--', linewidth=1, color="green")
asymp_rob=asymp_rob[asymp_rob["coord"]==1]

plt.plot(asymp_rob["d"], asymp_rob["LOCO"], label=r"AsymptoticRob",linestyle='--', linewidth=1, color="purple")
asymp_cpi=asymp_cpi[asymp_cpi["coord"]==1]
plt.plot(asymp_cpi["d"], asymp_cpi["LOCO"], label=r"AsymptoticCPI",linestyle='--', linewidth=1, color="blue")
theo=[]
for d in asymp["d"]:
    mat=toep(d, cor)
    sigma_1=mat[1]
    sigma_1=np.delete(sigma_1, 1)
    inv=np.delete(mat, 1, axis=0)
    inv=np.delete(inv, 1, axis=1)
    inv=np.linalg.inv(inv)
    theo.append((1-np.dot(np.dot(sigma_1,inv), sigma_1.T))*0.5)
plt.plot(asymp["d"],theo, label=r"Theoretical",linestyle='--', linewidth=2, color="gray")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_1$')
plt.xlabel(r'Dimension')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-d-lineplt1.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_d_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d.csv")
asymp_rob=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d_robust.csv")
asymp_cpi=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d_05CPI.csv")

# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='d',y='imp_V5',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
asymp=asymp[asymp["coord"]==5]
plt.plot(asymp["d"], asymp["LOCO"], label=r"Asymptotic",linestyle='--', linewidth=1, color="green")
asymp_rob=asymp_rob[asymp_rob["coord"]==5]

plt.plot(asymp_rob["d"], asymp_rob["LOCO"], label=r"AsymptoticRob",linestyle='--', linewidth=1, color="purple")
asymp_cpi=asymp_cpi[asymp_cpi["coord"]==5]
plt.plot(asymp_cpi["d"], asymp_cpi["LOCO"], label=r"AsymptoticCPI",linestyle='--', linewidth=1, color="blue")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_5$')
plt.xlabel(r'Dimension')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-d-lineplt5.pdf", bbox_inches="tight")
plt.show()


#%%

df = pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-diff_d_lineplt.csv")
asymp=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d.csv")
asymp_rob=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d_robust.csv")
asymp_cpi=pd.read_csv("results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d_05CPI.csv")
# Display the first few rows of the DataFrame
print(df.head())

palette = {'Robust-CPI': 'purple', '0.5*CPI': 'blue', 'LOCO':'green', 'PFI':'orange', "LOCO-AC": "red"}
sns.set(rc={'figure.figsize':(4,4)})
sns.lineplot(data=df,x='d',y='imp_V6',hue='method',palette=palette)#,style='Regressor',markers=markers, dashes=dashes)
asymp=asymp[asymp["coord"]==6]
plt.plot(asymp["d"], asymp["LOCO"], label=r"Asymptotic",linestyle='--', linewidth=1, color="green")
asymp_rob=asymp_rob[asymp_rob["coord"]==6]
plt.plot(asymp_rob["d"], asymp_rob["LOCO"], label=r"AsymptoticRob",linestyle='--', linewidth=1, color="purple")
asymp_cpi=asymp_cpi[asymp_cpi["coord"]==6]
plt.plot(asymp_cpi["d"], asymp_cpi["LOCO"], label=r"AsymptoticCPI",linestyle='--', linewidth=1, color="blue")

#plt.ylim((1e-2,1e3))
#plt.legend()

#plt.legend(bbox_to_anchor=(-1.20, 0.5), loc='center left', borderaxespad=0.)
plt.legend().remove()
plt.subplots_adjust(right=0.75)

#plt.xscale('log')
#plt.yscale('log')


plt.ylabel(r'Importance of $X_6$')
plt.xlabel(r'Dimension')
plt.savefig("visualization/plots_Angel/simulation_CPI-LOCO-Bias-diff-d-lineplt6.pdf", bbox_inches="tight")
plt.show()


















# %%
n=10000
dim=[10, 20, 35, 50, 100]
interest_coord=[0, 1, 6, 7]
asymp_df={}
asymp_df=pd.DataFrame(asymp_df)
for i_p in range(len(dim)):
    X,y=GenToysDataset(n=n, d=dim[i_p], cor='toep', y_method="imp1", k=2, mu=np.zeros(dim[i_p]), rho_toep=cor)
    gb_param_grid = {
    'n_estimators': [100, 300],  
    'learning_rate': [0.01, 0.1], 
    'max_depth': [3, 7], 
    'min_samples_split': [2, 10],  
    'min_samples_leaf': [1, 4],
    'subsample': [0.8, 1.0], 
    'loss': ['squared_error', 'huber']  
}
    bbi_model3 = BlockBasedImportance(
                estimator=GradientBoostingRegressor(),
                do_hyper=True,
                importance_estimator=None,
                dict_hyper=gb_param_grid,
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
    intermediate_imp=res_CPI_Rob["importance"].reshape((dim[i_p],))*n_cal/(n_cal+1)
    intermediate_pval=1/num_rep*res_CPI_Rob["pval"].reshape((dim[i_p],))
    for i in interest_coord:
        asymp1={}
        asymp1["LOCO"]=intermediate_imp[i]
        asymp1["p_value"]=intermediate_pval[i]
        asymp1["coord"]=i
        asymp1["d"]=dim[i_p]
        asymp1=pd.DataFrame([asymp1])
        asymp_df=pd.concat([asymp_df, asymp1], ignore_index=True)


asymp_df.to_csv(
    f"results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d_robust.csv",
    index=False,
) 

#%%

n=10000
dim=[10, 20, 35, 50, 100]
interest_coord=[0, 1, 6, 7]
asymp_df={}
asymp_df=pd.DataFrame(asymp_df)
for i_p in range(len(dim)):
    X,y=GenToysDataset(n=n, d=dim[i_p], cor='toep', y_method="imp1", k=2, mu=np.zeros(dim[i_p]), rho_toep=cor)
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
    intermediate_imp=res_CPI["importance"].reshape((dim[i_p],))*0.5
    intermediate_pval=1/num_rep*res_CPI["pval"].reshape((dim[i_p],))
    for i in interest_coord:
        asymp1={}
        asymp1["LOCO"]=intermediate_imp[i]
        asymp1["p_value"]=intermediate_pval[i]
        asymp1["coord"]=i
        asymp1["d"]=dim[i_p]
        asymp1=pd.DataFrame([asymp1])
        asymp_df=pd.concat([asymp_df, asymp1], ignore_index=True)


asymp_df.to_csv(
    f"results/results_csv_Angel/simulation_CPI-LOCO-highDim-asympt_d_05CPI.csv",
    index=False,
) 

# %%
