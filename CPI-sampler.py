


#%%

def hypertune_predictor(estimator, X, y, param_grid):
    # param_grid = {"max_depth": [2, 5, 10]}
    grid_search = GridSearchCV(estimator, param_grid=param_grid, cv=2)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

#%%


class CPI_sampler():

    def __init__(
        self,
        estimator=None,
        do_hyper=True,
        dict_hyper=None,
        random_state=2024,
    ):
        self.estimator = estimator
        self.importance_estimator = importance_estimator
        self.do_hyper = do_hyper
        self.dict_hyper = dict_hyper
        self.random_state = random_state
        if dict_hyper==None and estimator == None:
            self.dict_hyper={
                    "lr": [1e-4, 1e-3, 1e-2],
                    "l1_weight": [0, 1e-4, 1e-2],
                    "l2_weight": [0, 1e-4, 1e-2],
                }
    
    def fit(self, X, y):
        self.estimator.fit(X, y)




    def __tuning_hyper(self, X, y, ind_fold=None):
        """ """
        (
            X_train_scaled,
            y_train_scaled,
            X_valid_scaled,
            y_valid_scaled,
            X_scaled,
            __,
            scaler_x,
            scaler_y,
            ___,
        ) = create_X_y(
            X,
            y,
            bootstrap=self.bootstrap,
            split_perc=self.split_perc,
            prob_type=self.prob_type,
            list_cont=self.list_cont,
            random_state=self.random_state,
        )
        list_hyper = list(
            itertools.product(*list(self.dict_hyper.values()))
        )
        list_loss = []
        if self.type == "DNN":
            list_loss = self.estimator.hyper_tuning(
                X_train_scaled,
                y_train_scaled,
                X_valid_scaled,
                y_valid_scaled,
                list_hyper,
                random_state=self.random_state,
            )
        else:
            # list_alphas = np.logspace(-3, 5, num=100)
            # cv = KFold(
            #     n_splits=10, random_state=self.random_state, shuffle=True
            # )
            # self.ridge_stack = [
            #     make_pipeline(StandardScaler(), RidgeCV(list_alphas))
            #     for _ in range(len(self.list_grps))
            # ]

            # self.df_pred = [
            #     pd.DataFrame(
            #         columns=["fold", "y_pred", "age"],
            #         index=np.arange(X.shape[0]),
            #         dtype=float,
            #     )
            #     for _ in range(len(self.list_grps))
            # ]
            # for grp_ind, grp in enumerate(self.list_grps):
            #     for fold_idx, (train, test) in enumerate(cv.split(X)):
            #         X_train, X_test = X[train][:, grp], X[test][:, grp]
            #         y_train, y_test = y[train], y[test]
            #         self.ridge_stack[grp_ind].fit(X_train[:, grp], y_train)
            #         y_pred = (
            #             self.ridge_stack[grp_ind].predict(X_test).ravel()
            #         )
            #         self.df_pred[grp_ind].loc[test, "y_pred"] = y_pred
            #         self.df_pred[grp_ind].loc[test, "fold"] = fold_idx
            #         self.df_pred[grp_ind].loc[test, "age"] = y_test

            # self.list_cont = list(np.arange(len(self.list_grps)))

            for ind_el, el in enumerate(list_hyper):
                curr_params = dict(
                    (k, v)
                    for v, k in zip(el, list(self.dict_hyper.keys()))
                )
                list_hyper[ind_el] = curr_params
                self.estimator.set_params(**curr_params)
                if self.prob_type == "regression":
                    y_train_curr = (
                        y_train_scaled * scaler_y.scale_ + scaler_y.mean_
                    )
                    y_valid_curr = (
                        y_valid_scaled * scaler_y.scale_ + scaler_y.mean_
                    )
                    func = lambda x: self.estimator.predict(x)
                else:
                    y_train_curr = y_train_scaled.copy()
                    y_valid_curr = y_valid_scaled.copy()
                    func = lambda x: self.estimator.predict_proba(x)
                self.estimator.fit(X_train_scaled, y_train_curr)

                list_loss.append(
                    self.loss(y_valid_curr, func(X_valid_scaled))
                )

        ind_min = np.argmin(list_loss)
        best_hyper = list_hyper[ind_min]
        if not isinstance(best_hyper, dict):
            best_hyper = dict(zip(self.dict_hyper.keys(), best_hyper))

        self.estimator.set_params(**best_hyper)
        self.estimator.fit(X_scaled, y)

        # If not a DNN learner case, need to save the scalers
        self.scaler_x[ind_fold] = scaler_x
        self.scaler_y[ind_fold] = scaler_y

