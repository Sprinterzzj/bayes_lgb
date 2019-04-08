"""
A library for lightGBM parameter tuning using Bayesian Optimization
"""
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

import lightgbm as lgb
import warnings

from .Bayes_opt import base_opt
from .utils import _check_objective_func, _check_param_bounds


__all__=['BayesianLGB']


class BayesianLGB(base_opt):

    def __init__(self,
                 early_stopping_rounds=500,
                 objective='rmse',
                 num_boost_round=10000,
                 param_bounds=None,
                 learning_rate=0.01,
                 **kwargs
                  ):
        super().__init__(**kwargs)
        if self.application not in {'regression', 'classification'}:
            raise ValueError('application must be either'
                             'regression or classification'
                             'found %s' % self.application)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.bayes_lr = learning_rate
        self.model_lr = min(.1, self.bayes_lr * 5)
        self._hyper_params_bounds = _check_param_bounds(param_bounds=param_bounds,
                                                        key='lgb',
                                                        allow_none=True)
        self.metric = _check_objective_func(self.application,
                                            objective)
        
        self._boosting_params = dict(
            application=self.application,
            boosting='gbdt',
            metric=self.metric,
            learning_rate=self.bayes_lr,
            verbosity=-1,
            data_random_seed=self.random_state
        )



    def _fit(self, X, y):

        def _train_lgb(**params):

            params = deepcopy(params)
            for param in ['num_leaves', 'max_depth', 'max_bin',
                          'bagging_freq', 'min_child_samples']:
                if param in params:
                    params[param] = int(params[param])
            params.update(self._boosting_params)

            kFold_splits = self.stratified_kfold(X, y)

            score = 0.
            for train_index, valid_index in list(kFold_splits):
                X_train = X[train_index]
                y_train = y[train_index]

                X_val = X[valid_index]
                y_val = y[valid_index]

                d_train = lgb.Dataset(X_train, label=y_train)
                d_valid = lgb.Dataset(X_val, label=y_val)
                watchlist = [d_train, d_valid]

                model = lgb.train(params=params,
                                  train_set=d_train,
                                  valid_sets=watchlist,
                                  early_stopping_rounds=self.early_stopping_rounds,
                                  verbose_eval=1000,
                                  num_boost_round=self.num_boost_round)

                score += self.score_func(model, X_val, y_val)

            return score

        self.set_bayes_opt(target_func=_train_lgb,
                           param_bounds=self._hyper_params_bounds)

        self._best_params = deepcopy(self.bayes_optimization())
        for param in ['num_leaves', 'max_depth', 'max_bin',
                      'bagging_freq', 'min_child_samples']:
            if param in self._best_params:
                self._best_params[param] = int(self._best_params[param])

    def fit(self, X, y, feature_name = None):

        self._fit(X, y)

        print('-' * 130)
        self._best_n_estimators = self._find_best_n_estimators(X, y)
        if self.application == 'regression':
            self.model = lgb.LGBMRegressor(n_estimators=self._best_n_estimators,
                                           objective=self.metric, 
                                           learning_rate=self.model_lr,
                                           **self._best_params)
        else:
            self.model = lgb.LGBMClassifier(n_estimator=self._best_n_estimators,
                                            objective=self.metric,
                                            learning_rate=self.model_lr,
                                            **self._best_params)

        self.model.fit(X, y, feature_name=feature_name)
        return self.model

    def _find_best_n_estimators(self, X, y):

        check_is_fitted(self, ['_best_params'])

        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          test_size=0.25, shuffle=True)
        lgb_train = lgb.Dataset(data=X_train, label=y_train)
        lgb_val = lgb.Dataset(data=X_val, label=y_val)

        params = deepcopy(self._best_params)
        params.update(self._boosting_params)
        params['learning_rate'] = self.model_lr
        model = lgb.train(params=params,
                          train_set=lgb_train,
                          valid_sets=[lgb_train, lgb_val],
                          early_stopping_rounds=self.early_stopping_rounds,
                          verbose_eval=1000,
                          num_boost_round=self.num_boost_round)

        return model.best_iteration


    def predict(self, X, y = None):

        check_is_fitted(self, ['model'])
        return self.model.predict(X)
    
    @property
    def best_params(self):
        check_is_fitted(self, ['_best_params', '_best_n_estimators'])
        params = deepcopy(self._best_params)
        params['n_estimators']=self._best_n_estimators
        params['learning_rate']=self.model_lr
        return params

    def plot_importance(self, **kwargs):
        check_is_fitted(self, ['model'])
        return lgb.plot_importance(booster=self.model,**kwargs)






















