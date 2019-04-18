"""
A library for lightGBM parameter tuning using Bayesian Optimization
"""
from copy import deepcopy
from lightgbm import (cv, Dataset,
                      LGBMClassifier, LGBMRegressor, LGBMRanker,
                      plot_importance)
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted




from .Bayes_opt import base_opt
from .utils import _check_obj_and_metric, _check_param_bounds, _sklearn_fn2lgb_fn


__all__=['BayesianLGB']


class BayesianLGB(base_opt):

    def __init__(self,
                 early_stopping_rounds=500,
                 eval_metric='rmse',
                 objective='rmse',
                 num_class=None,
                 class_weight=None,
                 scale_pos_weight=1,
                 num_boost_round=10000,
                 param_bounds=None,
                 learning_rate=0.05,
                 lr_ratio=10,
                 **kwargs
                  ):
        super().__init__(**kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.n_estimators = num_boost_round
        self.learning_rate = learning_rate
        self.lr = self.learning_rate / lr_ratio
        self._hyper_params_bounds = _check_param_bounds(param_bounds=param_bounds,
                                                        key='lgb',
                                                        allow_none=True)
        self.eval_metric = _check_obj_and_metric(self._application,
                                                 eval_metric)
        self.objective = _check_obj_and_metric(self._application,
                                               objective)
        self._additional_params = dict()

        if self._application == 'regression':
            self._model = LGBMRegressor
        elif self._application in {'binary', 'multiclass'}:
            self._model = LGBMClassifier
        else:
            self._model = LGBMRanker

        if self._application == 'regression':
            self._kFold_splits = self.kfold
        elif self._application in {'binary', 'milticlass'}:
            self._kFold_splits = self.stratified_kfold

        if self._application == 'multiclass':
            if num_class is None and not callable(self.objective):
                raise ValueError('You must set num_class in'
                                 'multi-class classification.')
            self.num_class = num_class
            self.class_weight = class_weight
            self._additional_params['num_class'] = self.num_class
            self._additional_params['class_weight'] = self.class_weight
        elif self._application == 'binary':
            if scale_pos_weight is not None:
                self.scale_pos_weight = scale_pos_weight
                self._additional_params['scale_pos_weight'] =\
                    self.scale_pos_weight
            else:
                self.class_weight = class_weight
                self._additional_params['class_weight'] =\
                    self.class_weight

    def _fit(self, X, y):

        def _train_lgb(**params):

            params = deepcopy(params)
            for param in ['num_leaves', 'max_depth', 'max_bin',
                          'bagging_freq', 'min_child_samples']:
                if param in params:
                    params[param] = int(params[param])
            score = 0.
            for train_index, valid_index in list(self._kFold_splits(X, y)):

                X_train = X[train_index]
                y_train = y[train_index]

                X_val = X[valid_index]
                y_val = y[valid_index]

                model = self._model(boosting_type='gbdt',
                                    learning_rate=self.learning_rate,
                                    random_state=self.random_state,
                                    n_jobs=-1,
                                    objective=self.objective,
                                    n_estimators=self.n_estimators
                                    )
                model.set_params(**params)
                model.set_params(**self._additional_params)
                model.fit(X=X_train,
                          y=y_train,
                          eval_metric=self.eval_metric,
                          eval_set=(X_val, y_val),
                          early_stopping_rounds=self.early_stopping_rounds,
                          verbose=1000
                          )
                score += self.score_func(model, X_val, y_val)

            return score

        self.set_bayes_opt(target_func=_train_lgb,
                           param_bounds=self._hyper_params_bounds)

        self._best_params = deepcopy(self.bayes_optimization())
        for param in ['num_leaves', 'max_depth', 'max_bin',
                      'bagging_freq', 'min_child_samples']:
            if param in self._best_params:
                self._best_params[param] = int(self._best_params[param])

    def fit(self, X, y, feature_name=None):

        self._fit(X, y)

        print('-' * 130)
        print('Best Parameters are {0}'.format(self._best_params))
        print('-' * 130)
        print('Begin find best n_estimators.')
        self._best_iteration = self._tuning_best_iteration(X, y, self.lr)
        self.model = self._model(boosting_type='gbdt',
                                 learning_rate=self.lr,
                                 random_state=self.random_state,
                                 n_jobs=-1,
                                 objective=self.objective,
                                 n_estimators=self._best_iteration
                                 )
        self.model.set_params(**self._best_params)
        self.model.set_params(**self._additional_params)
        self.model.fit(X, y, feature_name=feature_name)
        return self.model

    def _tuning_best_iteration(self, X, y, learning_rate):
        check_is_fitted(self, ['_best_params'])
        X_train, X_test, y_train, y_test =\
            train_test_split(X, y, shuffle=True)

        model = self._model(boosting_type='gbdt',
                            learning_rate=learning_rate,
                            random_state=self.random_state,
                            n_jobs=-1,
                            objective=self.objective,
                            n_estimators=self.n_estimators
                            )
        model.set_params(**self._best_params)
        model.set_params(**self._additional_params)
        model.fit(X=X_train,
                  y=y_train,
                  eval_metric=self.eval_metric,
                  eval_set=(X_test, y_test),
                  early_stopping_rounds=self.early_stopping_rounds,
                  verbose=1000
                  )
        return model.best_iteration_

    def predict(self, X, y=None):

        check_is_fitted(self, ['model'])
        return self.model.predict(X)
    
    @property
    def best_params(self):
        check_is_fitted(self, ['_best_params', '_best_iteration', 'lr'])
        params = deepcopy(self._best_params)
        params['n_estimators'] = self._best_iteration
        params['learning_rate'] = self.lr
        return params

    def plot_importance(self, **kwargs):
        check_is_fitted(self, ['model'])
        return plot_importance(booster=self.model, **kwargs)

    # def _set_boosting_params(self, key, value):
    #     self._boosting_params[key] = value

    def _set_additional_params(self, key, value):
        self._additional_params[key] = value


##############################################################
    def set_alpha(self, alpha):
        if self.objective in {'huber','quantile'}:
            if alpha <= 0.:
                raise ValueError('alpha should >0.')
            else:
                self._set_additional_params('alpha', alpha)
        else:
            raise KeyError('Only huber and quantile loss have alpha parameter.')

    def set_scale_pos_weight(self, scale_pos_weight):
        pass























