"""
A library for lightGBM parameter tuning using Bayesian Optimization
"""
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from lightgbm import LGBMClassifier, LGBMRegressor


from .Bayes_opt import base_opt
from .utils import _check_obj_and_metric, _check_param_bounds


__all__=['BayesianLGB']


class BayesianLGB(base_opt):

    def __init__(self,
                 early_stopping_rounds=500,
                 eval_metric='rmse',
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
        self.n_estimators = num_boost_round
        self.bayes_lr = learning_rate
        self.model_lr = min(.1, self.bayes_lr * 5)
        self._hyper_params_bounds = _check_param_bounds(param_bounds=param_bounds,
                                                        key='lgb',
                                                        allow_none=True)
        self.eval_metric = _check_obj_and_metric(self.application,
                                                 eval_metric)
        self.objective = _check_obj_and_metric(self.application,
                                               objective)
        self._additional_params = dict()
        self._model = LGBMRegressor if self.application == 'regression' else LGBMClassifier
        self._kFold_splits = lambda X, y: self.stratified_kfold(X, y)\
            if self.application == 'regression' else self.kfold(X, y)

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
                                    learning_rate=self.bayes_lr,
                                    random_state=self.random_state,
                                    n_jobs=-1,
                                    objective=self.objective,
                                    eval_metric=self.eval_metric,
                                    eval_set=(X_val, y_val),
                                    early_stopping_rounds=self.early_stopping_rounds,
                                    n_estimators=self.n_estimators,
                                    verbose=1000)
                model.set_params(**params)
                model.fit(X_train, y_train)
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
        objective = self.fobj if self.fobj is not None else self.metric
        if self.application == 'regression':
            self.model = lgb.LGBMRegressor(n_estimators=self._best_n_estimators,
                                           objective=objective,
                                           learning_rate=self.model_lr,
                                           random_state=self.random_state,
                                           **self._best_params)
        else:
            self.model = lgb.LGBMClassifier(n_estimator=self._best_n_estimators,
                                            objective=objective,
                                            learning_rate=self.model_lr,
                                            random_state=self.random_state,
                                            **self._best_params)
        self.model.set_params(**self._additional_params)
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

    def predict(self, X, y=None):

        check_is_fitted(self, ['model'])
        return self.model.predict(X)
    
    @property
    def best_params(self):
        check_is_fitted(self, ['_best_params', '_best_n_estimators'])
        params = deepcopy(self._best_params)
        params['n_estimators'] = self._best_n_estimators
        params['learning_rate'] = self.model_lr
        return params

    def plot_importance(self, **kwargs):
        check_is_fitted(self, ['model'])
        return lgb.plot_importance(booster=self.model,**kwargs)

    # def _set_boosting_params(self, key, value):
    #     self._boosting_params[key] = value

    def _set_additional_params(self, key, value):
        self._additional_params[key] = value

    def set_alpha(self, alpha):
        if self.metric == 'huber' or self.metric == 'quantile':
            if alpha <= 0.:
                raise ValueError('alpha should >0.')
            else:
                # self._set_boosting_params('alpha', alpha)
                self._set_additional_params('alpha', alpha)
        else:
            raise KeyError('Only huber and quantile loss have alpha parameter.')

    def set_scale_pos_weight(self, scale_pos_weight):
        pass























