"""
A library for lightGBM parameter tuning using Bayesian Optimization
"""
from bayes_opt import BayesianOptimization
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.validation import check_is_fitted

import lightgbm as lgb
import warnings

from .utils import check_eval_metric, check_score_func
from .utils import DEFAULT_BOUNDS


class BayesianLGB(object):

    def __init__(self,
                 lgb_application='regression',
                 lgb_early_stop=500,
                 lgb_metric='rmse',
                 lgb_num_boost_round=10000,
                 lgb_param_bounds=None,
                 lgb_verbose=1000,
                 lgb_learning_rate=0.01,

                 bayes_init_points=5,
                 bayes_n_iter=5,
                 bayes_score='rmse',

                 n_splits = 5,
                 random_state = 32,

                  ):
        """

        :param lgb_application:
        :param lgb_early_stop:
        :param lgb_metric:
        :param lgb_num_boost_round:
        :param lgb_param_bounds:
        :param lgb_verbose:
        :param lgb_learning_rate:
        :param bayes_init_points:
        :param bayes_n_iter:
        :param bayes_score:
        :param n_splits:
        :param random_state:
        """

        if lgb_application not in {'regression', 'classification'}:
            raise ValueError('\'application\' must be'
                             'either regression or classification')
        self.task = lgb_application
        self.early_stopping_rounds = lgb_early_stop
        self.verbose_eval = lgb_verbose
        self.num_boost_round = lgb_num_boost_round
        self.n_splits = n_splits
        self.random_state = random_state
        self.bayes_lr = lgb_learning_rate
        self.model_lr = min(.1, self.bayes_lr * 5)
        self._hyper_params_bounds = self._check_param_bounds(lgb_param_bounds)
        self.metric = check_eval_metric(self.task, lgb_metric)
        
        self._boosting_params = dict(
            application=self.task,
            boosting='gbdt',
            metric=self.metric,
            learning_rate=self.bayes_lr,
            verbosity=-1,
            data_random_seed=self.random_state
        )
        self._bayes_score = check_score_func(self.task, bayes_score)
        self._bayes_ops_params = dict(
            init_points=bayes_init_points,
            n_iter=bayes_n_iter,
            acq='ucb',
            xi=0.0,
            alpha=1e-6
        )


    def _check_param_bounds(self, param_bounds, allow_none = True):
        """

        To do: check the bound of each parameters
        :param param_bounds:
        :param allow_none:
        :return:
        """

        default_bounds = deepcopy(DEFAULT_BOUNDS)
        param_bounds = deepcopy(param_bounds)

        if param_bounds is None:
            if allow_none:
                return default_bounds
            else:
                raise ValueError('param_bounds should not be None.')
        if not(set(param_bounds.keys()) <= set(default_bounds.keys())):
            raise KeyError('The parameters should be {0},but {1} are found.'.\
                           format(set(default_bounds.keys()),
                                  set(param_bounds.keys()) - set(default_bounds.keys())))
        else:
            return param_bounds

    def _fit(self, X, y):

        def _train_lgb(**params):

            params = deepcopy(params)
            for param in ['num_leaves', 'max_depth', 'max_bin',
                          'bagging_freq', 'min_child_samples']:
                if param in params:
                    params[param] = int(params[param])
            params.update(self._boosting_params)

            kFold = StratifiedKFold(n_splits=self.n_splits,
                                    random_state=self.random_state,
                                    shuffle=True)
            kFold_splits = kFold.split(X, y)

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
                                  verbose_eval=self.verbose_eval,
                                  num_boost_round=self.num_boost_round)

                score += self._bayes_score(model, X_val, y_val)

            return score

        self._LGB_BO = BayesianOptimization(_train_lgb,
                                            self._hyper_params_bounds,
                                            self.random_state
                                            )
        print('-' * 130)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self._LGB_BO.maximize(**self._bayes_ops_params)

        self._best_params = deepcopy(self._LGB_BO.max['params'])
        for param in ['num_leaves', 'max_depth', 'max_bin',
                      'bagging_freq', 'min_child_samples']:
            if param in self._best_params:
                self._best_params[param] = int(self._best_params[param])

    def fit(self, X, y):

        self._fit(X, y)

        print('-' * 130)
        self._best_n_estimators = self._find_best_n_estimators(X, y)
        if self.task == 'regression': 
            self.model = lgb.LGBMRegressor(n_estimators=self._best_n_estimators,
                                           objective = self.metric, 
                                           learning_rate=_learning_rate,
                                           **self._best_params)
        else:
            self.model = lgb.LGBMClassifier(n_estimator=self._best_n_estimators,
                                            objective = self.metric,
                                            learning_rate=_learning_rate,
                                            **self._best_params)

        self.model.fit(X, y)
        return self.model

    def _find_best_n_estimators(self, X, y):

        check_is_fitted(self, ['_best_params'])

        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          test_size=0.25, shuffle=True)
        lgb_train = lgb.Dataset(data=X_train, label=y_train)
        lgb_val = lgb.Dataset(data=X_val, label=y_val)

        params = deepcopy(self._best_params)
        params.update(self._boosting_params)
        params['learning_rate'] = min(.1, self.learning_rate * 5)
        model = lgb.train(params=params,
                          train_set=lgb_train,
                          valid_sets=[lgb_train, lgb_val],
                          early_stopping_rounds=self.early_stopping_rounds,
                          verbose_eval=self.verbose_eval,
                          num_boost_round=self.num_boost_round)

        return model.best_iteration


    def predict(self, X, y = None):

        check_is_fitted(self, ['model'])
        return self.model.predict(X)
    
    @property
    def best_params(self):
        check_is_fitted(self, ['_best_params', '_best_n_estimators'])
        params = deepcopy(self._best_params)
        params['n_estimators'] = self._best_n_estimators
        return params






















