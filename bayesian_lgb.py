from .base import base_opt
from .utils import _check_param_bounds, _get_default_bounds

from bayes_opt import BayesianOptimization
import lightgbm as lgb
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
import warnings

class BayesianLGB(base_opt):

    def __init__(self, init_points=5, n_iter=5,
                 n_splits=5, param_bounds=None, **kwargs):
        super().__init__(**kwargs)
        self._init_points = init_points
        self._n_iter = n_iter
        self.n_splits = n_splits
        self._param_bounds = _check_param_bounds(param_bounds)
        self._all_params = set(_get_default_bounds().keys())
        self.bayes_ops_params = dict(
            init_points=self._init_points,
            n_iter=n_iter,
            acq='ucb',
            xi=0.0,
            alpha=1e-6
        )

    def set_param_bound(self, param, bound):

        if param not in self._all_params:
            raise KeyError('The valid parameter sets are {0}'
                           'found {1}'.format(self._all_params, param))
        self._param_bounds[param] = bound


    def _fit(self, X, y):

        def _train_lgb(**params):

            params = deepcopy(params)
            for param in ['num_leaves', 'max_depth', 'max_bin',
                          'bagging_freq', 'min_child_samples']:
                if param in params:
                    params[param] = int(params[param])
            params.update(self.params)

            kFold = StratifiedKFold(n_splits=self.n_splits,
                                    random_state=self._random_state,
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
                                  early_stopping_rounds=self._early_stopping_rounds,
                                  verbose_eval=1000,
                                  num_boost_round=self._num_boost_round)

                score += self._score(model, X_val, y_val)

            return score

        self._LGB_BO = BayesianOptimization(_train_lgb,
                                            self._param_bounds,
                                            self._random_state
                                            )
        print('-' * 130)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self._LGB_BO.maximize(**self.bayes_ops_params)

        self._best_params = deepcopy(self._LGB_BO.max['params'])
        for param in ['num_leaves', 'max_depth', 'max_bin',
                      'bagging_freq', 'min_child_samples']:
            if param in self._best_params:
                self._best_params[param] = int(self._best_params[param])

    def fit(self, X, y, feature_name = None):

        self._fit(X, y)

        print('-' * 130)
        self._best_n_estimators = self._find_best_n_estimators(X, y, test_size=.25)
        lr = min(.1, self._learning_rate)
        if self._application == 'regression':
            self.model = lgb.LGBMRegressor(n_estimators=self._best_n_estimators,
                                           objective=self._objective,
                                           learning_rate=lr,
                                           **self._best_params)
        else:
            self.model = lgb.LGBMClassifier(n_estimator=self._best_n_estimators,
                                            objective=self._objective,
                                            learning_rate=lr,
                                            **self._best_params)

        self.model.fit(X, y, feature_name=feature_name)
        return self.model


