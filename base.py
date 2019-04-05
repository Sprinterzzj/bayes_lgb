from copy import deepcopy
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from .utils import _check_score_func, _check_objective_func


class base_opt(object):

    def __init__(self, application='regression',
                 boosting='gbdt', learning_rate=.01,
                 objective='rmse', random_state=32,
                 num_boost_round=10000, early_stopping_rounds=300,
                 score='rmse'
                 ):
        """

        :param application:
        :param boosting:
        :param learning_rate:
        :param metric:
        :param random_state:
        """
        self._application = application
        self._boosting = boosting
        self._learning_rate = learning_rate
        self._objective = _check_objective_func(self._application, objective)
        self._random_state = random_state
        self.params = dict(
            application=self._application,
            boosting=self._boosting,
            learning_rate=self._learning_rate,
            metric=self._objective,
            data_random_seed=self._random_state,
            verbosity=-1
        )
        self._num_boost_round = num_boost_round
        self._early_stopping_rounds = early_stopping_rounds
        self._score = _check_score_func(self._application, score)

    def fit(self, X, y, feature_name = None):

        raise NotImplementedError

    def _find_best_n_estimators(self, X, y, test_size):

        check_is_fitted(self, ['_best_params'])

        X_train, X_val, y_train, y_val =\
            train_test_split(X, y, test_size=test_size, shuffle=True)
        lgb_train = lgb.Dataset(data=X_train, label=y_train)
        lgb_val = lgb.Dataset(data=X_val, label=y_val)

        params = deepcopy(self._best_params)
        params.update(self.params)
        params['learning_rate'] = min(.1, 5 * self._learning_rate)

        model = lgb.train(params=params,
                          train_set=lgb_train,
                          valid_sets=[lgb_train, lgb_val],
                          early_stopping_rounds=self._early_stopping_rounds,
                          num_boost_round=self._num_boost_round,
                          verbose_eval=1000
                          )
        return model.best_iteration

    def predict(self, X, y=None):
        check_is_fitted(self, ['model'])
        return self.model.predict(X)

    @property
    def best_params(self):
        check_is_fitted(self, ['_best_params', '_best_n_estimators'])
        params = deepcopy(self._best_params)
        params['n_estimators'] = self._best_n_estimators
        params['learning_rate'] = min(.1, 5 * self._learning_rate)
        return params

    def plot_importance(self, **kwargs):
        check_is_fitted(self, ['model'])
        return lgb.plot_importance(booster=self.model, **kwargs)

    def predict_proba(self, X, y=None):
        check_is_fitted(self, ['model'])
        if self._application == 'classification':
            return self.model.predict_proba(X)
        else:
            raise AttributeError("Only classification task"
                                 "has predict_proba.")

