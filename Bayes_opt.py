from bayes_opt import BayesianOptimization
from copy import deepcopy
from .utils import _get_scoring_func, _get_application
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.validation import check_is_fitted
import warnings


class base_opt(object):

    def __init__(self, application='regression', init_point=5,
                 n_iter=5, score=None,
                 n_splits=5, random_state=32):

        self._application = _get_application(application)
        self.init_points = init_point
        self.n_iter = n_iter
        self.score_func = _get_scoring_func(application=application,
                                            score=score)
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y, features_name=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    @property
    def best_params(self):
        raise NotImplementedError

    def stratified_kfold(self, X, y):
        return StratifiedKFold(n_splits=self.n_splits,
                               random_state=self.random_state,
                               shuffle=True).split(X, y)
    def kfold(self, X, y):
        return KFold(n_splits=self.n_splits,
                     random_state=self.random_state,
                     shuffle=True).split(X, y)

    def set_bayes_opt(self, target_func, param_bounds):
        self._bayes_opt = BayesianOptimization(target_func,
                                               param_bounds,
                                               self.random_state)

    def bayes_optimization(self):
        check_is_fitted(self, ['_bayes_opt'])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self._bayes_opt.maximize(init_points=self.init_points,
                                     n_iter=self.n_iter,
                                     acq='ucb',
                                     xi=0.0,
                                     alpha=1e-6
                                     )

        return self._bayes_opt.max['params']

