from sklearn.metrics import get_scorer
from copy import deepcopy


def _check_score_func(application, score):

    if application not in {'regression', 'classification'}:
        raise ValueError('application should be either regression or classification')

    if score is None:
        if application == 'regression':
            return get_scorer('neg_mean_squared_error')
        else:
            return get_scorer('accuracy')
    #From sklearn/metric/scorer.py
    elif callable(score):
        # Heuristic to ensure user has not passed a metric
        module = getattr(score, '__module__', None)
        if hasattr(module, 'startswith') and \
           module.startswith('sklearn.metrics.') and \
           not module.startswith('sklearn.metrics.scorer') and \
           not module.startswith('sklearn.metrics.tests.'):
            raise ValueError('scoring value %r looks like it is a metric '
                             'function rather than a scorer. A scorer should '
                             'require an estimator as its first parameter. '
                             'Please use `make_scorer` to convert a metric '
                             'to a scorer.' % score)
        return get_scorer(score)

    elif isinstance(score, str):
        if application == 'regression' and\
           score in {'neg_mean_absolute_error', 'neg_mean_squared_error',
                     'neg_mean_squared_log_error', 'neg_median_absolute_error'}:
            return get_scorer(score)
        elif application == 'classification' and\
             score in {'accuracy', 'balanced_accuray', 'average_precision',
                      'f1', 'f1_micro', 'f1_macro', 'roc_auc', 'precision',
                      'recall', 'neg_log_loss', 'roc_auc'}:
            return get_scorer(score)
        else:
            raise ValueError('%r is not a valid score value. '
                             'Use sorted(sklearn.metrics.SCORERS.keys()) '
                             'to get valid options.' % (score))
    else:
        raise ValueError('score value should either be a callable, string or"\
                         " None. %r was passed" % scoring')


def _check_obj_and_metric(application, function, allow_none=True):

    if application not in {'regression', 'classification'}:
        raise ValueError('application should be either regression or classification')

    if function is None:
        if allow_none:
            return None
        else:
            raise ValueError('eval_metric should not be None.')

    if application == 'regression' and\
       function in {'rmse', 'mae', 'mse', 'mape', 'huber', 'quantile'}:
        return function
    elif application == 'classification' and\
         function in {'binary', 'binary_error', 'softmax', 'multi_error'}:
        return function
    else:
        raise ValueError('%r is not a valid eval_metric value.' % function)


def _check_param_bounds(param_bounds,key,allow_none=True):

        default_bounds = deepcopy(_get_default_params(key=key))
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


def _get_default_params(key='lgb'):
    if key == 'lgb':
        return deepcopy(DEFAULT_LGB_BOUNDS)
    else:
        raise KeyError


DEFAULT_LGB_BOUNDS = dict(
    num_leaves=(30, 200),
    max_depth=(5, 15),
    max_bin=(20, 80),
    subsample=(0.5, 1.0),
    bagging_freq=(1, 50),
    colsample_by_tree=(0.5, 0.8),
    min_split_gain=(0.0, 1.0),
    min_child_samples=(25, 125),
    min_child_weight=(0.0, 1.0),
    reg_alpha=(0.0, 5.0),
    reg_lambda=(0.0, 5.0)

)















