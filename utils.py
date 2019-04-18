from sklearn.metrics import get_scorer
from copy import deepcopy


def _get_scoring_func(application, score):

    if score is None:
        if application == 'regression':
            return get_scorer('neg_mean_squared_error')
        elif application in {'binary', 'multiclass'}:
            return get_scorer('neg_log_loss')
        else:
            raise ValueError
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
        elif application == 'binary':
            if score in {'f1','accuracy', 'average_precision',
                         'f1_macro','roc_auc','neg_log_loss',
                         'precision','recall'}:
                print('%s function does not take label imbalance into account.' 
                      'You may define your own scoring function'% score)
                return get_scorer(score)

            elif score in {'balanced_accuray', 'f1_micro', 'f1_weighted'}:
                return get_scorer(score)
        elif application =='multiclass':
            if score in {'accuracy', 'average_precision',
                         'f1_macro','roc_auc','neg_log_loss',}:
                print('%s function does not take label imbalance into account.' 
                      'You may define your own scoring function'% score)
                return get_scorer(score)

            elif score in {'f1_micro', 'f1_weighted'}:
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
    global DEFAULT_LGB_BOUNDS
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
APPLICATIONS = {
    'reg': {'regression', 'reg'},
    'binary': {'binary', 'classification:binary'},
    'multi': {'multi', 'multi-class', 'classification:multi'},
    'rank': {'rank', 'ranking'}
}


def _get_application(application):
    global APPLICATIONS
    if application in APPLICATIONS['reg']:
        return 'regression'
    elif application in APPLICATIONS['binary']:
        return 'binary'
    elif application in APPLICATIONS['multi']:
        return 'multiclass'
    elif application in APPLICATIONS['rank']:
        return 'rank'
    else:
        raise ValueError('%s is not a valid application.' % application)


def _sklearn_fn2lgb_fn(func):
    def wrapper(preds, train_data):
        tests = train_data.get_label()
        return func(tests, preds)
    return wrapper
















