import time
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from src.plots import plot_validation_curve, plot_2D_validation
from sklearn.model_selection import KFold
import warnings


def parameter_search(estimator, param_grid, X_train, y_train, cv=5, scoring='accuracy', verbose=2, save_file=None,
                     n_jobs=-1):
    """
    Search a parameter with GridSearchCV. Supports 1D (plot) and 2D (seaborn heatmap) 
    :param estimator: 
    :param param_grid: Dictionary {parameter_name : parameter_array}
    :param X_train: 
    :param y_train: 
    :param cv: 
    :param scoring: 
    :param verbose: 
    :return: 
    """
    start = time.time()
    param_name = []
    param_range = []
    # Get the parameters name and range for the plot
    for key in param_grid:
        param_name.append(key)
        param_range.append(param_grid[key])

    gsCV = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring, verbose=verbose, n_jobs=n_jobs)
    gsCV.fit(X_train, y_train)
    val_scores_mean = gsCV.cv_results_['mean_test_score']
    val_scores_std = gsCV.cv_results_['std_test_score']
    train_scores_mean = gsCV.cv_results_['mean_train_score']
    train_scores_std = gsCV.cv_results_['std_train_score']

    if len(param_grid) == 1:
        plot_validation_curve(param_name[0], param_range[0], train_scores_mean, val_scores_mean, train_scores_std,
                              val_scores_std, scoring, save_file=save_file)
    else:
        plot_2D_validation(param_range, param_name, val_scores_mean, save_file=save_file)
    print("Best {} =".format(param_name), gsCV.best_params_, "with score :", gsCV.best_score_)

    end = time.time()
    print("Execution time : {} seconds".format(end - start))

    return gsCV


def parameter_search_feature_selection(estimator, feature_selector, param_name, param_range, X, y, cv=5,
                                       scoring='accuracy', random_state=None, save_file=None):
    """
    Parameter search for the feature selection
    :param estimator: 
    :param feature_selector: 
    :param param_name: 
    :param param_range: 
    :param X: 
    :param y: 
    :param cv: 
    :param scoring: supports only accuracy for the moment
    :param random_state: 
    :return: 
    """
    warnings.filterwarnings('ignore')
    start = time.time()
    if scoring == 'accuracy':
        score_func = accuracy_score
    else:
        # Need to modify the score func and plot_validation_curve function
        raise Exception('Scoring functions other than accuracy are undefined')

    val_scores = np.zeros((len(param_range), cv))
    train_scores = np.zeros((len(param_range), cv))

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_counter = 0
    for k_idx, k in enumerate(param_range):
        cv_counter = 0
        for train, test in kf.split(X):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            feature_selector.set_params(**{param_name: k})
            X_train_reduced = feature_selector.fit_transform(X_train, y_train)
            X_test_reduced = feature_selector.transform(X_test)
            # Trying it
            estimator.fit(X_train_reduced, y_train)
            y_pred_test = estimator.predict(X_test_reduced)
            y_pred_train = estimator.predict(X_train_reduced)
            val_scores[k_idx, cv_counter] = score_func(y_test, y_pred_test)
            train_scores[k_idx, cv_counter] = score_func(y_train, y_pred_train)

            cv_counter += 1
    val_mean = np.mean(val_scores, axis=1)

    plot_validation_curve(param_name, param_range, np.mean(train_scores, axis=1), val_mean,
                          np.std(train_scores, axis=1), np.std(val_scores, axis=1), scoring=scoring,
                          save_file=save_file)

    idx_max = np.argmax(val_mean)

    print("Best {} =".format(param_name), param_range[idx_max], "with score :", val_mean[idx_max])

    end = time.time()

    print("Execution time : {} seconds".format(end - start))
