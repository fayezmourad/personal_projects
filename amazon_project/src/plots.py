# Inspired by the examples of the course

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_validation_curve(param_name, param_range, train_scores_mean, val_scores_mean, train_scores_std=None, val_scores_std=None,
                          scoring='accuracy',save_file=None):
    """
    Plot the CV scores 
    :param param_name: 
    :param param_range: 
    :param train_scores_mean: 
    :param val_scores_mean: 
    :param train_scores_std: 
    :param val_scores_std: 
    :param scoring: 
    :return: 
    """
    # plot training errors and test errors as a function of alpha parameter to tune the model
    # and obtain optimum value of alpha
    plt.title("Validation curve on TRAINING SET")
    plt.xlabel(param_name)
    plt.ylabel(scoring)

    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="training score", color="darkorange", lw=lw)
    plt.semilogx(param_range, val_scores_mean, label="validation score", color="navy", lw=lw)
    if train_scores_std != None and val_scores_std != None:

        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)

        plt.fill_between(param_range, val_scores_mean - val_scores_std,
                         val_scores_mean + val_scores_std, alpha=0.1,
                         color="navy", lw=lw)
    plt.legend(loc="best")
    
    if not (save_file == None) :
        plt.savefig(save_file)
    plt.show()

def plot_2D_validation(param_ranges, param_names, val_scores_mean,save_file=None):
    """
    Plot a 2d Graph for 2 parameters(heatmap)
    :param param_ranges: 
    :param param_names: 
    :param val_scores_mean: 
    :return: 
    """
    val_scores_mean_reshaped=np.reshape(val_scores_mean,(param_ranges[0].shape[0],param_ranges[1].shape[0]))
    val_scores_df=pd.DataFrame(data=val_scores_mean_reshaped,index=param_ranges[0],columns=param_ranges[1])
    ax = sns.heatmap(val_scores_df)
    ax.set_ylabel(param_names[0])
    ax.set_xlabel(param_names[1])
    
    if not (save_file == None) :
        plt.savefig(save_file)
    plt.show()
        