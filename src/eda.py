import json,sys, os, glob, re
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

import warnings
warnings.filterwarnings("ignore")

from helper import *

def main_eda(cond, lst, filen1, filen2, filen3):
    unseen = ''
    if cond =='unseen': 
        unseen = 'unseen'
    
    fpath1 = os.path.join(os.getcwd() , "outputs", unseen + filen1)
    df_1 = pd.read_csv(fpath1)
    fpath2 = os.path.join(os.getcwd() , "outputs", unseen + filen2)
    df_2 = pd.read_csv(fpath2)
    fpath3 = os.path.join(os.getcwd() , "outputs", unseen + filen3)
    df_3 = pd.read_csv(fpath3)
    
    plottogether(cond, lst, df_1, filen1.strip(".csv")) # trends over subset
    plottogether(cond, lst, df_3, filen3.strip(".csv")) # trends over entire data
    # plotloss(cond, df_2)

    plot_correlation_matrix(cond, df_2) # correlation matrix
    plotlongest(df_3, cond, 6000, 14000)
    # below makes rest of visualizations
    plotbytes(df_3, 100, 15000, 100, 500, cond)
    #plot_detailed_bytes()

def model_eda():
    '''uses some unseen data to show model performance. assumes unseen data is alr run'''
    tempdatac = 'data/temp/tempdata_c' # TODO PUT in other repo
    data = listdir(tempdatac)
    if len(data) == 0:
        print('warning: no tempdata to work with for model_eda')
        return
    df = genfeat(pd.read_csv(os.path.join(tempdatac, data[0])))
    
    plot_detailed_bytes(df)

    plot_model_predictions(df, label='loss')
    plot_model_predictions(df, label='latency')
    print('model predictions plotted')

def plotbytes(df, loss1, loss2, lat1, lat2, cond):
    unseen = ''
    if cond =='unseen': 
        unseen = 'unseen'
    # series over loss
    lst_loss = ["byte_ratio", "pkt_ratio", "time_spread", "total_bytes", "2->1Pkts"]
    # series over latency
    lst_lat = [
            "total_pkts",
            "total_pkt_sizes",
            "2->1Bytes",
            "number_ms",
            "mean_tdelta",
            "max_tdelta",
            "time_spread",
            "longest_seq",
        ]
    for i in lst_loss: 
        l1 = df[df['loss'] == loss1]
        l2 = df[df['loss'] == loss2]
        byte_agg1 = l1.groupby('Second').median().reset_index()[['Second', i,'loss']]
        byte_agg2 = l2.groupby('Second').median().reset_index()[['Second', i,'loss']]
        plt.figure(figsize = (12,5))
        plt.plot(byte_agg1['Second'], byte_agg1[i], label = str(loss1))
        plt.plot(byte_agg2['Second'], byte_agg2[i], label = str(loss2))
        plt.legend(title = "Packet Loss (1/x)", loc="upper right")
        plt.xlabel('Seconds')
        plt.ylabel(i.replace("_", '').capitalize())
        plt.title('Median ' + i.replace("_", ' ').capitalize() + ' Per Second')
        path = os.path.join(os.getcwd() , "outputs")
        saveto = os.path.join(path, "eda",unseen +  i.replace('>', '')  + ".png")
        plt.savefig(saveto)

        ax = plt.figure(figsize = (5,5))
        ax.suptitle(f'{i} Density Plot')
        sns.displot(pd.concat([l1, l2]), x=i, hue="loss", kind="kde", palette=['tab:blue', 'tab:orange'], ax=ax
            ).set(title=f'{i} Density Plot')
        saveto2 = os.path.join(path, "eda",unseen +  i.replace('>', '')  + "distplot.png")
        plt.savefig(saveto2)
        
    for i in lst_lat:
        l1 = df[df['latency'] == lat1]
        l2 = df[df['latency'] == lat2]
        byte_agg1 = l1.groupby('Second').median().reset_index()[['Second', i,'latency']]
        byte_agg2 = l2.groupby('Second').median().reset_index()[['Second', i,'latency']]
        plt.figure(figsize = (12,5))
        plt.plot(byte_agg1['Second'], byte_agg1[i], label = str(lat1))
        plt.plot(byte_agg2['Second'], byte_agg2[i], label = str(lat2))
        plt.legend(title = "Latencies", loc="upper right")
        plt.xlabel('Seconds')
        plt.ylabel(i.replace("_", '').capitalize())
        plt.title('Median ' + i.replace("_", '').capitalize() + ' Per Second')
        path = os.path.join(os.getcwd() , "outputs")
        saveto = os.path.join(path, "eda",unseen + i.replace('>', '') + ".png")
        plt.savefig(saveto)
        
        ax = plt.figure(figsize = (5,5))
        ax.suptitle(f'{i} Density Plot')
        sns.displot(pd.concat([l1, l2]), x=i, hue="latency", kind="kde", palette=['tab:blue', 'tab:orange'], ax=ax
            ).set(title=f'{i} Density Plot')
        saveto2 = os.path.join(path, "eda",unseen +  i.replace('>', '')  + "distplot.png")
        plt.savefig(saveto2)
        
    return

def plotlongest(df, cond, loss1, loss2):
    unseen = ''
    if cond =='unseen': 
        unseen = 'unseen'
    l1 = df[df['loss'] == loss1]
    l2 = df[df['loss'] == loss2]
    byte_agg1 = l1.groupby('Second').sum().reset_index()[['Second', 'longest_seq','loss']]
    byte_agg2 = l2.groupby('Second').sum().reset_index()[['Second', 'longest_seq','loss']]
    plt.figure(figsize = (15,10))
    plt.plot(byte_agg1['Second'], byte_agg1['longest_seq'], label = str(loss1))
    plt.plot(byte_agg2['Second'], byte_agg2['longest_seq'], label = str(loss2))
    plt.legend(title = "Packet Loss", loc="upper right")
    plt.xlabel('Seconds')
    plt.ylabel('Longest Sequence')
    plt.title('Longest Sequence Per Second')
    path = os.path.join(os.getcwd() , "outputs")
    saveto = os.path.join(path, "eda",unseen +  "longest_seq.png")
    plt.savefig(saveto)

def plot_correlation_matrix(cond, df):
    unseen = ''
    if cond =='unseen': 
        unseen = 'unseen'
    featurelst = ["byte_ratio", "pkt_ratio", "time_spread", "total_bytes", 
        "2->1Pkts", "total_pkts", "total_pkt_sizes", "2->1Bytes", "number_ms",
        "mean_tdelta", "max_tdelta", "time_spread", "longest_seq"]
    featurelst.sort()
    
    corrmat = df[featurelst].corr()
    top_corr_features = corrmat.index
    fig = plt.figure(figsize=(10,10))
    sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn", vmin=-1, vmax=1)
    #savefig
    path = os.path.join(os.getcwd() , "outputs")
    saveto = os.path.join(path, "eda", unseen + "correlation_matrix.png")
    fig.savefig(saveto)


def plot_main4(cond, df_1, l1, df_2, l2, picname):
    unseen = ''
    if cond =='unseen': 
        unseen = 'unseen'
    #separating all of the aggregates for each loss

    tp_sum_agg, tb_agg  = main2(df_1) #tp_agg

    tp_sum_agg2, tb_agg2 = main2(df_2) # tp_agg2 

    label = 'loss'
  
    fig, axes = plt.subplots(2, 2,figsize=(18, 10))#, sharex=True)
    #print()
    sns.lineplot(ax=axes[0, 0], x = 'Second', y = 'sum', data = tp_sum_agg, hue = label)
    axes[0, 0].set_title("Packets Per Second over latency for run " + l1)
    sns.lineplot(ax=axes[0, 1], x = 'Second', y = 'sum', data = tp_sum_agg2, hue = label)
    axes[0, 1].set_title("Packets Per Second over latency for run " + l2)
    
    sns.lineplot(ax=axes[1, 0], x = 'Second', y = 'sum', hue = label , data = tb_agg)
    axes[1, 0].set_title("Bytes over packet latency for run " + l1)
    sns.lineplot(ax=axes[1, 1], x = 'Second', y = 'sum', hue = label , data = tb_agg2)
    axes[1, 1].set_title("Bytes over packet latency for run " + l2)

#     sns.lineplot(ax=axes[2, 0], x = 'Second', y = 'sum', hue = label , data = tp_agg)
#     axes[2, 0].set_title("Pkts over packet loss for run " + l1)
#     sns.lineplot(ax=axes[2, 1], x = 'Second', y = 'sum', hue = label , data = tp_agg2)
#     axes[2, 1].set_title("Pkts over packet loss for run " + l2)
    plt.subplots_adjust(hspace = 0.8)
    #savefig
    path = os.path.join(os.getcwd() , "outputs")
    saveto = os.path.join(path, "eda", unseen + "latency_trends_c" + picname + ".png")
    fig.savefig(saveto)
    print("plot_main4() executed")

def plottogether(cond, lst, df_e, picname): 
    unseen = ''
    if cond =='unseen': 
        unseen = 'unseen'
    leftrun = lst[0]
    rightrun = lst[1]
    ll = 'latency'
    values = df_e[ll].unique()
    # print(values)
    subset1 = df_e[df_e[ll] == leftrun]
    #subset1_2 = df_e[(df_e['loss'] >= 200) & (df_e['loss'] <= 10000)]
    subset2 = df_e[df_e[ll] == rightrun]
    #subset2_2 = df_e[(df_e['loss'] >= 200) & (df_e['loss'] <= 10000)]
    # print(subset1.shape)
    # print(subset2.shape)
    plot_main4(cond, subset1, str(leftrun), subset2, str(rightrun), picname)
    return subset1, subset2

def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
        source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(5, 3))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def plotloss(cond, df):
    unseen = ''
    if cond =='unseen': 
        unseen = 'unseen'
    fig, axes = plt.subplots(3, 4, figsize=(15, 15))

    X = df.drop(['latency'], axis=1)
    y = df.latency

    title = "Learning Curves (DecisionTree)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = DecisionTreeRegressor()
    plot_learning_curve(
        estimator, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4
    )

    title = r"Learning Curves (RandomForest)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = RandomForestRegressor(n_estimators=10)
    plot_learning_curve(
        estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4
    )

    title = r"Learning Curves (ExtraTrees)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = ExtraTreesRegressor(n_estimators=10)
    plot_learning_curve(
        estimator, title, X, y, axes=axes[:, 2], ylim=(0.7, 1.01), cv=cv, n_jobs=4
    )
    
    title = r"Learning Curves (GradientBoost)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = GradientBoostingRegressor(random_state=0)
    plot_learning_curve(
        estimator, title, X, y, axes=axes[:, 3], ylim=(0.7, 1.01), cv=cv, n_jobs=4
    )

    #plt.show()
    #savefig
    path = os.path.join(os.getcwd() , "outputs")
    saveto = os.path.join(path, "eda", unseen + "learning_curves.png")
    fig.savefig(saveto)
    
def plot_detailed_bytes(df, col='1->2Bytes', rollsec=10):
    rollcolor = '#6c2b6d'
    detailcolor = '#e98d6b'

    ax = plt.figure(figsize=(18,8))
    df[col].plot(title=f'{col}/s Rate', color=detailcolor)
    df[col].rolling(rollsec).mean().bfill().plot(color=rollcolor)
    plt.axvline(x=180, color='r')
    for i in df[df['event'] == 'drop'].index:
        plt.axvline(x=i, color='y', alpha=.45)
    custom_lines = [Line2D([0], [0], color=detailcolor, lw=2),
        Line2D([0], [0], color=rollcolor, lw=2),
        Line2D([0], [0], color='y', lw=2, alpha=0.45),
        Line2D([0], [0], color='r', lw=2)]
    plt.legend(custom_lines, 
               [f'{col} per Second', f'{col} per Second ({rollsec}s rolling avg)', 'Packet drop', '180s Mark'], 
               loc='upper right', framealpha=1);
    path = os.path.join(os.getcwd() , "outputs")
    saveto = os.path.join(path, "eda",f'trendline_unseen.png')
    plt.savefig(saveto)

def plot_model_predictions(df, label='loss', windowsize=25, shift_time=180):
    '''requires a dataframe merged with with losslog data, plots its shifttime'''

    df = df.copy()
    if(label=='loss'):
        # latency features
        indexcol = ['byte_ratio', 'pkt_ratio', 'time_spread', 'total_bytes', '2->1Pkts']
    else:
        # loss features
        indexcol = ['total_pkts', 'total_pkt_sizes', '2->1Bytes', 'number_ms', 'mean_tdelta', 'max_tdelta'] 
        
    # print(len(indexcol),df[[x for x in indexcol if x in df.columns]])
    ct = pd.read_csv('outputs/combined_transform.csv').rename(
        {'mean_tdelta.1': 'mean_tdelta_amin', 'mean_tdelta.2':'mean_tdelta_amax'}, 
        axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        ct[[x for x in indexcol if x in df.columns]], ct[label])
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    etree = RandomForestRegressor(n_estimators=100, n_jobs=4)
    etreeft = etree.fit(X_train,y_train)

    df['prediction'] = etree.predict(df[indexcol].rolling(10).mean().bfill()) # prediction

    # fig = plt.figure()
    def emp_loss(df, window=25):
        '''returns empirical loss over a window of time.'''
        return (df['total_pkts'].rolling(window).sum() / 
            df['event'].str.replace('switch', '').str.split(';').str.len().fillna(0).rolling(window).sum())


    df[[label, f'later_{label}', 'prediction']].plot(
        figsize=(14,6), title=f'Real Time Prediction on {label}', 
        xlabel='Time (sec)', ylabel=label)
    if (label=='loss'):
        emp_loss(df, windowsize).plot()
    plt.axvline(x=shift_time, color='r')
    if(label=='loss'):
        plt.legend([
            f'Early {label} label', 
            f'Later {label} label', 
            'Prediction',
            f'Empirical Loss (window {windowsize} sec)',
            'Anomalous switch event'
            ], loc='center left')
    else:
        plt.legend([
            f'Early {label} label', 
            f'Later {label} label', 
            'Prediction',
            'Anomalous switch event'
            ], loc='center left')
    for i in df[~df['event'].isnull()].index:
        plt.axvline(x=i, color='y', alpha=.45)
    path = os.path.join(os.getcwd() , "outputs")
    saveto = os.path.join(path, "eda",f'{label}_model_preds.png')
    plt.savefig(saveto)