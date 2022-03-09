import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
)
import warnings

warnings.filterwarnings("ignore")

from helper import *
from etl import *


def test_feat(cond, df, cols, p, df_u):
    """tests different features on different data combinations"""
    unseen = ""
    if cond == "unseen":
        unseen = "unseen"
    # col is feauture comb
    # p is for loss or latency   1: loss  # 2 : latency
    X = df[cols]

    X2 = df_u[cols]

    if p == 1:  # flag found in test_mse
        y = df.loss
        y2 = df_u.loss
    if p == 2:
        y = df.latency
        y2 = df_u.latency

    # randomly split into train and test sets, test set is 80% of data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1
    )

    if unseen == "unseen":
        X_test = X2
        y_test = y2

    clf = DecisionTreeRegressor()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # acc1 = mean_squared_error(y_test, y_pred)
    acc1 = clf.score(X_test, y_test)

    clf2 = RandomForestRegressor(n_estimators=200, n_jobs=-1)
    clf2 = clf2.fit(X_train, y_train)
    y_pred2 = clf2.predict(X_test)
    # acc2= mean_squared_error(y_test, y_pred2)
    acc2 = clf2.score(X_test, y_test)

    clf3 = ExtraTreesRegressor(n_estimators=200, n_jobs=-1)
    clf3 = clf3.fit(X_train, y_train)
    y_pred3 = clf3.predict(X_test)
    # acc3= mean_squared_error(y_test, y_pred3)
    acc3 = clf3.score(X_test, y_test)

    #     pca = PCA(n_components = 2)
    #     X_transformed = pca.fit_transform(X_train)
    #     clf4 = ExtraTreesRegressor(n_estimators=100, n_jobs = -1)
    #     clf4 = clf4.fit(X_transformed, y_train)
    #     newdata_transformed = pca.transform(X_test)
    #     y_pred4 = clf4.predict(newdata_transformed)
    #     #acc4 = mean_squared_error(y_test, y_pred4)
    #     acc4 = clf4.score(X_test, y_test)

    clf_gbc = GradientBoostingRegressor(random_state=0, max_depth=6, n_estimators=200)
    clf_gbc.fit(X_train, y_train)
    y_pred5 = clf_gbc.predict(X_test)
    # acc5 = mean_squared_error(y_test, y_pred5)
    acc5 = clf_gbc.score(X_test, y_test)
    return [acc1, acc2, acc3, acc5]


def test_mse(cond, all_comb1, all_comb2):
    """generates initial model metrics on seen or unseen data"""
    unseen = ""
    if cond == "unseen":
        unseen = "unseen"
    filedir_unseen = os.path.join(
        os.getcwd(), "outputs", unseen + "combined_transform.csv"
    )
    df_unseen = pd.read_csv(filedir_unseen)
    filedir = os.path.join(os.getcwd(), "outputs", "combined_transform.csv")
    df = pd.read_csv(filedir)

    all_comb1 = pd.Series(all_comb1).apply(lambda x: list(x))
    all_comb2 = pd.Series(all_comb2).apply(lambda x: list(x))

    dt = []
    rf = []
    et = []
    # pca = []
    gbc = []
    for i in all_comb1:
        acc_loss = test_feat(cond, df, i, 1, df_unseen)
        dt.append(acc_loss[0])
        rf.append(acc_loss[1])
        et.append(acc_loss[2])
        # pca.append(acc_loss[3])
        gbc.append(acc_loss[3])

    # optimze by adding a flag called losslat to avoid making two dataframes of results
    dt2 = []
    rf2 = []
    et2 = []
    # pca2 = []
    gbc2 = []
    for i in all_comb2:
        # 1 = loss
        # 2 = latency
        acc_latency = test_feat(cond, df, i, 2, df_unseen)
        dt2.append(acc_latency[0])
        rf2.append(acc_latency[1])
        et2.append(acc_latency[2])
        # pca2.append(acc_latency[3])
        gbc2.append(acc_latency[3])

    dict1 = pd.DataFrame({"feat": all_comb1, "dt": dt, "rf": rf, "et": et, "gbc": gbc})
    dict2 = pd.DataFrame(
        {"feat2": all_comb2, "dt2": dt2, "rf2": rf2, "et2": et2, "gbc2": gbc2}
    )

    path = os.path.join(os.getcwd(), "outputs")
    dict1.to_csv(os.path.join(path, unseen + "feat_df1.csv"), index=False)
    dict2.to_csv(os.path.join(path, unseen + "feat_df2.csv"), index=False)


def best_performance(cond):
    """returns the performances of different model architectures on our data."""
    unseen = ""
    if cond == "unseen":
        unseen = "unseen"
    # print("finding best loss performance")
    filedir1 = os.path.join(os.getcwd(), "outputs", unseen + "feat_df1.csv")
    df1 = pd.read_csv(filedir1)
    df1_round = df1.round(decimals=3)
    print("\n")
    # print("Loss Performance sorted from highest to lowest metric: r2", "\n")
    print("Best performance for Loss Models")
    dt_p1 = df1_round.sort_values(by=["dt"], ascending=False)
    print(dt_p1[:2], "\n")
    dt_p2 = df1_round.sort_values(by=["rf"], ascending=False)
    print(dt_p2[:2], "\n")
    dt_p3 = df1_round.sort_values(by=["et"], ascending=False)
    print(dt_p3[:2], "\n")
    dt_p4 = df1_round.sort_values(by=["gbc"], ascending=False)
    print(dt_p4[:2], "\n")

    # print("finding best latency performance")
    filedir2 = os.path.join(os.getcwd(), "outputs", unseen + "feat_df2.csv")
    df2 = pd.read_csv(filedir2)
    df2_round = df2.round(decimals=3)
    # print("Latency Performance sorted from highest to lowest metric: r2", "\n")
    # print(df2_round.sort_values(by=['dt2', 'rf2', 'et2', 'gbc2'], ascending = False)[:5], "\n")
    print("Best performance for Latency Models")
    dt2_p1 = df2_round.sort_values(by=["dt2"], ascending=False)
    print(dt2_p1[:2], "\n")
    dt2_p2 = df2_round.sort_values(by=["rf2"], ascending=False)
    print(dt2_p2[:2], "\n")
    dt2_p3 = df2_round.sort_values(by=["et2"], ascending=False)
    print(dt2_p3[:2], "\n")
    dt2_p4 = df2_round.sort_values(by=["gbc2"], ascending=False)
    print(dt2_p4[:2], "\n")


def getAllCombinations(cond_):
    """Returns all possible combinations of features for training on"""
    lst = [
        "total_bytes",
        "max_bytes",
        "1->2Bytes",
        "2->1Bytes",
        "1->2Pkts",
        "2->1Pkts",
        "total_pkts",
        "number_ms",
        "pkt_ratio",
        "time_spread",
        "pkt sum",
        "longest_seq",
        "total_pkt_sizes",
        "mean_tdelta",
        "max_tdelta",
    ]  # 'proto',
    latency_lst = ["byte_ratio", "pkt_ratio", "time_spread", "total_bytes", "2->1Pkts"]
    loss_lst = [
        "total_pkts",
        "total_pkt_sizes",
        "2->1Bytes",
        "number_ms",
        "mean_tdelta",
        "max_tdelta",
    ]

    if cond_ == 1:
        lst = loss_lst
    if cond_ == 2:
        lst = latency_lst
    uniq_objs = set(lst)
    combinations = []
    for obj in uniq_objs:
        for i in range(0, len(combinations)):
            combinations.append(combinations[i].union([obj]))
        combinations.append(set([obj]))
    print("all combinations generated")
    return combinations


def feat_impt(labl):
    """generates feature importances"""
    label_col = labl

    df = pd.read_csv(os.path.join(os.getcwd(), "outputs", "combined_transform.csv"))

    indexcol = [
        "total_bytes",
        "max_bytes",
        "2->1Bytes",
        "2->1Pkts",
        "total_pkts",
        "total_pkts_min",
        "total_pkts_max",
        "number_ms",
        "pkt_ratio",
        "time_spread",
        "time_spread_min",
        "time_spread_max",
        "pkt sum",
        "longest_seq",
        "longest_seq_min",
        "longest_seq_max",
        "total_pkt_sizes",
        "byte_ratio",
        "mean_tdelta",
        "max_tdelta",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        df[[x for x in indexcol if x in df.columns]], df[label_col]
    )
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    etree = ExtraTreesRegressor(n_estimators=400, n_jobs=4)
    etreeft = etree.fit(X_train, y_train)

    y_pred3 = etree.predict(X_test)
    acc3 = mean_squared_error(y_test, y_pred3)

    print(f"mse: {acc3}, r2: {etree.score(X_test, y_test)}")
    feat_imp = pd.Series(
        index=[x for x in indexcol if x in df.columns], data=etree.feature_importances_
    ).sort_values(ascending=False)
    return feat_imp


def pct_err_correct(labels, preds, threshold=0.1):
    """calculates percentage of "correct" predictions within a given threshold
    for model performance, given a label array and a prediction array.
    Optional threshold parameter specifies what predictions would fall under correct"""
    pct_err = (abs(labels - preds) / labels) < threshold
    return pct_err.sum() / pct_err.size


def emp_loss(df, window=25, upperlimit=20000):
    """
    helper function that returns empirical loss of a dataframe with the provided data over a specified window of time.
    empirical loss is defined as the total count of packet drops divided by the total number of packets.
    -works with dataframe outputs from readfilerun/readfilerunsimple
    -upper limit is set to default at a maximum of 20000 for visualizations
    """
    eloss = (
        df["total_pkts"].rolling(window, min_periods=1).sum()
        / df["event"]
        .str.replace("switch", "")
        .str.split(";")
        .str.len()
        .fillna(0)
        .rolling(window, min_periods=1)
        .sum()
    )
    eloss.replace([np.inf, -np.inf], upperlimit, inplace=True)
    eloss[eloss > upperlimit] = upperlimit
    return eloss.bfill().ffill()


def gen_model(
    label, n_jobs=-1, train_window=20, pca_components=4, test_size=0.005, verbose=True
):
    """generates predictive model and outputs predictions to new column in input df"""
    # other parameters:
    max_depth = 7  # see ext tree regressor
    cv = 5  # see grid search cv
    threshold = 0.15  # pct error margin threshold

    if label == "loss":
        # loss features
        indexcol = ["byte_ratio", "pkt_ratio", "time_spread", "total_bytes", "2->1Pkts"]
    elif label == "latency":
        # latency features
        indexcol = [
            "total_pkts",
            "total_pkt_sizes",
            "2->1Bytes",
            "number_ms",
            "mean_tdelta",
            "max_tdelta",
            "time_spread",
            "longest_seq",
        ]

    ## takes transformed datababy from outputs (should be in gdrive)
    ct = pd.read_csv(f"outputs/combined_transform_{train_window}.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        ct[[x for x in indexcol if x in ct.columns]],
        ct[label],
        test_size=test_size,
        random_state=2020,
    )

    pipe = Pipeline(
        steps=[
            ("reduce_dim", PCA(pca_components)),
            ("clf", ExtraTreesRegressor(max_depth=max_depth, n_jobs=n_jobs)),
        ]
    )
    param_grid = [
        {
            "clf": [ExtraTreesRegressor(max_depth=max_depth, n_jobs=n_jobs)],
            "clf__n_estimators": list(range(80, 180, 30)),
        }
    ]
    mdl = GridSearchCV(pipe, param_grid=param_grid, cv=cv, verbose=False, n_jobs=n_jobs)

    # pipe = Pipeline(steps=[ # K Neighbors Regressor worked well! we didn't use it, though
    #     ('reduce_dim', PCA(4)),
    #     ('clf', KNeighborsRegressor())]) # better for packet loss, apparently
    # param_grid = [{'clf' : [KNeighborsRegressor()],
    #                'clf__n_neighbors' : list(range(1,10))}
    # ]
    # mdl = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)

    mdl = mdl.fit(X_train, y_train)
    if verbose:
        print(f"----------{label} model validation set statistics----------")
        print(
            f"{label} model validation MAPE:",
            mean_absolute_percentage_error(mdl.predict(X_test), y_test),
        )
        print(
            f"{label} model validation Median Absolute Error:",
            median_absolute_error(mdl.predict(X_test), y_test),
        )
        print(
            f"{label} model validation pct of predictions within {threshold} margin:",
            pct_err_correct(mdl.predict(X_test), y_test, threshold=threshold),
        )

        print("R2 Score:", mdl.score(X_test, y_test))

    return mdl


def vis_model(
    df,
    label,
    mdl,
    classify=False,
    loss_thresh=-0.15,
    lat_thresh=0.06,
    window=20,
    emplosswindow=25,
    pct_change_window=2,
    verbose=True,
):
    """generates visualizations and evaluations of model performance
    works in conjunction with gen_model() and run inside performance_metrics()"""

    # other parameters
    smooth_window = 10
    pcterr_thresh = 0.15

    if label == "loss":
        threshold = loss_thresh

        # loss features
        indexcol = ["byte_ratio", "pkt_ratio", "time_spread", "total_bytes", "2->1Pkts"]
    elif label == "latency":
        threshold = lat_thresh

        # latency features
        indexcol = [
            "total_pkts",
            "total_pkt_sizes",
            "2->1Bytes",
            "number_ms",
            "mean_tdelta",
            "max_tdelta",
            "time_spread",
            "longest_seq",
        ]

    # prediction on rolling means
    df[f"pred_{label}"] = mdl.predict(
        df.rolling(window, min_periods=1)[indexcol].mean()
    )  # prediction on mean aggregation

    test_mape = mean_absolute_percentage_error(mdl.predict(df[indexcol]), df[label])
    test_mederr = median_absolute_error(mdl.predict(df[indexcol]), df[label])
    test_pcterr = pct_err_correct(
        mdl.predict(df[indexcol]), df[label], threshold=pcterr_thresh
    )
    if verbose:
        print(
            f"-----{label} model test input dataframe statistics (latency: {list(df['latency'].unique())}, loss: {list(df['loss'].unique())})-----"
        )
        print(f"test dataframe {label} prediction MAPE:", test_mape)
        print(f"test dataframe {label} prediction Median Absolute Error:", test_mederr)
        print(
            f"test dataframe {label} prediction within pct err threshold of {pcterr_thresh}:",
            test_pcterr,
        )

    ## plotting performance visual
    if label == "latency":
        ylabel = "Latency (ms)"
    if label == "loss":
        ylabel = "Loss ratio"

    if classify:
        title_ = f"Real Time Prediction on {label} (latency: {list(df['latency'].unique())}, loss: {list(df['loss'].unique())})"
    else:
        title_ = f"Real Time Anomaly Classification on {label} (latency: {list(df['latency'].unique())}, loss: {list(df['loss'].unique())})"
    fig, ax = plt.subplots(figsize=(12, 5))
    df[[label, f"pred_{label}"]].plot(
        figsize=(12, 5), title=title_, xlabel="Time (sec)", ylabel=ylabel, ax=ax
    )

    # vertical line for legends below
    vertical_line = Line2D(
        [],
        [],
        color="red",
        marker="|",
        linestyle="None",
        alpha=0.45,
        markersize=10,
        markeredgewidth=1.5,
        label="Vertical line",
    )

    if label == "loss":

        def format_func1(value, tick_number):
            # find number of multiples of pi/2
            # N = np.exp(value).astype(int)
            return f"1/{value.astype(int)}"

        ax.yaxis.set_major_formatter(format_func1)

        # empirical loss calculation
        eloss = emp_loss(df, emplosswindow)
        eloss.plot(color="green")
        test_mape_eloss = mean_absolute_percentage_error(
            mdl.predict(df[indexcol]), eloss
        )
        test_pcterr_eloss = pct_err_correct(
            mdl.predict(df[indexcol]), eloss, threshold=pcterr_thresh
        )
        if verbose:
            print("test dataframe MAPE w/ emp loss:", test_mape_eloss)
            print(
                f"test dataframe {label} prediction w/ emp loss within pct err threshold of {pcterr_thresh}:",
                test_pcterr_eloss,
            )

        # plot legend
        lines = [
            Line2D([0], [0], color="tab:blue"),
            Line2D([0], [0], color="tab:orange"),
            Line2D([0], [0], color="tab:green"),
            vertical_line,
        ]
        labels = [
            f"{label} label",
            "Prediction",
            f"Empirical Loss",
            "Percent Change Anomaly",
        ]

    else:
        test_mape_eloss = np.nan
        test_pcterr_eloss = np.nan
        # ax.set_yscale('log')
        ax.ticklabel_format(useOffset=False)
        # latency plot legend
        lines = [
            Line2D([0], [0], color="blue"),
            Line2D([0], [0], color="orange"),
            vertical_line,
        ]
        labels = [f"{label} label", "Prediction", "Percent Change Anomaly"]
    # for i in df[~df["event"].isnull()].index: # adds when packet drops happen as yellow lines
    #     plt.axvline(x=i, color="y", alpha=0.45)

    ## adds new column to df
    if classify:
        plt.legend(lines, labels, loc="upper right")
    else:
        plt.legend(lines[:-1], labels[:-1], loc="upper right")

    df[f"pred_{label}_pctc2_smooth"] = (
        (df[f"pred_{label}"].rolling(smooth_window, min_periods=1).mean())
        .pct_change(pct_change_window)
        .rolling(smooth_window, min_periods=1)
        .mean()
    )

    if classify:
        if label == "loss":
            anomalies = df[f"pred_{label}_pctc2_smooth"] <= threshold
            anomaly_idx = df[anomalies].index - (window // 2)
            for i in anomaly_idx[
                anomaly_idx > 0
            ]:  # indices where *negative* percent change is higher than threshold
                plt.axvline(x=i, color="r", alpha=0.45)
        elif label == "latency":
            anomalies = df[f"pred_{label}_pctc2_smooth"] >= threshold
            anomaly_idx = df[anomalies].index - (window // 2)
            for i in anomaly_idx[
                anomaly_idx > 0
            ]:  # indices where percent change is greater than threshold
                plt.axvline(x=i, color="r", alpha=0.45)
    idx = f"latency_{str(list(df['latency'].unique())).strip('[]').replace(', ', '-')}_loss_{str(list(df['loss'].unique())).strip('[]').replace(', ', '-')}"

    # save predictions to outputs/model
    predictions = os.path.join(
        "data/out/anomaly_detection", f"{label}_anomalies_with_{idx}.csv"
    )
    pd.DataFrame({"Time": df["Time"], "Anomalies": anomalies}).to_csv(
        predictions, index=False
    )  # outputs classifier

    # save plot
    saveto = os.path.join("outputs/model", f"{label}_model_perf_with_{idx}.png")
    plt.savefig(saveto)

    return (
        idx,
        test_mape,
        test_pcterr,
        test_mape_eloss,
        test_pcterr_eloss,
        test_mederr,
    )


def performance_metrics(
    filedir,
    lossmodel,
    latmodel,
    classify=False,
    loss_thresh=-0.15,
    lat_thresh=0.06,
    window=20,
    emplosswindow=25,
    transformed_dir=False,
    verbose=True,
):
    """
    using a file directory of raw dane runs, generates two dataframes of model performance metrics on both loss and latency models in a tuple
    we used this to run vis_model() on every test dane run and generate visualizations
    """
    losslst = []
    latencylst = []
    for i in [x for x in listdir(filedir) if not "losslog" in x]:
        if not transformed_dir:
            mergedtable = readfilerun_simple(
                os.path.join(filedir, i), filedir
            )  # merges losslogs into one table
            df_ = genfeat(
                mergedtable
            )  # generates all the adjacent features we train on!
        else:
            mergedtable = pd.read_csv(
                os.path.join(filedir, i)
            )  # merges losslogs into one table
            df_ = genfeat(
                mergedtable
            )  # generates all the adjacent features we train on!
        losslst.append(
            vis_model(
                df_,
                "loss",
                lossmodel,
                classify,
                loss_thresh,
                lat_thresh,
                window,
                emplosswindow,
                verbose=verbose,
            )
        )

        latencylst.append(
            vis_model(
                df_,
                "latency",
                latmodel,
                classify,
                loss_thresh,
                lat_thresh,
                window,
                emplosswindow,
                verbose=verbose,
            )
        )

    metrics = [
        "idx",
        "test_mape",
        "test_pcterr",
        "test_mape_eloss",
        "test_pcterr_eloss",
        "test_mederr",
    ]
    lossperf = pd.DataFrame(losslst, columns=metrics).set_index("idx")
    latperf = pd.DataFrame(latencylst, columns=metrics).set_index("idx")
    print("----------loss model test set statistics----------")
    print(lossperf.mean())
    print("----------latency model test set statistics----------")
    print(latperf.mean())
    return (lossperf, latperf)
