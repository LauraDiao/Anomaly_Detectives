import numpy as np
import pandas as pd
from os import listdir


import warnings

warnings.filterwarnings("ignore")

# Data cleaning to turn columns into list of integers
def return_int(x):
    return [int(i) for i in x.split(";")[:-1]]


def longest_seq(aList):
    "find longest sequence in packet dirs"
    maxCount = 1
    actualCount = 1
    for i in range(len(aList) - 1):
        if aList[i] == aList[i + 1]:
            actualCount += 1
        else:
            actualCount = 1
        if actualCount > maxCount:
            maxCount = actualCount
    return maxCount


def better_agg(t_df, interval=10):
    """similar to agg10, takes dataframe with features from output of genfeat function and aggregates them in 10 second intervals"""
    temp = t_df.iloc[20:-3].copy()
    df = temp.groupby(temp.index // interval).agg(
        {
            "total_bytes": [pd.Series.mean],
            # 'max_bytes': [np.mean, np.std], # takes way too long to calculate
            "Proto": [pd.Series.mode],
            "1->2Bytes": [np.mean, np.var],
            "2->1Bytes": [np.mean, np.var],
            "1->2Pkts": [np.mean, np.var],
            "2->1Pkts": [np.mean, np.var],
            "total_pkts": [np.mean, np.min, np.max],
            "number_ms": [np.mean, np.var],
            "pkt_ratio": [np.mean, np.var],
            "time_spread": [np.mean, np.min, np.max],
            "pkt sum": [np.mean, np.var],
            "longest_seq": [np.mean, np.min, np.max],
            "total_pkt_sizes": [np.mean, np.var],
            "byte_ratio": [np.mean, np.var],
            "mean_tdelta": [np.mean, np.min, np.max, np.var],
            "max_tdelta": [np.mean, np.min, np.max],
            "latency": [pd.Series.mode],
            "loss": [pd.Series.mode],
            "later_latency": [pd.Series.mode],
            "later_loss": [pd.Series.mode],
        }
    )
    df.columns = [
        "_".join(a) for a in df.columns.to_flat_index()
    ]  # flattens MultiIndex
    df.columns = [
        a[:-5] if ("mode" in a[-5:]) or ("mean" in a[-5:]) else a for a in df.columns
    ]  # simplifies names of certain features
    return df


def agg10(t_df):
    """takes dataframe with features from output of genfeat function and aggregates them in 10 second intervals"""
    # print(t_df.columns)
    indexcol = [
        "total_bytes",
        "max_bytes",
        "proto",
        "1->2Bytes",
        "2->1Bytes",
        "1->2Pkts",
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
        "loss",
        "latency",
    ]
    df = pd.DataFrame(
        [
            t_df[:10]["total_bytes"].mean(),
            #    t_df[:10]['max_bytes'].std(),
            t_df[:10]["Proto"].value_counts().idxmax(),  # most frequent protocol
            t_df[:10]["1->2Bytes"].mean(),
            t_df[:10]["2->1Bytes"].mean(),
            t_df[:10]["1->2Pkts"].mean(),
            t_df[:10]["2->1Pkts"].mean(),
            t_df[:10]["total_pkts"].mean(),
            t_df[:10]["total_pkts"].min(),
            t_df[:10]["total_pkts"].max(),
            t_df[:10]["number_ms"].mean(),
            t_df[:10]["pkt_ratio"].mean(),
            t_df[:10]["time_spread"].mean(),
            t_df[:10]["time_spread"].min(),
            t_df[:10]["time_spread"].max(),
            t_df[:10]["pkt sum"].mean(),
            t_df[:10]["longest_seq"].mean(),
            t_df[:10]["longest_seq"].min(),
            t_df[:10]["longest_seq"].max(),
            t_df[:10]["total_pkt_sizes"].mean(),
            t_df["byte_ratio"].mean(),
            t_df[:10]["mean_tdelta"].mean(),
            t_df[:10]["max_tdelta"].mean(),
            t_df.loss.unique()[0],
            t_df.latency.unique()[0],
        ],
        index=indexcol,
    ).T

    for i in range(20, t_df.shape[0], 10):
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        t_df[i - 10 : i]["total_bytes"].mean(),
                        #    t_df[i-10:i]['max_bytes'].std(),
                        t_df[i - 10 : i]["Proto"]
                        .value_counts()
                        .idxmax(),  # most frequent protocol
                        t_df[i - 10 : i]["1->2Bytes"].mean(),
                        t_df[i - 10 : i]["2->1Bytes"].mean(),
                        t_df[i - 10 : i]["1->2Pkts"].mean(),
                        t_df[i - 10 : i]["2->1Pkts"].mean(),
                        t_df[i - 10 : i]["total_pkts"].mean(),
                        t_df[i - 10 : i]["total_pkts"].min(),
                        t_df[i - 10 : i]["total_pkts"].max(),
                        t_df[i - 10 : i]["number_ms"].mean(),
                        t_df[i - 10 : i]["pkt_ratio"].mean(),
                        t_df[i - 10 : i]["time_spread"].mean(),
                        t_df[i - 10 : i]["time_spread"].min(),
                        t_df[i - 10 : i]["time_spread"].max(),
                        t_df[i - 10 : i]["pkt sum"].mean(),
                        t_df[i - 10 : i]["longest_seq"].mean(),
                        t_df[i - 10 : i]["longest_seq"].min(),
                        t_df[i - 10 : i]["longest_seq"].max(),
                        t_df[i - 10 : i]["total_pkt_sizes"].mean(),
                        t_df["byte_ratio"].mean(),
                        t_df[i - 10 : i]["mean_tdelta"].mean(),
                        t_df[i - 10 : i]["max_tdelta"].mean(),
                        t_df.loss.unique()[0],
                        t_df.latency.unique()[0],
                    ],
                    index=indexcol,
                ).T,
            ],
            ignore_index=True,
        )
    return df


def onehot_(df):
    d_proto = pd.get_dummies(df["Proto"]).rename(columns={6: "TCP", 58: "IPv6"})
    d_p1 = pd.get_dummies(df["Port1"])
    d_p2 = pd.get_dummies(df["Port2"])
    temp_df = pd.concat([df, d_proto, d_p1, d_p2], axis=1)
    return temp_df


def time__(dataframe):
    # scales seconds
    mini = dataframe["Time"].min()
    temp1 = dataframe.assign(Second=lambda x: x["Time"] - mini)

    return temp1


def main2(temp_df):
    """helper function that aggregates the input by count and sum for plot_main_4"""
    transformed = temp_df  # time(temp_df)
    label = "loss"

    # print(transformed.columns)
    s = ["Second", label]

    p_sum_agg = transformed.groupby(s)["total_pkts"].agg(["count", "sum"]).reset_index()
    # print(p_sum_agg.columns)

    b_agg = transformed.groupby(s)["total_bytes"].agg(["count", "sum"]).reset_index()

    #     p_agg =  transformed.groupby(s)['total_pkts'].agg(['count', 'sum']).reset_index()
    # print("all aggregations made")
    return [p_sum_agg, b_agg]  # , p_agg]


def genfeat(df):
    tdf = df
    tdf["total_bytes"] = tdf["1->2Bytes"] + tdf["2->1Bytes"]  # combining bytes
    tdf["total_pkts"] = tdf["1->2Pkts"] + tdf["2->1Pkts"]  # combining packets
    tdf["packet_sizes"] = (
        tdf["packet_sizes"].astype("str").apply(return_int)
    )  # converting list of packet sizes to type int
    tdf["pkt sum"] = tdf["packet_sizes"].apply(lambda x: sum(x))  # summing packets
    tdf["packet_dirs"] = (
        tdf["packet_dirs"].astype("str").apply(return_int)
    )  # converting to type int
    tdf["longest_seq"] = tdf["packet_dirs"].apply(
        longest_seq
    )  # finding longest sequence

    tdf["mean_tdelta"] = tdf["packet_times"].str.split(";").apply(mean_diff)
    tdf["max_tdelta"] = tdf["packet_times"].str.split(";").apply(max_diff)

    tdf["packet_times"] = tdf["packet_times"].apply(
        return_int
    )  # converting to int list

    tdf["number_ms"] = tdf["packet_times"].apply(lambda x: pd.Series(x).nunique())
    # tdf['max_bytes'] = tdf.apply(lambda x: max_bytes(x['packet_times'],x['packet_sizes']),axis=1)
    tdf["total_pkt_sizes"] = tdf.packet_sizes.apply(lambda x: sum(x))
    tdf["pkt_ratio"] = tdf.total_pkt_sizes / tdf.total_pkts
    tdf["time_spread"] = tdf.packet_times.apply(lambda x: x[-1] - x[0])
    tdf["byte_ratio"] = tdf.total_bytes / tdf.total_pkts
    # tdf['mean_tdelta'] = tdf['packet_times'].apply(lambda x: mean_diff(str(x).split(';')))
    # tdf['max_tdelta'] = tdf['packet_times'].apply(lambda x: max_diff(str(x).split(';')))

    # print("max bytes generated")
    return tdf


def mean_diff(lst):
    """
    returns mean difference in a column,
    meant to be used on transformed 'packet_times' column
    >>> df['packet_times'].str.split(';').apply(mean_diff)
    """
    lst = np.array(list(filter(None, lst)))  # takes out empty strings
    mn = np.mean(
        [int(t) - int(s) for s, t in zip(lst, lst[1:])]
    )  # TODO use numpy diff if needed
    return 0 if np.isnan(mn) else mn


def max_diff(lst):
    """
    returns max difference in a column,
    meant to be used on transformed 'packet_times' column
    >>> df['packet_times'].str.split(';').apply(max_diff)
    """
    lst = np.array(list(filter(None, lst))).astype(np.int64)
    # mn = max([int(t) - int(s) for s, t in zip(lst, lst[1:])]) if len(lst) > 0 else np.nan
    diffs = np.diff(lst)
    mn = max(diffs) if len(diffs) > 0 else np.nan  # length of diffs might be zero
    return 0 if np.isnan(mn) else mn


def max_bytes(x, y):
    maxbytes = pd.DataFrame([x, y]).T.groupby(0).sum().values.max()
    return maxbytes


def accuracy(a, b):
    """Function to return accuracy based on 10% difference"""
    scale = a * 0.1
    if b <= a + scale and b >= a - scale:
        return 1
    else:
        return 0
