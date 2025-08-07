import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
import statsmodels.api as sm
import statsmodels.stats.diagnostic as dg
import matplotlib.pyplot as plt
from collections import Counter


def load_diff_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df = df.drop_duplicates(subset=["nr"], keep="last")
    df["time"] = df["time"].str[:19]
    df["opost"] = df["opost"].str[:19]
    df["time1"] = pd.to_datetime(df["time"])
    df["time3"] = pd.to_datetime(df["opost"])
    df["timediff2"] = df["time1"] - df["time3"]
    return df


def load_cascade_data(path: Path, time_diff_dict: dict) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', encoding="utf-8")
    df["community"] = df["community"].replace({
        "liberal": "radicalleft",
        "left": "liberalleft",
        "news": "general"
    })
    df = df[df["community"] != "999"]
    df["duration"] = df["contents"].map(lambda sub: time_diff_dict.get(sub, [pd.Timedelta(seconds=0)])[0])
    df["duration"] = df["duration"].dt.total_seconds()
    return df[["max-breadth", "depth", "community", "size", "duration"]]


def plot_community_durations(cascade_df: pd.DataFrame):
    communities = cascade_df["community"].unique()
    fig, ax = plt.subplots(figsize=(20, 8))

    colors = plt.cm.get_cmap("tab10", len(communities))
    for i, comm in enumerate(communities):
        x = cascade_df[cascade_df["community"] == comm]["depth"].to_numpy()
        y = cascade_df[cascade_df["community"] == comm]["duration"].to_numpy()
        ax.plot(x, y, "o", label=comm, color=colors(i))
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m * x + b, color=colors(i))

    ax.legend()
    ax.set_xlabel("Cascade Depth")
    ax.set_ylabel("Duration (seconds)")
    ax.set_title("Cascade Depth vs. Duration by Community")
    plt.show()


def prepare_time_series(df: pd.DataFrame) -> pd.DataFrame:
    df["time"] = df["time"].str[:10]
    df = df[~df["S_modularity"].isin(["999", "7", "8"])]
    df = df[~df["T_modularity"].isin(["999", "7", "8"])]
    df["commlabel"] = df["label"]
    df["commlabel_y"] = df["label"] + "_y"

    outgroup = df[df["samegroup"] == 0]
    ingroup = df[df["samegroup"] == 1]

    out_ts = pd.crosstab(outgroup["time"], outgroup["commlabel_y"])
    in_ts = pd.crosstab(ingroup["time"], ingroup["commlabel"])
    ts = pd.concat([in_ts, out_ts], axis=1).fillna(0).sort_index()
    return ts


def check_stationarity_kpss(ts: pd.DataFrame) -> list:
    def test(series):
        try:
            stat, p, *_ = kpss(series, nlags="auto")
            return int(p < 0.05)
        except:
            return 1

    return [test(ts[col].dropna()) for col in ts.columns]


def difference_nonstationary(ts: pd.DataFrame, stationarity_flags: list) -> pd.DataFrame:
    for i, stationary in enumerate(stationarity_flags):
        if stationary:
            ts.iloc[:, i] = ts.iloc[:, i] - ts.iloc[:, i].shift(1)
    return ts.dropna()


def run_granger(ts: pd.DataFrame, max_lag: int = 1) -> pd.DataFrame:
    labels = ts.columns
    target_labels = [l for l in labels if l.endswith("_y")]
    base_labels = [l[:-2] for l in target_labels]

    results = {
        "commlabel": [],
        "bg": [],
        "f": [],
        "f_p": [],
        "sig": [],
        "x_size": [],
        "y_size": []
    }

    counter_y = Counter(ts.columns[ts.columns.str.endswith("_y")])
    counter_x = Counter(ts.columns[~ts.columns.str.endswith("_y")])

    for base in base_labels:
        x = sm.add_constant(ts[base])
        y = ts[base + "_y"]
        model = sm.OLS(y, x).fit()

        bg_stats = [dg.acorr_breusch_godfrey(model, nlags=i)[3] for i in range(1, max_lag + 1)]
        best_lag = np.argmin(bg_stats) + 1

        res = grangercausalitytests(ts[[base + "_y", base]], [best_lag], verbose=False)
        f_stat = res[best_lag][0]["ssr_ftest"][0]
        p_val = res[best_lag][0]["ssr_ftest"][1]

        if p_val > 0.05:
            sig = "-"
        elif p_val > 0.01:
            sig = "*"
        elif p_val > 0.001:
            sig = "**"
        else:
            sig = "***"

        results["commlabel"].append(base)
        results["bg"].append(best_lag)
        results["f"].append(f_stat)
        results["f_p"].append(p_val)
        results["sig"].append(sig)
        results["x_size"].append(counter_x.get(base, 0))
        results["y_size"].append(counter_y.get(base + "_y", 0))

    return pd.DataFrame(results)


def main():
    base_path = Path("/Users/xixuan/Desktop/twitter_test/fff_api_alltweets")
    df_diff = load_diff_data(base_path / "diff_gephi_sto.csv")
    time_diff_dict = df_diff.groupby("ref")["timediff2"].apply(list).to_dict()

    cascade_df = load_cascade_data(base_path / "diff_cascades_stat.csv", time_diff_dict)
    plot_community_durations(cascade_df)

    ts = prepare_time_series(df_diff)
    stationary_flags = check_stationarity_kpss(ts)
    ts = difference_nonstationary(ts, stationary_flags)

    results = run_granger(ts, max_lag=1)
    results.to_csv(base_path / "fvalue-rtbridge1-labelfi.csv", index=False)
    print(results)


if __name__ == "__main__":
    main()
