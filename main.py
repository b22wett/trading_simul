# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import scipy
import sklearn
import matplotlib.pyplot as plt

# %%
"""
AUDITING
"""

# %%
df = pd.read_csv("stocks_daily.csv")
print(df.duplicated().sum())
df = df.drop_duplicates()

# %%
print(df.describe())
print(df.info())
"""
close, news_count missing values.
Need to:
x Convert date to date
x Make ticker uppercase
x Make sector lowercase
x Examine vol_state (Ordinal encoding fine)
- Fill values in close and news_count
"""

# %%
weekdays = (pd.to_datetime(df.date[df.date.apply(lambda x: "." in x)], format="%d.%m.%Y").dt.day_name())
# df.date = pd.to_datetime(df.date,format="mixed") # Mix of ISO and dotted dates
print(weekdays.apply(lambda x: x in ["Saturday", "Sunday"]).sum())
df.date = pd.to_datetime(df.date, format="mixed", dayfirst=True)

# %%
print(df.date.apply(lambda x: x in ["Saturday", "Sunday"]).sum()) # Done!

# %%
df.ticker = df.ticker.str.upper()
df.ticker = df.ticker.str.strip()
df.sector = df.sector.str.lower()
df.sector = df.sector.str.strip()
df = df.sort_values(["date", "sector", "ticker"])
df = df.reset_index(drop=True)

# %%
print(df.sector.unique())
print(df.ticker.unique())

# %% [raw]
#

# %%
print(df.vol_state.unique())

# %%
df["bad"] = df.close.isna()

# %%
df["bad"] = df["bad"].where(df["volume"] > 0, True)

# %%
g = df.groupby("ticker")
df["lr1"] = g["close"].transform(lambda x: np.log(x).diff())
print((df["lr1"] >= 0.4).sum())
df[~df.bad]["lr1"].plot()
df["bad"] = df["bad"].where(df["lr1"].apply(lambda x: abs(x) < 0.4), True)
df["lrd"] = df["close"].apply(np.log) - df["open"].apply(np.log)
g = df.groupby("ticker")
#df["dvol"] = g["lrd"].transform(lambda x: x.expanding().std())
print(df.lr1, df.lrd, df.realised_vol)

# %%
gg = df[~df.bad].groupby("ticker")
gg["realised_vol"].plot()
plt.show()
# Much too high at certain values -- bad prints.
print((df[~df.bad]["realised_vol"] > 0.1).sum())
print((df[~df.bad]["realised_vol"] == 0).sum())

# %%
gg = df[~df.bad].groupby("ticker")
df["vol_zscore"] = ((df["realised_vol"] - gg["realised_vol"].transform("mean")) / gg["realised_vol"].transform("std"))
gg = df[~df.bad].groupby("ticker")
for g in gg["vol_zscore"]:
    g[1].plot()
    plt.show()
# Uncertain whether we can exclude any of these -- clearly not Gaussian distribution (asymmetric), jumps are sort-of regular.
# Rerun after excluding bad prints in block below.

# %%
df["bad"] = df["bad"].where(df["realised_vol"].apply(lambda x: abs(x) < 0.1 or x == 0), True) # 1000 observations per stock means
gg = df[~df.bad].groupby("ticker")
df["var_ohlc"] = df.apply(lambda x: 0.5*(np.log(x.high) - np.log(x.low))**2 - (2*np.log(2) - 1)*(np.log(x.close) - np.log(x.open))**2,axis=1)
# Using Garman and Klass (1980) from https://assets.bbhub.io/professional/sites/10/intraday_volatility-3.pdf.
# We do not necessarily trust the realised volatility from our source -- however, as this is the target, we will still predict on it.

# %%
df["target"] = df.groupby("ticker").shift(-1)["realised_vol"] # Took much longer than I thought it would.
df["target"].plot()

# %%
print(df.bad.sum(), df["target"].apply(lambda x: x < 0.1))
df["bad"] = df["bad"].where(df["target"].apply(lambda x: abs(x) < 0.1 or x == 0), True) # Peaks from visual inspection above.
gg = df[~df.bad].groupby("ticker")

# %%
df["realised_var"] = df.realised_vol.map(lambda x: x**2)

# %%
df["r1_sqr"] = gg["close"].pct_change(1).map(lambda x: x**2)
print(df.r1_sqr)

# %%
# We use the EWM model from RiskMetrics(TM) — Technical Document (JPMorgan/Reuters 1996)
reference = sklearn.dummy.DummyRegressor(strategy="mean")
#print(df[["target", "realised_vol"]])
def ewm_pred(lam, gg):
    r1_squared = np.asarray(gg["r1_sqr"])
    var = np.asarray(gg["realised_var"])
    #print(r1_squared, var, gg.target)
    return np.sqrt(r1_squared * (1 - lam) + var * lam)
    
def pred_err(lam, te):
    gg = df[~df.bad].loc[te]
    #print(ewm_pred(lam, gg) - gg.target.apply(lambda x: x))
    #print(gg.target)
    return ((ewm_pred(lam, gg) - np.asarray(gg.target))**2).sum()


# %%
ind = df[~df.bad].index
print(ind)
num = len(ind)
ind_train = ind[:(num * 4) // 5]
ind_test = ind[-num//5:]
train = df[~df.bad]

# %%
flag = False
FEATURES = ["target", "realised_vol", "r1_sqr", "vol_state"]
lams = []
old_model = []
new_model = []
for tr in np.array_split(ind_train, 185):
    lam = scipy.optimize.minimize(pred_err, 0.94, tr, bounds=[(0, 1)]).x[0]
    lams.append(lam)
    old_model.extend(tr) if lam > 0.9 else new_model.extend(tr)

lams = pd.Series(lams)
#print(lams.mean(), lams.std())
print(lams.describe())
print(lams.mean(), lams.sem())
lams.plot()
plt.show()
lambdavals = [0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
plt.plot(lambdavals, [pred_err(l, ind_test) for l in lambdavals])
plt.show()

# %%
new_regime = train.loc[new_model]
old_regime = train.loc[old_model]

# %%
EXAMINE = ["volume", "lr1", "news_count", "earnings_day"]
print(new_regime[EXAMINE].mean(axis=0), new_regime[EXAMINE].sem(axis=0))
print(old_regime[EXAMINE].mean(axis=0), old_regime[EXAMINE].sem(axis=0))

# %%
df = df[~df.bad]
df["lrv"] = np.log(df.realised_vol.where(df.realised_vol > 0))     # after masking bad values
g = df.groupby("ticker")                                            # sorted by ticker, date
df["lrv5"] = g["lrv"].transform(lambda s: s.rolling(5, min_periods=3).mean())
df["lrv22"] = g["lrv"].transform(lambda s: s.rolling(22, min_periods=10).mean())
df["mkt_lrv"] = df.groupby("date")["lrv"].transform("mean")          # market-wide vol today
df["earn_tomorrow"] = g["earnings_day"].shift(-1)
df["y"] = g["lrv"].shift(-1)                                        # target on the log scale

#df.bad = df.bad.where(np.logical_not(df.earnings_day), True)

X = ["lrv", "lrv5", "lrv22", "mkt_lrv", "earn_tomorrow"] # drop earn_tomorrow if we drop earnings_day too.
d = df.dropna(subset=X + ["y"])
rmses = []
baselines = []
CUTOFFS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for i, cuttime in enumerate(CUTOFFS):
    if cuttime == 1:
        continue
    cut = d.date.quantile(cuttime)
    nxtcut = d.date.quantile(CUTOFFS[i + 1])
    tr, te = d[d.date <= cut - pd.Timedelta("5 days")], d[np.logical_and(cut < d.date, d.date < nxtcut)]
    print(cut, nxtcut)
    lr = sklearn.linear_model.LinearRegression().fit(tr[X], tr["y"])
    rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))
    print("persistence", rmse(te["lrv"], te["y"]), "model", rmse(lr.predict(te[X]), te["y"]))
    rmses.append(rmse(lr.predict(te[X]), te["y"]))
    baselines.append(rmse(te["lrv"], te["y"]))

# %%
DATES = [d.date.quantile(cuttime) for cuttime in CUTOFFS[:-1]]

plt.bar(DATES, baselines, label="Baseline model error", width=-100, align="edge")
plt.bar(DATES, rmses, label="Linear model error", width=-100, align="edge")
plt.legend()
plt.xlabel("Dates considered")
plt.xlim(d.date.iloc[0], d.date.iloc[-1])
plt.ylabel("Root-mean-squared error")
plt.title("Comparison of errors for two models")
plt.savefig("err_comp.svg")

# %%
