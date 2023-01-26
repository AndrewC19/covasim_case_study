#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:23:22 2023

@author: michael
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

pd.set_option("display.float_format", lambda x: "%.5f" % x)

df = pd.read_csv("results/complete_data/passively_observed.csv", index_col=0)
# df = df.where(df["location"] == "france").dropna()


means = df.where(df["location"] == "france").dropna()[[c for c in df.columns if "total" in c or "avg_" in c]].mean()

covariates = {
    "total_contacts_h": means["total_contacts_h"],
    "total_contacts_s": means["total_contacts_s"],
    "total_contacts_w": means['total_contacts_w'],
    "avg_rel_sus": means['avg_rel_sus'],
}
Xc = {"beta": 0.016}
Xt = {"beta": 0.02672}

X = pd.DataFrame([covariates | Xc, covariates | Xt])


for frame in [df, X]:
    frame["log(beta)"] = np.log(frame["beta"])
    frame["log(avg_rel_sus)"] = np.log(frame["avg_rel_sus"])
    frame["log(total_contacts_h)"] = np.log(frame["total_contacts_h"])
    frame["log(total_contacts_s)"] = np.log(frame["total_contacts_s"])
    frame["log(total_contacts_w)"] = np.log(frame["total_contacts_w"])
    # frame.drop(
    #     ["total_contacts_h", "total_contacts_s", "total_contacts_w", "avg_rel_sus"],
    #     axis=1,
    #     inplace=True,
    # )

model = sm.OLS(df["cum_infections"], df[list(X.columns)]).fit()
Y = model.predict(X)
print(model.params)

print(f"Y\n{Y}")
print("ATE", Y.iloc[1] - Y.iloc[0])
