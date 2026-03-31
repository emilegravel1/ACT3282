

import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import seaborn
import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.rcParams["figure.figsize"] = (20, 8)

tsx60 = [
    "AEM.TO","ATD.TO","BMO.TO","BNS.TO","ABX.TO","BCE.TO",
    "BAM.TO","BN.TO","BIP-UN.TO","CAE.TO","CCO.TO","CM.TO",
    "CNR.TO","CNQ.TO","CP.TO","CTC-A.TO","CCL-B.TO","CLS.TO",
    "CVE.TO","GIB-A.TO","CSU.TO","DOL.TO","EMA.TO","ENB.TO",
    "FFH.TO","FM.TO","FSV.TO","FTS.TO","FNV.TO","WN.TO",
    "GIL.TO","H.TO","IMO.TO","IFC.TO","K.TO","L.TO",
    "MG.TO","MFC.TO","MRU.TO","NA.TO","NTR.TO","OTEX.TO",
    "PPL.TO","POW.TO","QSR.TO","RCI-B.TO","RY.TO","SAP.TO",
    "SHOP.TO","SLF.TO","SU.TO","TRP.TO","TECK-B.TO","T.TO",
    "TRI.TO","TD.TO","TOU.TO","WCN.TO","WPM.TO","WSP.TO"
]

data = yf.download(
    tsx60,
    start="2025-10-27",
    end="2026-03-27",
    auto_adjust=False,
    progress=False
)

if "Adj Close" in data.columns.get_level_values(0):
    prices_df = data["Adj Close"].copy()
else:
    prices_df = data["Close"].copy()

print(prices_df.head())
print(prices_df.shape)

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

scores, pvalues, pairs = find_cointegrated_pairs(prices_df)

labels = prices_df.columns.tolist()

seaborn.heatmap(pvalues, xticklabels=labels, yticklabels=labels, cmap='RdYlGn_r'
                , mask = (pvalues >= 0.05)
                )
plt.show()
print(pairs)

pairs_with_pvalues = []

for i in range(len(pvalues)):
    for j in range(len(pvalues)):
        if i < j:
            pval = pvalues[i, j]
            if pval < 0.05:
                pairs_with_pvalues.append(
                    (prices_df.columns[i], prices_df.columns[j], pval)
                )

pairs_sorted = sorted(pairs_with_pvalues, key=lambda x: x[2])

print("Top 15 pairs:")
for stock1, stock2, pval in pairs_sorted[0:15]:
    try:
        sector1 = yf.Ticker(stock1).info.get("sector")
        sector2 = yf.Ticker(stock2).info.get("sector")
    except Exception:
        sector1 = None
        sector2 = None

    print(stock1,"-", stock2,", p-value =", pval,", sectors are" ,sector1, "and",sector2)

# La paire CLS et OTEX semble cointégrée et les deux actions sont dans
# le secteur technologique

S1 = prices_df['CLS.TO']
S2 = prices_df['OTEX.TO']

X = sm.add_constant(S1)
results = sm.OLS(S2, X).fit()
b = results.params['CLS.TO']

spread = S2 - b * S1

spread_mavg1 = spread.rolling(window=1).mean()
spread_mavg30 = spread.rolling(window=30).mean()
std_30 = spread.rolling(window=30).std()
zscore_30_1 = (spread_mavg1 - spread_mavg30) / std_30

plt.figure(figsize = (20, 7.8))
plt.plot(zscore_30_1.index, zscore_30_1.values)
plt.axhline(0, linestyle='--')
plt.axhline(1, linestyle='--')
plt.axhline(-1, linestyle='--')

plt.title(f"{'CLS.TO'} vs {'OTEX.TO'}\np={pval:.4g}")
plt.show()