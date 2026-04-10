import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import seaborn
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf

plt.rcParams["figure.figsize"] = (20, 7.8)

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

_,pval,_=coint(S1, S2)

plt.title(f"{'CLS.TO'} vs {'OTEX.TO'}\np={pval:.4g}")
plt.show()

###################################### 3 #######################################

signal = pd.Series(index=zscore_30_1.index, dtype=float)
signal[zscore_30_1 < -1] = 1
signal[zscore_30_1 > 1] = -1
signal[abs(zscore_30_1) < 0.2] = 0
signal = signal.ffill().fillna(0)

spread_change = spread.diff()

position = signal.shift(1).fillna(0)

pnl = position * spread_change

cumulative_pnl = pnl.cumsum()

plt.plot(cumulative_pnl)
plt.title("Pairs Trading PnL")
plt.show()

###################################### 4 #######################################

data_test = yf.download(
    tsx60,
    start="2024-10-27",
    end="2025-03-27",
    auto_adjust=False,
    progress=False
)

if "Adj Close" in data_test.columns.get_level_values(0):
    prices_df_test = data_test["Adj Close"].copy()
else:
    prices_df_test = data_test["Close"].copy()

S1_test = prices_df_test['CLS.TO']
S2_test = prices_df_test['OTEX.TO']

spread_test = S2_test - b * S1_test

mavg30 = spread_test.rolling(30).mean()
std30 = spread_test.rolling(30).std()

zscore = (spread_test - mavg30) / std30

signal = pd.Series(index=zscore.index, dtype=float)

signal[zscore < -1] = 1
signal[zscore > 1] = -1
signal[abs(zscore) < 0.2] = 0

signal = signal.ffill().fillna(0)

position = signal.shift(1).fillna(0)

spread_test_change = spread_test.diff()

pnl = position * spread_test_change

cumulative_pnl = pnl.cumsum()

plt.plot(cumulative_pnl)
plt.title("Pairs Trading PnL")
plt.show()

###################################### 5 #######################################

returns = np.log(prices_df / prices_df.shift(1)).dropna()
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)
pca = PCA()
pca_components = pca.fit_transform(returns_scaled)
explained_variance = pca.explained_variance_ratio_

print("Explained Variance by Each Component:")
for i in range(min(10, len(explained_variance))):
    print(f"PC{i+1}: {explained_variance[i]:.4f} ({explained_variance[i]*100:.2f}%)")

cumsum_var = np.cumsum(explained_variance)

fig, ax = plt.subplots(figsize=(20, 7.8))

ax.bar(range(1, len(explained_variance)+1), explained_variance,
       alpha=0.6, label='Individual Variance', color='steelblue')
ax.plot(range(1, len(cumsum_var)+1), cumsum_var, 'ro-',
        linewidth=2, label='Cumulative Variance', markersize=6)
ax.axhline(y=0.9, color='green', linestyle='--', label='90% Threshold')
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Explained Variance Ratio', fontsize=12)
ax.set_title('PCA: Variance Explained by Each Component', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 20)

plt.tight_layout()
plt.show()

n_components_60 = np.argmax(cumsum_var >= 0.6) + 1
print(f"Number of components needed to explain 60% variance: {n_components_60}")

###################################### 6 #######################################

loadings = pca.components_.T

K = 8

pca_components_K = pca_components[:, :K]

loadings_K = loadings[:, :K]

reconstructed_scaled = pca_components_K @ loadings_K.T

reconstructed_returns = reconstructed_scaled * scaler.scale_ + scaler.mean_

for stock_name in ['CLS.TO', 'OTEX.TO']:
    stock_idx = returns.columns.get_loc(stock_name)

    actual_returns = returns.iloc[:, stock_idx].values
    predicted_returns = reconstructed_returns[:, stock_idx]
    errors = actual_returns - predicted_returns

    print(f"Analyzing: {stock_name}")
    print(f"\nActual returns - shape: {actual_returns.shape}")
    print(f"Predicted returns - shape: {predicted_returns.shape}")
    print(f"Prediction errors - shape: {errors.shape}")
    print(f"\nError statistics:")
    print(f"  Mean error: {errors.mean():.6f}")
    print(f"  Std error: {errors.std():.6f}")
    print(f"  Min error: {errors.min():.6f}")
    print(f"  Max error: {errors.max():.6f}")

    fig, axes = plt.subplots(2, 1, figsize=(20, 7.8))

    axes[0].plot(actual_returns, linewidth=1.5, color='steelblue', label='Actual')
    axes[0].plot(predicted_returns, linewidth=1.5, color='orange', label='Predicted (8 PCs)', alpha=0.8)
    axes[0].set_ylabel('Return', fontsize=11)
    axes[0].set_title(f'{stock_name}: Actual vs Predicted Returns', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(errors, bins=30, color='darkred', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Error Value', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f'{stock_name}: Distribution of Prediction Errors', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

###################################### 7 #######################################

S1 = prices_df['CLS.TO']
S2 = prices_df['OTEX.TO']

X = sm.add_constant(S1)
results = sm.OLS(S2, X).fit()
b = results.params['CLS.TO']

residuals = results.resid.dropna()

plt.figure(figsize=(20,7.8))
plt.plot(residuals)
plt.axhline(0, color='black', linestyle='--')
plt.title("Résidus du spread")
plt.show()

plot_acf(residuals, lags=20)
plt.show()

print("\n" + "="*70)
print("AUTO-ARIMA: Automated Model Selection")
print("="*70)

auto_model = auto_arima(
    residuals,
    d=0,
    max_p=5,
    max_q=5,
    ic='aic',
    trace=True,
    suppress_warnings=True
)

print(f"\nBest model selected: ARIMA{auto_model.order}")
print(f"AIC: {auto_model.aic():.4f}")

residuals.index = pd.to_datetime(residuals.index)
residuals = residuals.asfreq('B')

model = ARIMA(residuals, order=auto_model.order).fit()

###################################### 8 #######################################

n_forecast = 5

forecast_result = model.get_forecast(steps=n_forecast)
forecast_values = forecast_result.predicted_mean

print(f"\nForecasted errors for next {n_forecast} periods:")
print(forecast_values)

fig, ax = plt.subplots(figsize=(20, 7.8))

ax.plot(range(len(residuals)), residuals, linewidth=2, color='steelblue', label='Historical Errors', alpha=0.9)

forecast_index = range(len(residuals), len(residuals) + n_forecast)
ax.plot(forecast_index, forecast_values, linewidth=2.5, color='red', label='Forecast', marker='o', markersize=6)

plt.show()

###################################### 9 #######################################

last_spread = residuals.iloc[-1]
predicted_spread = forecast_values.iloc[-1]


print("Dernier spread observé :", last_spread)
print("Spread prévu :", predicted_spread)

if predicted_spread < last_spread:
    signal = "SHORT spread"
    print("\nPosition fictive prise : SHORT spread")
    print(f"→ SHORT {S2.name}")
    print(f"→ LONG {S1.name}")
    print(f"→ Ratio : 1 unité de {S2.name} contre {b:.4f} unité(s) de {S1.name}")

elif predicted_spread > last_spread:
    signal = "LONG spread"
    print("\nPosition fictive prise : LONG spread")
    print(f"→ LONG {S2.name}")
    print(f"→ SHORT {S1.name}")
    print(f"→ Ratio : 1 unité de {S2.name} contre {b:.4f} unité(s) de {S1.name}")

else:
    signal = "AUCUNE position"
    print("\nPosition fictive prise : aucune position")

##################################### 10 #######################################

for i in range(3):
    residuals = np.concatenate([residuals[3:], forecast_values[:3]])

    auto_model = auto_arima(
        residuals,
        d=0,
        max_p=5,
        max_q=5,
        ic='aic',
        trace=True,
        suppress_warnings=True
    )

    model = ARIMA(residuals, order=auto_model.order).fit()

    n_forecast = 5

    forecast_result = model.get_forecast(steps=n_forecast)
    forecast_values = forecast_result.predicted_mean

    last_spread = residuals[-1]
    predicted_spread = forecast_values[-1]

    fig, ax = plt.subplots(figsize=(20, 7.8))
    ax.plot(range(len(residuals)), residuals, linewidth=2, color='steelblue',
            label='Historical Errors', alpha=0.9)
    forecast_index = range(len(residuals), len(residuals) + n_forecast)
    ax.plot(forecast_index, forecast_values, linewidth=2.5, color='red',
            label='Forecast', marker='o', markersize=6)
    plt.show()


    print("Dernier spread observé :", last_spread)
    print("Spread prévu :", predicted_spread)

    print(f"dans {i} jours")

    if predicted_spread < last_spread:
        signal = "SHORT spread"
        print("\nPosition fictive prise : SHORT spread")
        print(f"→ SHORT {S2.name}")
        print(f"→ LONG {S1.name}")
        print(f"→ Ratio : 1 unité de {S2.name} contre {b:.4f} unité(s) de {S1.name}")

    elif predicted_spread > last_spread:
        signal = "LONG spread"
        print("\nPosition fictive prise : LONG spread")
        print(f"→ LONG {S2.name}")
        print(f"→ SHORT {S1.name}")
        print(f"→ Ratio : 1 unité de {S2.name} contre {b:.4f} unité(s) de {S1.name}")

    else:
        signal = "AUCUNE position"
        print("\nPosition fictive prise : aucune position")

##################################### 11 #######################################

# la raison derrière l'applatissement est que le code prédit les moyennes des
# valeurs futures donc il y a un lissage naturel car les données prédites
# tendent vers la moyenne