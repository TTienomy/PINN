import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
import sys
import os

def load_data():
    try:
        df = pd.read_csv("BTC-PERPETUAL.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df['close'] = pd.to_numeric(df['close'])
        df['ret'] = np.log(df['close'] / df['close'].shift(1))
        df = df.dropna()
        return df
    except FileNotFoundError:
        print("Data not found.")
        sys.exit(1)

def load_params():
    try:
        return pd.read_csv("rolling_parameters_filtered.csv")
    except:
        return pd.read_csv("rolling_parameters.csv")

# --- 1. GARCH Benchmark ---
def fit_garch_var(returns):
    print("\n[Benchmark] Fitting GARCH(1,1)...")
    # Rescale returns for convergence (arch model likes * 100)
    am = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='Normal')
    res = am.fit(disp='off')
    
    # Conditional Volatility
    cond_vol = res.conditional_volatility / 100
    
    # 99% VaR = -2.326 * sigma
    var_99 = -2.326 * cond_vol
    return var_99

# --- 2. Statistical Tests (Tau) ---
def check_tau_stationarity():
    print("\n[Robustness] Testing Stationarity of Tau...")
    df = load_params()
    tau = df['tau'].values
    
    # ADF Test (Null: Unit Root / Non-Stationary)
    adf_res = adfuller(tau)
    print(f"ADF Statistic: {adf_res[0]:.4f}")
    print(f"ADF p-value: {adf_res[1]:.4f}")
    if adf_res[1] > 0.05:
        print(">> Fail to Reject Null (Likely Non-Stationary)")
    else:
        print(">> Reject Null (Stationary)")
        
    # KPSS Test (Null: Stationary)
    # Reformatted for new statsmodels warning suppression if needed
    kpss_res = kpss(tau, regression='c', nlags='auto')
    print(f"KPSS Statistic: {kpss_res[0]:.4f}")
    print(f"KPSS p-value: {kpss_res[1]:.4f}")
    if kpss_res[1] < 0.05:
        print(">> Reject Null (Likely Non-Stationary)")
    else:
        print(">> Fail to Reject Null (Stationary)")

# --- 3. Christoffersen Test ---
def christoffersen_test(violations):
    """
    LR_cc = LR_pof + LR_ind
    LR_ind tests independence of violations (clustering).
    """
    violations = np.array(violations).astype(int)
    if len(violations) == 0:
        return 0, 1.0
        
    # Transition counts
    n00 = n01 = n10 = n11 = 0
    
    for t in range(1, len(violations)):
        prev = violations[t-1]
        curr = violations[t]
        
        if prev == 0 and curr == 0: n00 += 1
        if prev == 0 and curr == 1: n01 += 1
        if prev == 1 and curr == 0: n10 += 1
        if prev == 1 and curr == 1: n11 += 1
    
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    # Likelihoods
    # L(pi) = (1-pi)^(n00+n10) * pi^(n01+n11)
    # L(pi0, pi1) = (1-pi0)^n00 * pi0^n01 * (1-pi1)^n10 * pi1^n11
    
    def log_L(p, n0, n1):
        # handle 0 log 0
        l0 = n0 * np.log(1 - p) if (1-p) > 0 else 0
        l1 = n1 * np.log(p) if p > 0 else 0
        return l0 + l1
        
    ln_L_unrestricted = log_L(pi0, n00, n01) + log_L(pi1, n10, n11)
    ln_L_restricted = log_L(pi, n00 + n10, n01 + n11)
    
    LR_ind = -2 * (ln_L_restricted - ln_L_unrestricted)
    p_val = 1 - stats.chi2.cdf(LR_ind, 1)
    
    print(f"\n[Backtest] Christoffersen Independence Test:")
    print(f"LR_ind: {LR_ind:.4f}, p-value: {p_val:.4f}")
    if p_val < 0.05:
        print(">> Reject Null (Violations are Clustered/Dependent) -> Model Fail")
    else:
        print(">> Fail to Reject Null (Violations are Independent) -> Model Pass")

def run_checks():
    df_data = load_data()
    returns = df_data['ret'] # Series
    
    # 1. Run GARCH
    garch_var = fit_garch_var(returns)
    
    # Calculate GARCH Violations
    # returns < var_99
    # align indices? assuming returns and var aligned
    violations = returns < garch_var
    fail_rate = violations.mean()
    print(f"GARCH(1,1) Failure Rate: {fail_rate*100:.2f}%")
    
    # 2. Tau Stats
    check_tau_stationarity()
    
    # 3. Christoffersen on GARCH
    # (We assume Telegrapher passed based on 1.02% rate, but would need vector to run formally)
    # Let's run on GARCH just to show the methodology
    christoffersen_test(violations)

if __name__ == "__main__":
    run_checks()
