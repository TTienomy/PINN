import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import sys
import os

# Calibrated Z from Phase 5
Z_TELE_RAW = -4.9076

def load_data():
    df = pd.read_csv("BTC-PERPETUAL.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['close'] = pd.to_numeric(df['close'])
    df['ret'] = np.log(df['close'] / df['close'].shift(1))
    return df.dropna()

def calculate_failure_rate(z_score, returns, window=365):
    # Rolling volatility
    rolling_std = returns.rolling(window=window).std().shift(1) # Predict for *next* day
    
    # Calculate VaR
    var_thresholds = z_score * rolling_std
    
    # Check breaches
    # Drop first 'window' NaNs
    valid_data = pd.concat([returns, var_thresholds], axis=1).dropna()
    valid_data.columns = ['ret', 'var']
    
    n_days = len(valid_data)
    n_fail = (valid_data['ret'] < valid_data['var']).sum()
    
    return n_fail / n_days

def optimize_efficiency():
    print("Optimization: Targetting 1.00% Failure Rate...")
    df = load_data()
    returns = df['ret']
    
    # Objective function: Rate(z) - 0.01 = 0
    def objective(z):
        rate = calculate_failure_rate(z, returns)
        return rate - 0.01
        
    # Search range: [-10 (Extreme), -1 (Loose)]
    try:
        z_opt = brentq(objective, -10.0, -1.0)
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Grid search fallback
        zs = np.linspace(-6, -1, 100)
        rates = [calculate_failure_rate(z, returns) for z in zs]
        idx = np.argmin(np.abs(np.array(rates) - 0.01))
        z_opt = zs[idx]

    final_rate = calculate_failure_rate(z_opt, returns)
    
    print(f"\nResults:")
    print(f"Original Telegrapher Z (99%): {Z_TELE_RAW:.4f} (Rate: ~0.08%)")
    print(f"Optimized Efficient Z (99%):  {z_opt:.4f} (Rate: {final_rate:.2%})")
    
    # Capital Efficiency Gain
    # Margin scales linearly with Z-factor
    # Savings = (Old_Z - New_Z) / Old_Z
    # Note: Z is negative, concern is magnitude
    margin_reduction = (abs(Z_TELE_RAW) - abs(z_opt)) / abs(Z_TELE_RAW)
    print(f"Capital Efficiency Gain: {margin_reduction:.2%} lower margin requirements")
    
    # Visualization
    window = 365
    rolling_std = returns.rolling(window=window).std().shift(1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], returns, color='gray', alpha=0.4, label='Return')
    plt.plot(df['date'], Z_TELE_RAW * rolling_std, color='red', linestyle='--', alpha=0.6, label='Original Telegrapher (Conservative)')
    plt.plot(df['date'], z_opt * rolling_std, color='green', linewidth=1.5, label=f'Optimized Telegrapher (Target 1%)')
    
    # Highlight failures for Optimized
    var_opt = z_opt * rolling_std
    failures = df[returns < var_opt]
    plt.scatter(failures['date'], failures['ret'], color='green', marker='x', s=30, label='Breaches (Optimized)')
    
    plt.title(f"Capital Efficiency Optimization\nReduced Margin by {margin_reduction:.1%} while maintaining 1% Risk Target")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("efficiency_optimization.png")
    print("Saved efficiency_optimization.png")
    
    return z_opt

if __name__ == "__main__":
    optimize_efficiency()
