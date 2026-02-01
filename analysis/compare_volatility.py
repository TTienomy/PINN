import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Calibrated Parameters
C_CALIBRATED = 0.0056
TAU_CALIBRATED = 5.11

def load_returns(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Ensure sorted by date ascending
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate Log Returns
    df['close'] = pd.to_numeric(df['close'])
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    return df['log_ret'].dropna().values

def analyze_volatility():
    returns = load_returns("BTC-PERPETUAL.csv")
    
    # 1. Classical Volatility (Daily)
    sigma_daily = np.std(returns)
    sigma_annual = sigma_daily * np.sqrt(365)
    
    print(f"Classical Daily Volatility (sigma): {sigma_daily:.6f}")
    print(f"Classical Annual Volatility: {sigma_annual:.2f}")
    
    # 2. Relativistic Parameters
    c = C_CALIBRATED
    tau = TAU_CALIBRATED
    
    print(f"\nRelativistic Parameters:")
    print(f"Wave Speed (c): {c:.6f}")
    print(f"Relaxation Time (tau): {tau:.2f} days")
    
    # 3. Diffusion Limit Check
    # D_eff = c^2 * tau
    # In Heat equation: D = sigma^2 / 2
    # So sigma_eff = sqrt(2 * c^2 * tau) = c * sqrt(2 * tau)
    
    sigma_relativistic = c * np.sqrt(2 * tau)
    
    print(f"\nDiffusion Limit Comparison:")
    print(f"Effective Volatility (sigma_eff = c*sqrt(2*tau)): {sigma_relativistic:.6f}")
    
    error = abs(sigma_relativistic - sigma_daily) / sigma_daily * 100
    print(f"Difference: {error:.2f}%")
    
    # 4. Visualization
    # Compare Gaussian(0, sigma) vs Gaussian(0, sigma_eff)
    from scipy.stats import norm
    
    x = np.linspace(-0.15, 0.15, 1000)
    pdf_classical = norm.pdf(x, 0, sigma_daily)
    pdf_relativistic_diff = norm.pdf(x, 0, sigma_relativistic)
    
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=100, density=True, alpha=0.5, color='blue', label='Empirical Returns')
    plt.plot(x, pdf_classical, 'k--', label=f'Classical BS (sigma={sigma_daily:.4f})')
    plt.plot(x, pdf_relativistic_diff, 'r-', linewidth=2, label=f'Relativistic Diffusion Limit (sigma_eff={sigma_relativistic:.4f})')
    plt.title(f"Volatility Comparison\nc={c}, tau={tau} vs sigma={sigma_daily:.4f}")
    plt.xlabel("Log Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Interpretation text
    txt = (f"c={c}, tau={tau}\n"
           f"sigma_eff = {sigma_relativistic:.4f}\n"
           f"Error = {error:.1f}%")
    plt.text(0.05, 0.8, txt, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig("volatility_analysis.png")
    print("Saved volatility_analysis.png")

if __name__ == "__main__":
    analyze_volatility()
