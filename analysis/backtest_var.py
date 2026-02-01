import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from models.pinn import TelegrapherPINN
from models.physics_loss import pde_residual

# Calibrated Parameters
C_CALIBRATED = 0.0056
TAU_CALIBRATED = 5.11

def load_and_process_data(csv_path="BTC-PERPETUAL.csv"):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['close'] = pd.to_numeric(df['close'])
    df['ret'] = np.log(df['close'] / df['close'].shift(1))
    return df['ret'].dropna().values

def get_standardized_quantile(q=0.01):
    """
    Generates the standardized quantile from the PINN 
    by fitting to the BTC data to capture the heavy tails.
    """
    print("Calibrating PINN to Historical Data to extract Shape...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data & Empirical PDF Target
    returns = load_and_process_data()
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(returns, bw_method='scott')
    
    L = 0.5
    dict_x = np.linspace(-L, L, 1000)
    dict_y = kde(dict_x)
    
    x_target = torch.tensor(dict_x, dtype=torch.float32).view(-1, 1).to(device)
    u_target = torch.tensor(dict_y, dtype=torch.float32).view(-1, 1).to(device)
    
    # 2. Train Model
    model = TelegrapherPINN().to(device)
    c_param = torch.nn.Parameter(torch.tensor([0.5], device=device)) 
    tau_param = torch.nn.Parameter(torch.tensor([0.5], device=device))
    
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 1e-3},
        {'params': [c_param, tau_param], 'lr': 5e-3}
    ], lr=1e-3)
    
    # We need a decent fit, 2000 epochs with Data Loss
    Nx = 2000
    T = 1.0
    
    for i in tqdm(range(3000), desc="Learning Shape"):
        # Physics Inputs
        t_p = (torch.rand(Nx, 1) * T).to(device).requires_grad_(True)
        x_p = ((torch.rand(Nx, 1) * 2 * L) - L).to(device).requires_grad_(True)
        
        # Data Inputs
        t_data = torch.ones_like(x_target) * T
        
        # IC/BC
        t_ic = torch.zeros(500, 1).to(device)
        x_ic = ((torch.rand(500, 1) * 2 * L) - L).to(device)
        u_ic_target_val = torch.exp(-0.5 * (x_ic/0.005)**2) / (0.005 * np.sqrt(2*np.pi))
        
        optimizer.zero_grad()
        
        # Losses
        # Note: Unconstrained fit to match Phase 3 result
        c_val = torch.abs(c_param) + 1e-6
        tau_val = torch.abs(tau_param) + 1e-6
        
        res = pde_residual(model, t_p, x_p, c_val, tau_val)
        l_pde = torch.mean(res**2)
        l_ic = torch.mean((model(t_ic, x_ic) - u_ic_target_val)**2)
        
        # Data Loss (Crucial!)
        u_pred = model(t_data, x_target)
        l_data = torch.mean((u_pred - u_target)**2)
        
        loss = l_pde + 10.0*l_ic + 100.0*l_data
        
        loss.backward()
        optimizer.step()
        
    # 2. Extract PDF and Standardize
    with torch.no_grad():
        x_grid = torch.linspace(-0.5, 0.5, 5000).to(device)
        dx = x_grid[1] - x_grid[0]
        t_grid = torch.ones_like(x_grid).view(-1, 1) * T
        
        u_pdf_raw = model(t_grid, x_grid.view(-1, 1)).squeeze()
        u_pdf = torch.relu(u_pdf_raw) # Enforce positive
        
        # Normalize
        mass = torch.sum(u_pdf) * dx
        pdf_norm = u_pdf / mass
        
        # Calculate Variance
        mean = torch.sum(x_grid * pdf_norm) * dx
        variance = torch.sum((x_grid - mean)**2 * pdf_norm) * dx
        std_dev = torch.sqrt(variance)
        
        print(f"Learned Shape Mean: {mean.item():.6f}, Std: {std_dev.item():.6f}")
        
        # 3. Find Quantile
        cdf = torch.cumsum(pdf_norm, dim=0) * dx
        
        # 1% Quantile
        idx = (cdf >= q).nonzero(as_tuple=True)[0][0]
        x_q = x_grid[idx]
        
        # Z-score
        z_q = (x_q - mean) / std_dev
        
        print(f"Telegrapher {q*100}% Quantile (Z-score): {z_q.item():.4f}")
        
    return z_q.item()

def run_backtest():
    # 1. Get Coefficients
    z_tele = get_standardized_quantile(0.01)
    z_gauss = norm.ppf(0.01) # -2.326
    
    print(f"\nGaussian VaR Factor: {z_gauss:.4f}")
    print(f"Telegrapher VaR Factor: {z_tele:.4f}")
    
    # 2. Load Data
    df = pd.read_csv("BTC-PERPETUAL.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['close'] = pd.to_numeric(df['close'])
    df['ret'] = np.log(df['close'] / df['close'].shift(1))
    
    window = 365
    records = []
    
    print(f"\nRunning Rolling Backtest (Window={window})...")
    
    for t in range(window, len(df)):
        # History window
        hist = df['ret'].iloc[t-window:t].values
        
        # Current realization
        actual_ret = df['ret'].iloc[t]
        date = df['date'].iloc[t]
        
        # Volatility Estimate (on Window)
        sigma_t = np.std(hist)
        
        # Calculate VaR (1 day, 99%)
        # VaR is a negative return number
        var_gauss = z_gauss * sigma_t
        var_tele = z_tele * sigma_t
        
        # Check Breach (Loss > VaR => Return < VaR)
        breach_gauss = 1 if actual_ret < var_gauss else 0
        breach_tele = 1 if actual_ret < var_tele else 0
        
        records.append({
            'date': date,
            'ret': actual_ret,
            'sigma': sigma_t,
            'var_gauss': var_gauss,
            'var_tele': var_tele,
            'breach_gauss': breach_gauss,
            'breach_tele': breach_tele
        })
        
    res_df = pd.DataFrame(records)
    
    # 3. Analyze Results
    n_days = len(res_df)
    n_fail_gauss = res_df['breach_gauss'].sum()
    n_fail_tele = res_df['breach_tele'].sum()
    
    rate_gauss = n_fail_gauss / n_days
    rate_tele = n_fail_tele / n_days
    
    print("\n=== Backtest Results (Target Failure Rate: 1.00%) ===")
    print(f"Total Days Scored: {n_days}")
    print(f"Gaussian Failures: {n_fail_gauss} ({rate_gauss:.2%})")
    print(f"Telegrapher Failures: {n_fail_tele} ({rate_tele:.2%})")
    
    # 4. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(res_df['date'], res_df['ret'], color='gray', alpha=0.5, label='Daily Return')
    plt.plot(res_df['date'], res_df['var_gauss'], color='blue', linestyle='--', label=f'Gaussian VaR (Fail={rate_gauss:.2%})')
    plt.plot(res_df['date'], res_df['var_tele'], color='red', linewidth=1.5, label=f'Telegrapher VaR (Fail={rate_tele:.2%})')
    
    # Mark breaches
    breaches_tele = res_df[res_df['breach_tele'] == 1]
    plt.scatter(breaches_tele['date'], breaches_tele['ret'], color='red', marker='x', s=50, zorder=5)
    
    plt.title("99% VaR Backtest: Gaussian vs Telegrapher")
    plt.ylabel("Log Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("backtest_var_result.png")
    print("Saved backtest_var_result.png")

if __name__ == "__main__":
    run_backtest()
