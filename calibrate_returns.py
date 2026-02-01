import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.pinn import TelegrapherPINN
from models.physics_loss import pde_residual

def load_and_process_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Ensure sorted by date ascending
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate Log Returns
    # r_t = ln(P_t / P_{t-1})
    df['close'] = pd.to_numeric(df['close'])
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Drop NaN
    returns = df['log_ret'].dropna().values
    
    print(f"Loaded {len(returns)} daily returns.")
    print(f"Mean: {returns.mean():.6f}, Std: {returns.std():.6f}")
    return returns

def get_empirical_pdf(returns, num_points=1000, domain_factor=6.0):
    """Estimates PDF using Kernel Density Estimation."""
    std = returns.std()
    x_grid = np.linspace(-domain_factor*std, domain_factor*std, num_points)
    
    kde = gaussian_kde(returns, bw_method='scott')
    pdf_eval = kde(x_grid)
    
    return x_grid, pdf_eval

def calibrate_returns_unconstrained():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    csv_path = "BTC-PERPETUAL.csv"
    returns = load_and_process_data(csv_path)
    
    # Use natural scale
    # BTC daily returns are small ~0.02. L=0.5 covers almost everything with margin.
    L = 0.5
    T = 1.0 # 1 day unit
    
    # Get empirical PDF on the natural domain
    # Use a wider domain factor to ensure we capture the tails for visualization
    x_grid, pdf_target = get_empirical_pdf(returns, num_points=1000, domain_factor=10.0)
    
    # Mask target to strictly [-L, L]
    mask = (x_grid >= -L) & (x_grid <= L)
    x_target = x_grid[mask]
    u_target = pdf_target[mask]
    
    # Convert to Tensor
    x_target_t = torch.tensor(x_target, dtype=torch.float32).view(-1, 1).to(device)
    u_target_t = torch.tensor(u_target, dtype=torch.float32).view(-1, 1).to(device)
    
    # 2. Setup Model
    model = TelegrapherPINN().to(device)
    
    # Learnable Parameters - Unconstrained
    # Initialize with something reasonable but allow it to move anywhere
    c_param = nn.Parameter(torch.tensor([0.5], device=device)) 
    tau_param = nn.Parameter(torch.tensor([0.5], device=device))
    
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 1e-3},
        {'params': [c_param, tau_param], 'lr': 5e-3} # Higher LR for params
    ])
    
    # 3. Training Loop
    # Physics points
    Nx = 5000 
    
    epochs = 10000 # More epochs to ensure convergence
    hist_loss = []
    
    print("Starting Unconstrained Calibration...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Physics inputs
        t_p = (torch.rand(Nx, 1) * T).to(device).requires_grad_(True)
        x_p = ((torch.rand(Nx, 1) * 2 * L) - L).to(device).requires_grad_(True)
        
        # IC inputs (Narrow Gaussian approximating Delta)
        sigma0 = 0.005
        N_ic = 500
        t_ic = torch.zeros(N_ic, 1).to(device)
        x_ic = ((torch.rand(N_ic, 1) * 2 * L) - L).to(device)
        u_ic_target = torch.exp(-0.5 * (x_ic/sigma0)**2) / (sigma0 * np.sqrt(2*np.pi))
        
        # BC inputs (Zero)
        N_bc = 200
        t_bc = (torch.rand(N_bc, 1) * T).to(device)
        x_bc = torch.cat([-L*torch.ones(N_bc//2, 1), L*torch.ones(N_bc//2, 1)]).to(device)
        
        # Data inputs
        t_data = torch.ones_like(x_target_t) * T
        
        # PDE Loss
        # We allow c and tau to be whatever, but we take absolute value or square logic in PDE?
        # The pde_residual function uses c and tau directly.
        # Let's enforcing positivity via abs just in case, or softplus.
        # Simple abs helps gradient flow better than softplus sometimes near 0.
        c_val = torch.abs(c_param) + 1e-6
        tau_val = torch.abs(tau_param) + 1e-6
        
        res = pde_residual(model, t_p, x_p, c_val, tau_val)
        l_pde = torch.mean(res**2)
        
        # IC/BC Loss
        u_ic_pred = model(t_ic, x_ic)
        l_ic = torch.mean((u_ic_pred - u_ic_target)**2)
        u_bc_pred = model(t_bc, x_bc)
        l_bc = torch.mean(u_bc_pred**2)
        
        # Data Loss
        u_data_pred = model(t_data, x_target_t)
        l_data = torch.mean((u_data_pred - u_target_t)**2)
        
        # Total Loss
        # We really want to fit the data, so weight it high.
        loss = l_pde + 10.0*l_ic + l_bc + 100.0*l_data
        
        loss.backward()
        optimizer.step()
        
        hist_loss.append(loss.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.5f} | c={c_val.item():.4f}, tau={tau_val.item():.4f}")

    print(f"Final: c={c_val.item():.4f}, tau={tau_val.item():.4f}")
    
    # 4. Visualization
    model.eval()
    with torch.no_grad():
        u_final = model(t_data, x_target_t).cpu().numpy()
        
    plt.figure(figsize=(10, 6))
    plt.plot(x_target, u_target, label="Empirical PDF (Data)", color='blue', alpha=0.6, linewidth=2)
    plt.plot(x_target, u_final, label=f"Calibrated (c={c_val.item():.4f})", color='red', linestyle='--')
    plt.title(f"Unconstrained Calibration Result\nc={c_val.item():.4f}, tau={tau_val.item():.4f}")
    plt.xlabel("Log Return")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # limit x access to see the peak
    plt.xlim(-0.2, 0.2)
    
    plt.savefig("calibration_returns_result.png")
    print("Saved calibration_returns_result.png")

if __name__ == "__main__":
    calibrate_returns_unconstrained()
