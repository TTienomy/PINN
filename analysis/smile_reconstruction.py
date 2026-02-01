import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from models.pinn import TelegrapherPINN
from models.physics_loss import pde_residual
from models.pricing import price_option

# Calibrated Parameters
C_CALIBRATED = 0.0056
TAU_CALIBRATED = 5.11

def train_forward_model(attempts=1):
    """
    Trains a PINN to solve the Relativistic Telegrapher's Equation
    with fixed c and tau parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Forward Model on {device}...")
    
    model = TelegrapherPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    L = 0.5
    T = 1.0
    Nx = 5000
    
    # Pre-generate points
    t_p = (torch.rand(Nx, 1) * T).to(device).requires_grad_(True)
    x_p = ((torch.rand(Nx, 1) * 2 * L) - L).to(device).requires_grad_(True)
    
    sigma0 = 0.005
    N_ic = 500
    t_ic = torch.zeros(N_ic, 1).to(device)
    x_ic = ((torch.rand(N_ic, 1) * 2 * L) - L).to(device)
    u_ic_target = torch.exp(-0.5 * (x_ic/sigma0)**2) / (sigma0 * np.sqrt(2*np.pi))
    
    N_bc = 200
    t_bc = (torch.rand(N_bc, 1) * T).to(device)
    x_bc = torch.cat([-L*torch.ones(N_bc//2, 1), L*torch.ones(N_bc//2, 1)]).to(device)
    
    epochs = 4000
    pbar = tqdm(range(epochs))
    
    # Fixed parameters
    c = torch.tensor([C_CALIBRATED], device=device)
    tau = torch.tensor([TAU_CALIBRATED], device=device)
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        # PDE
        res = pde_residual(model, t_p, x_p, c, tau)
        l_pde = torch.mean(res**2)
        
        # IC
        u_ic_pred = model(t_ic, x_ic)
        l_ic = torch.mean((u_ic_pred - u_ic_target)**2)
        
        # BC
        u_bc_pred = model(t_bc, x_bc)
        l_bc = torch.mean(u_bc_pred**2)
        
        # Total Loss - Physics Only (Forward Problem)
        loss = l_pde + 10.0*l_ic + l_bc
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            pbar.set_description(f"Loss: {loss.item():.5f}")
            
    return model

def bs_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def implied_volatility(price, S, K, T, r):
    """Numerically inverts BS formula to find sigma."""
    def obj(sigma):
        return bs_price(S, K, T, r, sigma) - price
        
    try:
        # Bounded search for IV between 0.1% and 500%
        return brentq(obj, 1e-4, 5.0)
    except ValueError:
        return np.nan

def generate_smile():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_forward_model()
    model.eval()
    
    # Option Parameters
    S0 = 10000.0
    T = 1.0
    r = 0.0 # Simplify r=0 for crypto/perpetual proxy
    
    # Strikes Range: +/- 15% 
    # Log moneyness k = ln(K/S0) -> +/- 0.15
    # K = S0 * exp(k)
    k_grid = np.linspace(-0.15, 0.15, 30)
    strikes = S0 * np.exp(k_grid)
    strikes_t = torch.tensor(strikes, dtype=torch.float32).to(device)
    
    print("\nPricing Options...")
    with torch.no_grad():
        # Get raw PDF first
        x_grid = torch.linspace(-0.5, 0.5, 2000).to(device)
        dx = x_grid[1] - x_grid[0]
        
        t_tensor = torch.ones_like(x_grid).view(-1, 1).to(device) * T
        x_tensor = x_grid.view(-1, 1)
        
        # Enforce Positivity!
        u_pdf_raw = model(t_tensor, x_tensor).squeeze()
        u_pdf = torch.relu(u_pdf_raw) 
        
        # 1. Normalize Mass
        integral = torch.sum(u_pdf) * dx
        u_pdf_norm = u_pdf / integral
        print(f"PDF Mass corrected: {integral.item():.4f} -> 1.0")
        
        # 2. Enforce Martingale: E[S_T] = S0 * exp(rT)
        # S_T = S0 * exp(x)
        # We need sum( S0 * exp(x) * u * dx ) = S0 (assuming r=0)
        # i.e. sum( exp(x) * u * dx ) = 1
        
        expectation = torch.sum(torch.exp(x_grid) * u_pdf_norm) * dx
        drift_correction = -torch.log(expectation)
        print(f"Martingale Drift Correction: {drift_correction.item():.6f}")
        
        # Apply shift to x coordinates implicitly by shifting option strikes or underlying
        # Equivalent to acting as if x is shifted by drift_correction
        # S_T_new = S0 * exp(x + drift)
        
        # Re-calculate prices with corrected measure
        prices = []
        S_T_corrected = S0 * torch.exp(x_grid + drift_correction)
        
        for K in strikes:
             payoff = torch.relu(S_T_corrected - K)
             price = torch.sum(payoff * u_pdf_norm) * dx
             prices.append(price.item())
    
    # prices is list of floats
    prices = np.array(prices)
    
    print("Calculating Implied Volatilities...")
    ivs = []
    
    # Plot formatting for IV
    # Filter out NaNs for plotting
    clean_strikes = []
    clean_ivs = []
    
    for K, P in zip(strikes, prices):
        # Intrinsic value check
        intrinsic = max(S0 - K, 0)
        # Add small epsilon buffer
        if P < intrinsic - 1e-4:
            print(f"Warning: Arb violation at K={K:.0f}, P={P:.2f} <= Intrinsic={intrinsic:.2f}")
            ivs.append(np.nan)
        else:
            # Ensure price is not too high (Call < S0)
            if P >= S0:
                ivs.append(np.nan)
            else:
                iv = implied_volatility(P, S0, K, T, r)
                ivs.append(iv)
                if not np.isnan(iv):
                    clean_strikes.append(K)
                    clean_ivs.append(iv)
            
    # Visualize Smile
    plt.figure(figsize=(10, 6))
    if len(clean_strikes) > 0:
        plt.plot(clean_strikes, clean_ivs, 'o-', color='purple', linewidth=2, label='Relativistic PINN IV')
        
        # Classical Flat Reference (using ATM IV)
        atm_idx = np.argmin(np.abs(np.array(clean_strikes) - S0))
        atm_iv = clean_ivs[atm_idx]
        plt.axhline(atm_iv, color='gray', linestyle='--', label=f'Flat BS Vol (ATM={atm_iv:.2%})')
        
    plt.title(f"Implied Volatility Smile (Relativistic)\nc={C_CALIBRATED}, tau={TAU_CALIBRATED}")
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("volatility_smile.png")
    print("Saved volatility_smile.png")

if __name__ == "__main__":
    generate_smile()
