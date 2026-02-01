import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.pinn import TelegrapherPINN
from models.physics_loss import pde_residual
from models.pricing import price_option

def train_model_given_params(c_val, tau_val, epochs=2000, device='cuda'):
    """Trains a PINN for fixed parameters to serve as Ground Truth."""
    print(f"Training Teacher PINN with c={c_val}, tau={tau_val}...")
    model = TelegrapherPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Grid Setup
    L = 2.0 # Smaller domain for c=0.5
    T = 1.0
    Nx_col = 5000
    
    # Physics Points
    t_p = (torch.rand(Nx_col, 1) * T).to(device).requires_grad_(True)
    x_p = ((torch.rand(Nx_col, 1) * 2 * L) - L).to(device).requires_grad_(True)
    
    # IC/BC targets
    sigma_0 = 0.1 # Narrows IC
    N_ib = 500
    t_ic = torch.zeros(N_ib, 1).to(device)
    x_ic = ((torch.rand(N_ib, 1) * 2 * L) - L).to(device)
    u_ic = torch.exp(-0.5 * (x_ic/sigma_0)**2) / (sigma_0 * np.sqrt(2*np.pi))
    
    t_bc = (torch.rand(N_ib, 1) * T).to(device)
    x_bc = torch.cat([-L*torch.ones(N_ib//2,1), L*torch.ones(N_ib//2,1)]).to(device)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        res = pde_residual(model, t_p, x_p, c_val, tau_val)
        l_pde = torch.mean(res**2)
        
        u_pred_ic = model(t_ic, x_ic)
        l_ic = torch.mean((u_pred_ic - u_ic)**2)
        
        u_pred_bc = model(t_bc, x_bc)
        l_bc = torch.mean(u_pred_bc**2)
        
        loss = l_pde + 10.0*l_ic + l_bc
        loss.backward()
        optimizer.step()
        
    return model

def calibrate_synthetic_refined():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calibration using device: {device}")
    
    # --- 1. Realistic Ground Truth ---
    true_c = 0.5   # 50% max move
    true_tau = 0.2
    
    teacher_model = train_model_given_params(true_c, true_tau, epochs=3000, device=device)
    teacher_model.eval()
    
    # Generate Prices
    S0 = 10000.0
    strikes = torch.linspace(8000, 12000, 10).to(device)
    with torch.no_grad():
        market_prices = price_option(teacher_model, 1.0, strikes, S0, L=2.0, device=device)
        
    print("\nTarget Market Prices:")
    for k, p in zip(strikes, market_prices):
        print(f"K={k.item():.0f}, P={p.item():.2f}")
        
    # --- 2. Inverse Problem ---
    print("\nStarting Calibration (Inverse Problem)...")
    
    # Learnable Params (Initialize wrong)
    c_param = nn.Parameter(torch.tensor([0.3], device=device)) 
    tau_param = nn.Parameter(torch.tensor([0.5], device=device))
    
    student_model = TelegrapherPINN().to(device)
    optimizer = optim.Adam([
        {'params': student_model.parameters(), 'lr': 1e-3},
        {'params': [c_param, tau_param], 'lr': 5e-3}
    ])
    
    # Training Loop
    epsilon = 0.01
    hist_c, hist_tau, hist_loss = [], [], []
    
    # Re-use grid logic (simplified)
    L = 2.0
    T = 1.0
    t_p = (torch.rand(5000, 1) * T).to(device).requires_grad_(True)
    x_p = ((torch.rand(5000, 1) * 2 * L) - L).to(device).requires_grad_(True)
    
    # Setup IC/BC (Same as teacher)
    N_ib = 500
    t_ic = torch.zeros(N_ib, 1).to(device)
    x_ic = ((torch.rand(N_ib, 1) * 2 * L) - L).to(device)
    u_ic_target = torch.exp(-0.5 * (x_ic/0.1)**2) / (0.1 * np.sqrt(2*np.pi))
    t_bc = (torch.rand(N_ib, 1) * T).to(device)
    x_bc = torch.cat([-L*torch.ones(N_ib//2,1), L*torch.ones(N_ib//2,1)]).to(device)

    pbar = tqdm(range(3000))
    for epoch in pbar:
        optimizer.zero_grad()
        
        # 1. Physics (Student must obey ITS params c_param, tau_param)
        res = pde_residual(student_model, t_p, x_p, c_param, tau_param)
        l_pde = torch.mean(res**2)
        
        # 2. Data (Student prices must match Market prices)
        # Note: Gradients must flow through price_option -> model -> c_param/tau_param?
        # NO. Gradients flow: Loss -> Price -> Model Output -> Model Weights.
        # BUT c_param/tau_param are ONLY connected via l_pde.
        # So Model Weights update to satisfy Pricing AND PDE(c,tau).
        # c,tau update to minimize PDE(Model) where Model is trying to fit Pricing.
        # This is a bilevel-ish structure, but standard gradient descent handles it as a joint minimization:
        # min_{w, c} ( Loss_data(w) + Loss_pde(w, c) )
        # If w fits data but violates PDE(c), PDE loss is high. 
        # Changing c helps reduce PDE loss while w stays fitted to data.
        
        preds = price_option(student_model, T, strikes, S0, L=L, device=device)
        l_prc = torch.mean((preds - market_prices)**2)
        
        # 3. IC/BC
        u_ic_pred = student_model(t_ic, x_ic)
        l_ic = torch.mean((u_ic_pred - u_ic_target)**2)
        u_bc_pred = student_model(t_bc, x_bc)
        l_bc = torch.mean(u_bc_pred**2)
        
        loss = l_pde + l_prc + 10.0*l_ic + l_bc
        
        loss.backward()
        optimizer.step()
        
        hist_c.append(c_param.item())
        hist_tau.append(tau_param.item())
        hist_loss.append(loss.item())
        
        if epoch % 100 == 0:
            pbar.set_description(f"c:{c_param.item():.2f} (T:{true_c}) | tau:{tau_param.item():.2f} (T:{true_tau}) | L_prc:{l_prc.item():.1e}")

    print(f"\nFinal: c={c_param.item():.4f}, tau={tau_param.item():.4f}")
    
    # Plotting
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(hist_c)
    plt.axhline(true_c, c='r', ls='--')
    plt.title("Wave Speed c")
    
    plt.subplot(1, 3, 2)
    plt.plot(hist_tau)
    plt.axhline(true_tau, c='r', ls='--')
    plt.title("Relaxation tau")
    
    plt.subplot(1, 3, 3)
    plt.plot(hist_loss)
    plt.yscale('log')
    plt.title("Total Loss")
    
    plt.savefig("calibration_refined.png")
    print("Saved calibration_refined.png")

if __name__ == "__main__":
    calibrate_synthetic_refined()
