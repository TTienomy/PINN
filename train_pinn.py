import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.pinn import TelegrapherPINN
from models.physics_loss import pde_residual
from telegrapher_solver import solve_telegrapher # Can import parameters or verifying logic if needed, but we will replicate logic here.

def train_pinn():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Physics Parameters (Must match Phase 1 for validation)
    c = 5.0
    tau = 0.5
    L = 10.0
    T = 1.0
    sigma_0 = 0.2
    
    # Training Parameters
    epochs = 5000  # Start small for interactive debugging, can increase
    batch_size = 2000
    lr = 1e-3
    
    # --- Data Generation ---
    # 1. Collocation Points (Interior of domain)
    # Uniform sampling in [0, T] x [-L, L]
    N_collocation = 10000
    t_physics = torch.rand(N_collocation, 1) * T
    x_physics = (torch.rand(N_collocation, 1) * 2 * L) - L
    
    t_physics = t_physics.to(device)
    x_physics = x_physics.to(device)
    t_physics.requires_grad = True
    x_physics.requires_grad = True

    # 2. Initial Condition Points (t=0)
    # u(0, x) = Gaussian
    # ut(0, x) = 0
    N_ic = 1000
    t_ic = torch.zeros(N_ic, 1).to(device)
    x_ic = ((torch.rand(N_ic, 1) * 2 * L) - L).to(device)
    
    u_ic_target = torch.exp(-0.5 * (x_ic / sigma_0)**2) / (sigma_0 * np.sqrt(2 * np.pi))
    u_t_ic_target = torch.zeros_like(u_ic_target) # Initially static

    # 3. Boundary Condition Points (x = +/- L)
    # u(t, -L) = 0, u(t, L) = 0
    N_bc = 1000
    t_bc = (torch.rand(N_bc, 1) * T).to(device)
    x_bc_left = -L * torch.ones(N_bc // 2, 1).to(device)
    x_bc_right = L * torch.ones(N_bc - (N_bc // 2), 1).to(device)
    x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
    
    u_bc_target = torch.zeros(N_bc, 1).to(device)

    # --- Model Setup ---
    model = TelegrapherPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)

    # --- Training Loop ---
    # Weights for different loss terms
    lambda_pde = 1.0
    lambda_ic = 10.0  # Emphasize ICs early on
    lambda_bc = 1.0
    
    print("Starting Training...")
    loss_history = []
    
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        optimizer.zero_grad()
        
        # 1. PDE Loss (Physics)
        # Resample collocation points occasionally? Keeping fixed for now for stability.
        res = pde_residual(model, t_physics, x_physics, c, tau)
        loss_pde = torch.mean(res ** 2)
        
        # 2. IC Loss
        # Value match: u(0, x)
        u_pred_ic = model(t_ic, x_ic)
        loss_ic_val = torch.mean((u_pred_ic - u_ic_target) ** 2)
        
        # Derivative match: ut(0, x)
        # Need gradients for IC points too
        t_ic_grad = t_ic.clone().detach().requires_grad_(True)
        # Re-eval model on grad-enabled inputs
        u_pred_ic_grad = model(t_ic_grad, x_ic) 
        u_t_pred_ic = torch.autograd.grad(u_pred_ic_grad, t_ic_grad, 
                                          grad_outputs=torch.ones_like(u_pred_ic_grad), 
                                          create_graph=True)[0]
        loss_ic_dt = torch.mean((u_t_pred_ic - u_t_ic_target) ** 2)
        
        loss_ic = loss_ic_val + loss_ic_dt
        
        # 3. BC Loss
        u_pred_bc = model(t_bc, x_bc)
        loss_bc = torch.mean((u_pred_bc - u_bc_target) ** 2)
        
        # Total Loss
        loss = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        
        if epoch % 100 == 0:
            pbar.set_description(f"L_pde: {loss_pde.item():.2e} | L_ic: {loss_ic.item():.2e} | L_bc: {loss_bc.item():.2e}")

    # --- Validation & Visualization ---
    print("\nTraining Complete. Validating...")
    
    # Generate grid for plotting (reuse FDM grid logic)
    Nx = 200
    Nt = 100
    x_eval = torch.linspace(-L, L, Nx)
    t_eval = torch.linspace(0, T, Nt)
    
    # Meshgrid for PINN evaluation
    # Note: meshgrid 'ij' indexing vs 'xy'. 
    T_grid, X_grid = torch.meshgrid(t_eval, x_eval, indexing='ij')
    
    # Flatten for model
    t_flat = T_grid.reshape(-1, 1).to(device)
    x_flat = X_grid.reshape(-1, 1).to(device)
    
    model.eval()
    with torch.no_grad():
        u_pred = model(t_flat, x_flat).reshape(Nt, Nx).cpu().numpy()

    # Plot final profile at t=T
    plt.figure(figsize=(10, 6))
    plt.plot(x_eval.numpy(), u_pred[-1, :], 'r-', label=f"PINN (t={T})", linewidth=2)
    
    # Overlay Light Cone params
    plt.axvline(c*T, color='k', linestyle=':', label='Light Cone')
    plt.axvline(-c*T, color='k', linestyle=':')
    
    plt.title("PINN Solution at Final Time T")
    plt.xlabel("x")
    plt.ylabel("u(T, x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("pinn_result.png")
    print("Saved pinn_result.png")
    
    # Save Model
    torch.save(model.state_dict(), "models/telegrapher_pinn.pth")
    print("Saved models/telegrapher_pinn.pth")

if __name__ == "__main__":
    train_pinn()
