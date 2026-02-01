import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.pinn import ParametricTelegrapherPINN
from models.physics_loss import pde_residual

def train_universal_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Universal Adaptive Model on {device}...")
    
    model = ParametricTelegrapherPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 5000
    Nx = 5000
    L = 0.5
    T = 1.0
    
    # Ranges for Adaptive Parameters
    C_RANGE = [0.002, 0.01]
    TAU_RANGE = [1.0, 10.0]
    
    pbar = tqdm(range(epochs))
    loss_history = []
    
    for epoch in pbar:
        # 1. Sample Physics Coordinates (t, x)
        t_p = (torch.rand(Nx, 1) * T).to(device).requires_grad_(True)
        x_p = ((torch.rand(Nx, 1) * 2 * L) - L).to(device).requires_grad_(True)
        
        # 2. Sample Adaptive Parameters (c, tau)
        # Uniform sampling across their valid ranges
        c_p = (torch.rand(Nx, 1) * (C_RANGE[1] - C_RANGE[0]) + C_RANGE[0]).to(device)
        tau_p = (torch.rand(Nx, 1) * (TAU_RANGE[1] - TAU_RANGE[0]) + TAU_RANGE[0]).to(device)
        
        # 3. IC Points (t=0)
        N_ic = 1000
        t_ic = torch.zeros(N_ic, 1).to(device)
        x_ic = ((torch.rand(N_ic, 1) * 2 * L) - L).to(device)
        # IC parameters also need to change? 
        # Actually the solution u(t,x) depends on c,tau, but the INITIAL CONDITION u(0,x) 
        # is usually independent (just the initial distribution of returns).
        # We assume the same Gaussian IC for all regimes.
        sigma0 = 0.005
        u_ic_target = torch.exp(-0.5 * (x_ic/sigma0)**2) / (sigma0 * np.sqrt(2*np.pi))
        
        # We must provide c, tau for IC inputs too, so the network knows context
        c_ic = (torch.rand(N_ic, 1) * (C_RANGE[1] - C_RANGE[0]) + C_RANGE[0]).to(device)
        tau_ic = (torch.rand(N_ic, 1) * (TAU_RANGE[1] - TAU_RANGE[0]) + TAU_RANGE[0]).to(device)
        
        # 4. BC Points (+/- L)
        N_bc = 500
        t_bc = (torch.rand(N_bc, 1) * T).to(device)
        x_bc = torch.cat([-L*torch.ones(N_bc//2, 1), L*torch.ones(N_bc//2, 1)]).to(device)
        c_bc = (torch.rand(N_bc, 1) * (C_RANGE[1] - C_RANGE[0]) + C_RANGE[0]).to(device)
        tau_bc = (torch.rand(N_bc, 1) * (TAU_RANGE[1] - TAU_RANGE[0]) + TAU_RANGE[0]).to(device)
        
        optimizer.zero_grad()
        
        # Forward Passes
        # Note: We need to modify pde_residual to handle c, tau as TENSORS not scalars
        # But our pde_residual function takes c, tau as args. 
        # If we pass tensors of shape (N,1), broadcast should work if implemented correctly.
        
        # Check pde_residual implementation:
        # It does: residual = u_tt + (1/tau)*u_t - (c^2)*u_xx
        # This works perfectly with tensors!
        
        u_pred = model(t_p, x_p, c_p, tau_p) # Need to implement Forward in pde_residual? 
        # Wait, pde_residual calls model(t, x)... 
        # We need a custom pde_residual for parametric model because the signature is different!
        
        # Custom Residual Calculation Inline
        u = model(t_p, x_p, c_p, tau_p)
        grads = torch.autograd.grad(u, [t_p, x_p], torch.ones_like(u), create_graph=True)[0] # This returns tuple? No.
        # torch.autograd.grad returns tuple of grads matching inputs list
        
        grads = torch.autograd.grad(u, [t_p, x_p], torch.ones_like(u), create_graph=True)
        u_t = grads[0]
        u_x = grads[1]
        
        u_tt = torch.autograd.grad(u_t, t_p, torch.ones_like(u_t), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_p, torch.ones_like(u_x), create_graph=True)[0]
        
        res = u_tt + (1.0 / tau_p) * u_t - (c_p**2) * u_xx
        
        l_pde = torch.mean(res**2)
        
        # IC/BC Loss
        u_ic_pred = model(t_ic, x_ic, c_ic, tau_ic)
        l_ic = torch.mean((u_ic_pred - u_ic_target)**2)
        
        u_bc_pred = model(t_bc, x_bc, c_bc, tau_bc)
        l_bc = torch.mean(u_bc_pred**2)
        
        loss = l_pde + 10.0*l_ic + l_bc
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if epoch % 100 == 0:
            pbar.set_description(f"Loss: {loss.item():.5f}")
            
    # Save Model
    torch.save(model.state_dict(), "models/parametric_pinn.pth")
    print("Saved Universal Model to models/parametric_pinn.pth")
    
    return model

def validate_adaptation(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Regimes found in Phase 7
    regimes = [
        {"name": "2019 (High Diffusion)", "c": 0.0065, "tau": 2.76},
        {"name": "2026 (High Momentum)", "c": 0.0053, "tau": 6.11}
    ]
    
    plt.figure(figsize=(10, 6))
    
    with torch.no_grad():
        x_grid = torch.linspace(-0.5, 0.5, 1000).to(device)
        t_grid = torch.ones_like(x_grid).view(-1, 1) * 1.0 # T=1
        x_in = x_grid.view(-1, 1)
        
        for r in regimes:
            c_val = r["c"]
            tau_val = r["tau"]
            
            # Create parameter tensors
            c_tensor = torch.full_like(x_in, c_val)
            tau_tensor = torch.full_like(x_in, tau_val)
            
            u_pred = model(t_grid, x_in, c_tensor, tau_tensor).squeeze()
            u_pdf = torch.relu(u_pred) # Positivity
            
            # Normalize for plot
            dx = x_grid[1] - x_grid[0]
            mass = torch.sum(u_pdf) * dx
            u_pdf = u_pdf / mass
            
            plt.plot(x_grid.cpu().numpy(), u_pdf.cpu().numpy(), label=f"{r['name']}\nc={c_val}, tau={tau_val}")

    plt.title("Adaptive Solver: Instant Regime Switching")
    plt.xlabel("Log Return")
    plt.ylabel("Density")
    plt.xlim(-0.1, 0.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("adaptive_solver_demo.png")
    print("Saved adaptive_solver_demo.png")

if __name__ == "__main__":
    model = train_universal_model()
    validate_adaptation(model)
