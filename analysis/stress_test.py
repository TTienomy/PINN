import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from models.pinn import TelegrapherPINN
from models.physics_loss import pde_residual

# Calibrated
C_BASE = 0.0056
TAU_BASE = 5.11

def generate_pdf(model, c_val, tau_val, label, color):
    # We train a NEW model head or just evaluate?
    # Actually, we need to solve the PDE for these NEW parameters.
    # The current weights are learned for C_BASE.
    # We must re-solve (Forward Problem) for the Shock scenarios.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Solving for {label} (c={c_val:.4f}, tau={tau_val:.2f})...")
    
    model = TelegrapherPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    c = torch.tensor([c_val], device=device)
    tau = torch.tensor([tau_val], device=device)
    
    L = 0.5
    T = 1.0
    Nx = 2000
    
    # Fast solve
    for i in range(1000):
        t_p = (torch.rand(Nx, 1) * T).to(device).requires_grad_(True)
        x_p = ((torch.rand(Nx, 1) * 2 * L) - L).to(device).requires_grad_(True)
        
        # IC
        sigma0 = 0.005
        t_ic = torch.zeros(500, 1).to(device)
        x_ic = ((torch.rand(500, 1) * 2 * L) - L).to(device)
        u_ic_target = torch.exp(-0.5 * (x_ic/sigma0)**2) / (sigma0 * np.sqrt(2*np.pi))
        
        # BC
        t_bc = (torch.rand(200, 1) * T).to(device)
        x_bc = torch.cat([-L*torch.ones(100, 1), L*torch.ones(100, 1)]).to(device)
        
        optimizer.zero_grad()
        res = pde_residual(model, t_p, x_p, c, tau)
        # Physics only solve
        loss = torch.mean(res**2) + \
               10.0*torch.mean((model(t_ic, x_ic) - u_ic_target)**2) + \
               torch.mean(model(t_bc, x_bc)**2)
        loss.backward()
        optimizer.step()
        
    # Extract PDF
    with torch.no_grad():
        x_grid = torch.linspace(-0.5, 0.5, 1000).to(device)
        t_grid = torch.ones_like(x_grid).view(-1, 1) * T
        u_pdf = torch.relu(model(t_grid, x_grid.view(-1, 1)).squeeze())
        
        # Normalize
        dx = x_grid[1] - x_grid[0]
        mass = torch.sum(u_pdf) * dx
        u_pdf = u_pdf / mass
        
    return x_grid.cpu().numpy(), u_pdf.cpu().numpy()

def run_stress_test():
    base_x, base_y = generate_pdf(None, C_BASE, TAU_BASE, "Base Case", "blue")
    
    # Shock 1: Liquidity Crisis (c halves)
    # Lower speed limit -> Information propagates slower -> Energy trapped -> Heavier Tails?
    # Actually in Telegrapher, lower c means transition to Wave behavior is stronger?
    liq_x, liq_y = generate_pdf(None, C_BASE * 0.5, TAU_BASE, "Liquidity Crisis (c/2)", "red")
    
    # Shock 2: Panic (tau halves) (More friction/damping? No, tau is relaxation time)
    # Small tau -> Diffusion limit. Large tau -> Ballistic.
    # Panic usually means loss of memory/randomness? Or high momentum?
    # Let's say "High Friction Regmie" (small tau) vs "Ballistic Regime" (large tau)
    panic_x, panic_y = generate_pdf(None, C_BASE, TAU_BASE * 0.1, "High Friction (tau/10)", "green")
    
    plt.figure(figsize=(10, 6))
    plt.plot(base_x, base_y, label=f"Base (c={C_BASE}, tau={TAU_BASE})", color='blue', linewidth=2)
    plt.plot(liq_x, liq_y, label=f"Liquidity Crisis (c={C_BASE*0.5:.4f})", color='red', linestyle='--')
    plt.plot(panic_x, panic_y, label=f"High Friction (tau={TAU_BASE*0.1:.2f})", color='green', linestyle=':')
    
    plt.title("Stress Test: Physics Parameter Sensitivity")
    plt.xlabel("Log Return")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.15, 0.15)
    
    plt.savefig("stress_test_result.png")
    print("Saved stress_test_result.png")

if __name__ == "__main__":
    run_stress_test()
