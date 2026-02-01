import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import os

"""
Project Event Horizon: Relativistic Option Pricing Research System
Phase 1: Literature Implementation - Telegrapher's Equation Solver

This script numerically solves the 1D Telegrapher's Equation using the Finite Difference Method (FDM).
Equation: ∂²u/∂t² + (1/τ)∂u/∂t - c²∂²u/∂x² = 0

Physical Constants:
- c (wave speed): Represents the maximum speed of information propagation in the market.
  In a relativistic context, this replaces the infinite propagation speed of the Heat Equation (Black-Scholes).
  It enforces causality, creating a "Light Cone" outside of which the probability density is strictly zero.
- τ (relaxation time): Characteristic time scale for the system to adjust to changes.
"""

def solve_telegrapher():
    # --- Parameters ---
    L = 10.0          # Spatial domain [-L, L]
    T = 1.0           # Time domain [0, T]
    Nx = 1000         # Number of spatial grid points
    Nt = 1000         # Number of time steps
    
    c = 5.0           # Wave speed (Limit of information propagation)
    tau = 0.5         # Relaxation time
    
    # Derived parameters
    dx = 2 * L / (Nx - 1)
    dt = T / (Nt - 1)
    
    # Courant-Friedrichs-Lewy (CFL) condition for stability: c * dt / dx <= 1
    cfl = c * dt / dx
    if cfl > 1.0:
        print(f"Warning: CFL condition violated (CFL = {cfl:.2f} > 1.0). Instability likely.")
    else:
        print(f"CFL condition met: {cfl:.4f}")

    # Grid setup
    x = torch.linspace(-L, L, Nx)
    t_grid = torch.linspace(0, T, Nt)
    
    # Initialize solution tensor u(t, x)
    u = torch.zeros((Nt, Nx))
    
    # --- Initial Conditions ---
    # u(0, x) = Approximated Dirac Delta (Gaussian)
    # ∂u/∂t(0, x) = 0 (Initially static)
    
    sigma_0 = 0.2
    u[0, :] = torch.exp(-0.5 * (x / sigma_0)**2) / (sigma_0 * np.sqrt(2 * np.pi))
    
    # Second time step using Taylor expansion for ∂u/∂t(0, x) = 0
    # ∂²u/∂t² = c²∂²u/∂x² - (1/τ)∂u/∂t  => at t=0, ∂²u/∂t² = c²∂²u/∂x²
    # u(dt, x) ≈ u(0, x) + dt*ut(0, x) + 0.5*dt^2*utt(0, x)
    #          = u(0, x) + 0 + 0.5*dt^2 * (c² * u_xx)
    
    # Central difference for second derivative
    u_xx = (torch.roll(u[0], -1) - 2*u[0] + torch.roll(u[0], 1)) / dx**2
    # Apply BCs to derivative (Dirichlet u=0 at constant boundaries implies u_xx=0 approx if far enough)
    u_xx[0] = 0
    u_xx[-1] = 0
    
    u[1, :] = u[0, :] + 0.5 * (dt**2) * (c**2) * u_xx
    
    # --- Time Stepping (FDM) ---
    # Discretization of Telegrapher's eq:
    # (u_{n+1} - 2u_n + u_{n-1})/dt^2 + (1/tau)*(u_{n+1} - u_{n-1})/(2dt) = c^2 * (u_{n, i+1} - 2u_{n, i} + u_{n, i-1})/dx^2
    
    # Grouping terms for u_{n+1}:
    # u_{n+1} * (1/dt^2 + 1/(2*tau*dt)) = 
    #   2u_n/dt^2 - u_{n-1}/dt^2 + u_{n-1}/(2*tau*dt) + c^2/dx^2 * (u_{n, i+1} - 2u_{n, i} + u_{n, i-1})
    
    alpha = 1.0 / dt**2
    beta = 1.0 / (2 * tau * dt)
    gamma = c**2 / dx**2
    
    coeff_next = alpha + beta
    
    print("Solving Telegrapher's Equation...")
    for n in tqdm(range(1, Nt - 1)):
        u_prev = u[n-1]
        u_curr = u[n]
        
        # Spatial laplacian (central difference)
        # Using slicing for efficiency: u[i+1] corresponds to u_curr[2:], u[i-1] to u_curr[:-2]
        laplacian = torch.zeros_like(u_curr)
        laplacian[1:-1] = u_curr[2:] - 2*u_curr[1:-1] + u_curr[:-2]
        
        # Explicit update rule
        # (alpha + beta) * u_{n+1} = (2*alpha) * u_n - (alpha - beta) * u_{n-1} + gamma * laplacian
        
        rhs = (2 * alpha) * u_curr - (alpha - beta) * u_prev + gamma * laplacian
        u_next = rhs / coeff_next
        
        # Boundary Conditions (Dirichlet)
        u_next[0] = 0
        u_next[-1] = 0
        
        u[n+1] = u_next

    # --- Gaussian Comparison (Heat Equation) ---
    # For large t, Telegrapher's -> Heat equation with D = c^2 * tau
    D_eff = c**2 * tau
    
    def gaussian(x_t, t_val):
        if t_val == 0:
             return torch.exp(-0.5 * (x_t / sigma_0)**2) / (sigma_0 * np.sqrt(2 * np.pi))
        # Variance grows as sigma_0^2 + 2*D*t
        sigma_t = np.sqrt(sigma_0**2 + 2 * D_eff * t_val)
        return torch.exp(-0.5 * (x_t / sigma_t)**2) / (sigma_t * np.sqrt(2 * np.pi))

    # --- Visualization ---
    print("Generating Animation...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    line_tele, = ax.plot([], [], 'b-', label="Telegrapher's (Relativistic)", linewidth=2)
    line_gauss, = ax.plot([], [], 'r--', label=f"Gaussian (Heat Eq, D={D_eff:.1f})", alpha=0.6)
    
    # Light cones markers
    cone_left = ax.axvline(x=0, color='k', linestyle=':',  alpha=0.5, label='Light Cone (|x| = ct)')
    cone_right = ax.axvline(x=0, color='k', linestyle=':', alpha=0.5)
    
    ax.set_xlim(-L, L)
    ax.set_ylim(0, torch.max(u).item() * 1.1)
    ax.set_xlabel("x (Log-Price / Position)")
    ax.set_ylabel("Probability Density u(t, x)")
    ax.set_title(f"Relativistic Option Pricing: Telegrapher's vs Heat Eq\n$c={c}, \\tau={tau}$")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    text_time = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        line_tele.set_data([], [])
        line_gauss.set_data([], [])
        cone_left.set_xdata([0])
        cone_right.set_xdata([0])
        text_time.set_text('')
        return line_tele, line_gauss, cone_left, cone_right, text_time

    # Downsample frames for smoother/faster GIF
    skip = 5
    frames = range(0, Nt, skip)

    def update(frame_idx):
        t_current = t_grid[frame_idx].item()
        
        # Telegrapher solution
        y_tele = u[frame_idx].numpy()
        line_tele.set_data(x.numpy(), y_tele)
        
        # Gaussian analytical solution
        y_gauss = gaussian(x, t_current).numpy()
        line_gauss.set_data(x.numpy(), y_gauss)
        
        # Update Light Cones: x = +/- c * t
        cone_dist = c * t_current
        cone_left.set_xdata([-cone_dist, -cone_dist])
        cone_right.set_xdata([cone_dist, cone_dist])
        
        text_time.set_text(f"t = {t_current:.3f}")
        return line_tele, line_gauss, cone_left, cone_right, text_time

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    
    output_path = "telegrapher_evolution.gif"
    ani.save(output_path, writer='pillow', fps=30)
    print(f"Animation saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    solve_telegrapher()
