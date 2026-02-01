import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.pinn import TelegrapherPINN
from models.pricing import price_option

def debug_oracle():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = TelegrapherPINN().to(device)
    model.load_state_dict(torch.load("models/telegrapher_pinn.pth", map_location=device))
    model.eval()
    
    T = 1.0
    L = 10.0
    Nx = 1000
    x = torch.linspace(-L, L, Nx).to(device)
    dx = x[1] - x[0]
    
    t_tensor = torch.ones_like(x).reshape(-1, 1) * T
    x_tensor = x.reshape(-1, 1)
    
    with torch.no_grad():
        u_pdf = model(t_tensor, x_tensor).squeeze()
        
    u_cpu = u_pdf.cpu().numpy()
    x_cpu = x.cpu().numpy()
    
    # 1. Check Normalization
    integral = np.sum(u_cpu) * dx.item()
    print(f"PDF Integral over [-10, 10]: {integral:.4f}")
    
    # 2. Check Peak
    print(f"PDF Peak value: {np.max(u_cpu):.4f} at x={x_cpu[np.argmax(u_cpu)]:.2f}")
    
    # 3. Check Range (Positivity)
    print(f"PDF Min value: {np.min(u_cpu):.4f}")
    
    # 4. Check Prices Manually
    S0 = 10000.0
    K = 10000.0
    
    S_T = S0 * np.exp(x_cpu)
    payoff = np.maximum(S_T - K, 0)
    
    # Price = sum(payoff * u * dx)
    price_manual = np.sum(payoff * u_cpu) * dx.item()
    print(f"\nManual Price Calculation (K={K}):")
    print(f"Price: {price_manual:.2f}")
    
    # Plot PDF
    plt.figure()
    plt.plot(x_cpu, u_cpu)
    plt.title(f"Oracle PDF at T={T} (Integral={integral:.2f})")
    plt.xlabel("x")
    plt.grid(True)
    plt.savefig("debug_pdf.png")
    print("Saved debug_pdf.png")

if __name__ == "__main__":
    debug_oracle()
