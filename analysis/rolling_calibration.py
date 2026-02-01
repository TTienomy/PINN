import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from models.pinn import TelegrapherPINN
from models.physics_loss import pde_residual

def load_data(csv_path="BTC-PERPETUAL.csv"):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['close'] = pd.to_numeric(df['close'])
    df['ret'] = np.log(df['close'] / df['close'].shift(1))
    return df.dropna()

def train_window(model, returns, device, is_warm_start=False):
    # Setup Target PDF
    kde = gaussian_kde(returns, bw_method='scott')
    L = 0.5
    dict_x = np.linspace(-L, L, 1000)
    dict_y = kde(dict_x)
    
    x_target = torch.tensor(dict_x, dtype=torch.float32).view(-1, 1).to(device)
    u_target = torch.tensor(dict_y, dtype=torch.float32).view(-1, 1).to(device)
    
    # Optimizer
    # We learn C and Tau
    # If model is new, we need to init params.
    # If warm start, we assume model already has params attached? 
    # Actually, model.parameters() are weights. c and tau are separate tensors usually.
    # To enable warm start for c and tau, we need to pass them in or store them in model.
    
    pass

class RollingPINN:
    def __init__(self, device):
        self.device = device
        self.model = TelegrapherPINN().to(device)
        self.c = torch.tensor([0.2], device=device, requires_grad=True) # Start neutral
        self.tau = torch.tensor([2.0], device=device, requires_grad=True)
        
    def train(self, returns, epochs):
        # Target
        kde = gaussian_kde(returns, bw_method='scott')
        L = 0.5
        dict_x = np.linspace(-L, L, 1000)
        dict_y = kde(dict_x)
        
        x_target = torch.tensor(dict_x, dtype=torch.float32).view(-1, 1).to(self.device)
        u_target = torch.tensor(dict_y, dtype=torch.float32).view(-1, 1).to(self.device)
        
        optimizer = optim.Adam([
            {'params': self.model.parameters(), 'lr': 1e-3},
            {'params': [self.c, self.tau], 'lr': 5e-3}
        ])
        
        Nx = 1500
        T = 1.0
        
        # Training Loop
        for i in range(epochs):
            # Sample Physics Points
            t_p = (torch.rand(Nx, 1) * T).to(self.device).requires_grad_(True)
            x_p = ((torch.rand(Nx, 1) * 2 * L) - L).to(self.device).requires_grad_(True)
            
            # Sample Data Points
            t_d = torch.ones_like(x_target) * T
            
            # IC
            t_ic = torch.zeros(300, 1).to(self.device)
            x_ic = ((torch.rand(300, 1) * 2 * L) - L).to(self.device)
            u_ic_target = torch.exp(-0.5 * (x_ic/0.005)**2) / (0.005 * np.sqrt(2*np.pi))
            
            optimizer.zero_grad()
            
            # Physics Loss
            c_val = torch.abs(self.c) + 1e-6
            tau_val = torch.abs(self.tau) + 1e-6
            
            res = pde_residual(self.model, t_p, x_p, c_val, tau_val)
            l_pde = torch.mean(res**2)
            
            # IC Loss
            l_ic = torch.mean((self.model(t_ic, x_ic) - u_ic_target)**2)
            
            # Data Loss
            u_pred = self.model(t_d, x_target)
            l_data = torch.mean((u_pred - u_target)**2)
            
            loss = l_pde + 10.0*l_ic + 100.0*l_data
            loss.backward()
            optimizer.step()
            
        return loss.item(), torch.abs(self.c).item(), torch.abs(self.tau).item()

def run_rolling_calibration():
    print("Starting Rolling Window Calibration...")
    df = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    agent = RollingPINN(device)
    
    # Parameters
    WINDOW_SIZE = 365 
    STEP_SIZE = 90 # Re-calibrate every ~3 months to save time
    
    results = []
    
    # Create Indices
    # Start at index WINDOW_SIZE
    indices = range(WINDOW_SIZE, len(df), STEP_SIZE)
    
    print(f"Total Calibration Steps: {len(indices)}")
    
    import time
    start_global = time.time()
    
    for idx in tqdm(indices):
        # Slice Data
        window_data = df.iloc[idx-WINDOW_SIZE : idx]
        date = df.iloc[idx]['date']
        returns = window_data['ret'].values
        
        # Determine Epochs: First run needs more, subsequent needs less (Warm Start)
        if idx == indices[0]:
            epochs = 2000
        else:
            epochs = 300 # Warm update
            
        loss, c_est, tau_est = agent.train(returns, epochs)
        
        results.append({
            'date': date,
            'c': c_est,
            'tau': tau_est,
            'loss': loss,
            'price': df.iloc[idx]['close']
        })
        
        tqdm.write(f"Date: {date.date()} | c={c_est:.4f} | tau={tau_est:.2f} | Loss={loss:.4f}")
        
    print(f"Completed in {time.time() - start_global:.0f}s")
    
    # Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv("rolling_parameters.csv", index=False)
    
    # Visualization
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot Price
    ax1.plot(res_df['date'], res_df['price'], color='gray', alpha=0.3, label='BTC Price')
    ax1.set_yscale('log')
    ax1.set_ylabel('BTC Price (Log Scale)')
    
    # Plot Parameters
    ax2 = ax1.twinx()
    ax2.plot(res_df['date'], res_df['c'], color='red', marker='o', label='Wave Speed (c)')
    ax2.plot(res_df['date'], res_df['tau'], color='blue', marker='x', label='Relaxation (tau)')
    ax2.set_ylabel('Physical Parameters')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title("Relativistic Regime Switching:\nRolling Calibration (Window=365d)")
    plt.savefig("rolling_regimes.png")
    print("Saved rolling_regimes.png")

if __name__ == "__main__":
    run_rolling_calibration()
