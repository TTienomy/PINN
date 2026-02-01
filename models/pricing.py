import torch

def price_option(model, T, Strikes, S0, L=10.0, Nx=1000, device='cpu'):
    """
    Calculates option prices from the PINN's PDF output.
    
    Price = e^{-rT} * Integral( Payoff(S_T) * u(T, x) dx )
    where S_T = S0 * exp(x)
    
    Args:
        model: Trained PINN model u(t, x)
        T: Time to expiry
        Strikes: Tensor of strike prices [K1, K2, ...]
        S0: Current underlying price
        L: Integration domain bound [-L, L]
        Nx: Number of integration steps
    
    Returns:
        Tensor of option prices corresponding to Strikes.
    """
    # 1. Grid for integration
    x = torch.linspace(-L, L, Nx).to(device)
    dx = x[1] - x[0]
    
    # 2. Get PDF from model at time T
    t_tensor = (torch.ones(Nx) * T).reshape(-1, 1).to(device)
    x_tensor = x.reshape(-1, 1)
    
    # Disable gradient for the model evaluation part if we assume PDF is fixed, 
    # BUT for calibration we need gradients through the model w.r.t parameters (if parameters were inputs)
    # OR if model weights are fixed and we optimize c/tau, we can't use this function directly 
    # unless c/tau are inputs to the model.
    # CONSTANT PARAMETERS SCENARIO: 
    # If c, tau are part of the loss residual but not inputs to forward(), 
    # then they affect the model via training. 
    # INVERSE PROBLEM SCENARIO:
    # Usually we retrain the PINN (or fine-tune) to fit the prices.
    # So we need gradients flow from Price -> PDF -> Model Weights -> Residual(c,tau).
    
    u_pdf = model(t_tensor, x_tensor).squeeze() # (Nx,)
    
    # Normalize PDF just in case (optional, but good for stability)
    # integral = torch.sum(u_pdf) * dx
    # u_pdf = u_pdf / integral
    
    # 3. Calculate Prices for each Strike
    # Broadcasting: (NumStrikes, 1) vs (1, Nx)
    S_T = S0 * torch.exp(x) # (Nx,)
    
    prices = []
    for K in Strikes:
        payoff = torch.relu(S_T - K)
        # Trapezoidal integration: sum(y * dx)
        # Price = sum(payoff * pdf) * dx
        price = torch.sum(payoff * u_pdf) * dx
        prices.append(price)
        
    return torch.stack(prices)
