import torch

def pde_residual(model, t, x, c, tau):
    """
    Computes the residual of the Relativistic Telegrapher's Equation:
    R = ∂²u/∂t² + (1/τ)∂u/∂t - c²∂²u/∂x²
    
    Args:
        model: The PINN model u_theta(t, x)
        t, x: Collocation points (requires_grad=True)
        c: Wave speed parameter
        tau: Relaxation time parameter
        
    Returns:
        residual: Tensor of PDE errors at each point.
    """
    # Enable gradient tracking for inputs
    # Note: t and x must have requires_grad=True BEFORE passing here, 
    # but we can ensure it or clone. Assuming inputs are already prepared.
    
    u = model(t, x)
    
    # First derivatives
    # grad_outputs=ones_like(u) creates the vector-Jacobian product for scalar L = sum(u)
    # create_graph=True is essential for higher-order derivatives (computing grad of grad)
    grads = torch.autograd.grad(
        outputs=u,
        inputs=[t, x],
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )
    
    u_t = grads[0]
    u_x = grads[1]
    
    # Second derivatives
    u_tt = torch.autograd.grad(
        outputs=u_t,
        inputs=t,
        grad_outputs=torch.ones_like(u_t),
        create_graph=True, # Need graph if we were optimizing w.r.t parameters involved in derivatives (e.g. inverse problems), here maybe not strictly needed for u_tt but good practice.
        retain_graph=True
    )[0]
    
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Telegrapher's Equation Residual
    # ∂²u/∂t² + (1/τ)∂u/∂t - c²∂²u/∂x² = 0
    residual = u_tt + (1.0 / tau) * u_t - (c**2) * u_xx
    
    return residual

def martingale_residual(model, t_slice, x_grid, r=0.0):
    """
    Computes the Martingale Constraint violation.
    E^Q [ S_T ] = S_0 * e^{rT}
    => E^Q [ S_0 * e^x ] = S_0 * e^{rT}
    => Integral( e^x * u(t,x) dx ) = e^{rt}
    
    Args:
        model: PINN model
        t_slice: Tensor of shape (N, 1), representing a specific time slice T (usually constant)
        x_grid: Tensor of shape (N, 1), integration points covering the domain
        r: Risk free rate
        
    Returns:
        Scalar tensor representing the squared error of the martingale condition.
    """
    # 1. Get PDF values
    u_pred = model(t_slice, x_grid)
    u_pdf = torch.relu(u_pred) # Force positivity for valid probability
    
    # 2. Integration Weight (Riemann Sum approx)
    # Assuming x_grid is sorted and uniform? 
    # If random collocation, this is Monte Carlo integration.
    # Let's assume x_grid is uniform linspace for stability here, or MC average.
    # MC Integration: Integral(f(x) dx) ~= Mean(f(x)) * Length(Domain)
    
    # Let's use MC for flexibility with random points
    # Integral ~ Mean( e^x * u(x) ) * Volume? 
    # No, usually we sample x uniformly from [-L, L].
    # Integral = (2L) * Mean( e^x * u(x) )
    
    L_val = 0.5 # Domain limit used in training
    domain_length = 2 * L_val
    
    # Integrand: e^x * u(t,x)
    integrand = torch.exp(x_grid) * u_pdf
    
    integral_approx = domain_length * torch.mean(integrand)
    
    # Target
    # t_slice is a tensor, we take the mean time (should be constant slice)
    t_val = torch.mean(t_slice)
    target = torch.exp(r * t_val)
    
    return (integral_approx - target)**2
