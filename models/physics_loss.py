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
