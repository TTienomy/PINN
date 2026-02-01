import torch
import torch.nn as nn

class TelegrapherPINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) for solving the 1D Telegrapher's Equation.
    
    Architecture:
    - Input: [t, x] (2 dimensions)
    - Hidden: 4 layers x 50 neurons
    - Output: u(t, x) (1 dimension)
    - Activation: Tanh (smooth derivatives for higher-order derivatives in PDE)
    """
    def __init__(self, hidden_layers=4, neurons=50):
        super(TelegrapherPINN, self).__init__()
        
        layers = []
        # Input layer: 2 inputs -> neurons
        layers.append(nn.Linear(2, neurons))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())
            
        # Output layer: neurons -> 1 output
        layers.append(nn.Linear(neurons, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights with Xavier/Glorot initialization for better convergence
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, x):
        """
        Forward pass: u = Net(t, x)
        Concatenates t and x tensors along the last dimension.
        """
        # Ensure inputs are column vectors
        inputs = torch.cat([t, x], dim=1)
        return self.net(inputs)

class ParametricTelegrapherPINN(nn.Module):
    """
    Adaptive PINN that solves for u(t, x | c, tau).
    It treats physical parameters c and tau as *inputs*, allowing
    a single model to handle varying market regimes.
    
    Architecture:
    - Input: [t, x, c, tau] (4 dimensions)
    - Hidden: 5 layers x 64 neurons (Slightly larger for harder problem)
    - Output: u(t, x)
    """
    def __init__(self, hidden_layers=5, neurons=64):
        super(ParametricTelegrapherPINN, self).__init__()
        
        layers = []
        # Input layer: 4 inputs -> neurons
        layers.append(nn.Linear(4, neurons))
        layers.append(nn.Tanh())
        
        for _ in range(hidden_layers):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(neurons, 1))
        
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, x, c, tau):
        """
        Forward pass with dynamic parameters.
        All inputs must have shape (N, 1).
        """
        inputs = torch.cat([t, x, c, tau], dim=1)
        return self.net(inputs)
