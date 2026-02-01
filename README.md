# Relativistic Telegrapher's Equation for Crypto Pricing

**Project Event Horizon** explores the use of **Physics-Informed Neural Networks (PINNs)** to model cryptocurrency dynamics using the **Relativistic Telegrapher's Equation**. This framework replaces the infinite-speed assumption of Black-Scholes (Brownian Motion) with a finite wave speed ($c$) and relaxation time ($\tau$), recovering heavy tails and volatility smiles endogenously.

## Key Features

1.  **Telegrapher's Solver (FDM)**: Finite Difference verification of the PDE.
2.  **PINN Architecture**: Neural Network solver trained on physics residuals.
3.  **Real Data Calibration**: Inverse problem solution finding $c \approx 0.0056$ and $\tau \approx 5.11$ days for Bitcoin.
4.  **Volatility Smile**: Reconstructed implied volatility surface showing realistic skew.
5.  **Risk Backtest**: 99% VaR backtest demonstrating superior tail risk coverage compared to Gaussian models.

## Structure

- `models/`: PINN definition and physics loss functions.
- `data/`: Data loaders for CSV market data.
- `analysis/`: Scripts for calibration, smile reconstruction, and backtesting.
- `telegrapher_solver.py`: Standalone FDM solver for validation.

## Usage

**1. Calibration:**
```bash
python calibrate_returns.py
```

**2. Volatility Smile:**
```bash
python analysis/smile_reconstruction.py
```

**3. Risk Backtest:**
```bash
python analysis/backtest_var.py
```

## Results
See `analysis/final_report.md` for a detailed breakdown of findings.

## License
MIT
