import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import sys

def load_params():
    try:
        df = pd.read_csv("rolling_parameters.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print("Error: rolling_parameters.csv not found. Run rolling_calibration.py first.")
        sys.exit(1)

def apply_kalman_filter(series, n_iter=5):
    """
    Applies a 1D Kalman Filter (Local Level Model).
    State: x_t (hidden true value)
    Obs: z_t (noisy measurement)
    """
    # Simply assume Random Walk: x_t = x_{t-1} + noise
    # Obs: z_t = x_t + noise
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=series.iloc[0],
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.1 # Tune this: Lower = smoother, Higher = more responsive
    )
    
    # EM algorithm to estimate internal covariances
    kf = kf.em(series.values, n_iter=n_iter)
    
    # Smooth (Hindcast) and Filter (Nowcast)
    smoothed_state_means, _ = kf.smooth(series.values)
    filtered_state_means, filtered_state_covariances = kf.filter(series.values)
    
    return filtered_state_means.flatten(), filtered_state_covariances.flatten()

def run_kalman_analysis():
    df = load_params()
    
    print("Applying Kalman Filter to Parameters...")
    
    # Filter c
    c_smooth, c_cov = apply_kalman_filter(df['c'])
    df['c_filtered'] = c_smooth
    df['c_std'] = np.sqrt(c_cov)
    
    # Filter tau
    tau_smooth, tau_cov = apply_kalman_filter(df['tau'])
    df['tau_filtered'] = tau_smooth
    df['tau_std'] = np.sqrt(tau_cov)
    
    # Next Step Prediction (Naive Random Walk Forecast)
    # Pred_{t+1} = Current_Filtered_State
    # In a real system you might use a trend model (Local Linear Trend)
    
    print("Latest States:")
    latest = df.iloc[-1]
    print(f"Date: {latest['date'].date()}")
    print(f"Tau Raw: {latest['tau']:.4f} -> Filtered: {latest['tau_filtered']:.4f} (+/- {latest['tau_std']:.4f})")
    print(f"C   Raw: {latest['c']:.6f} -> Filtered: {latest['c_filtered']:.6f} (+/- {latest['c_std']:.6f})")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot Tau
    ax1.plot(df['date'], df['tau'], 'r.', alpha=0.3, label='Raw Measurment')
    ax1.plot(df['date'], df['tau_filtered'], 'r-', linewidth=2, label='Kalman Estimate')
    ax1.fill_between(df['date'], 
                     df['tau_filtered'] - 1.96*df['tau_std'], 
                     df['tau_filtered'] + 1.96*df['tau_std'], 
                     color='red', alpha=0.1, label='95% Confidence')
    ax1.set_ylabel('Relaxation Time (tau)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Parameter Tracking: Tau (Trend Filtering)")
    
    # Plot C
    ax2.plot(df['date'], df['c'], 'b.', alpha=0.3, label='Raw Measurment')
    ax2.plot(df['date'], df['c_filtered'], 'b-', linewidth=2, label='Kalman Estimate')
    ax2.fill_between(df['date'], 
                     df['c_filtered'] - 1.96*df['c_std'], 
                     df['c_filtered'] + 1.96*df['c_std'], 
                     color='blue', alpha=0.1, label='95% Confidence')
    ax2.set_ylabel('Wave Speed (c)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Parameter Tracking: C (Noise Filtering)")
    
    plt.savefig("kalman_tracking.png")
    print("Saved kalman_tracking.png")
    
    # Save Enriched Data
    df.to_csv("rolling_parameters_filtered.csv", index=False)

if __name__ == "__main__":
    run_kalman_analysis()
