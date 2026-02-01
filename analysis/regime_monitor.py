import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

def load_data():
    try:
        df = pd.read_csv("rolling_parameters.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print("Error: rolling_parameters.csv not found.")
        sys.exit(1)

def run_regime_monitor():
    print("Initializing Regime Monitor (Kill Switch)...")
    df = load_data()
    
    # Analyze Loss distribution
    # We assume log-normal loss distribution roughly? Or just use raw Z-score.
    # Loss can be spiky. Let's use Log Loss for Z-score to be more robust.
    df['log_loss'] = np.log(df['loss'])
    
    loss_mean = df['log_loss'].mean()
    loss_std = df['log_loss'].std()
    
    df['loss_z'] = (df['log_loss'] - loss_mean) / loss_std
    
    # Thresholds
    WARN_THRESH = 1.5
    KILL_THRESH = 2.5
    
    df['status'] = 'GREEN'
    df.loc[df['loss_z'] > WARN_THRESH, 'status'] = 'YELLOW'
    df.loc[df['loss_z'] > KILL_THRESH, 'status'] = 'RED (HALT)'
    
    # Print Critical Events
    print("\n--- Regime Alerts ---")
    critical_events = df[df['status'] != 'GREEN']
    if len(critical_events) > 0:
        for _, row in critical_events.iterrows():
            print(f"Date: {row['date'].date()} | Loss: {row['loss']:.2f} (Z={row['loss_z']:.2f}) -> Status: {row['status']}")
    else:
        print("No critical anomalies detected in history.")
        
    # Current Status
    latest = df.iloc[-1]
    status_dict = {
        "date": str(latest['date'].date()),
        "model_loss": float(latest['loss']),
        "loss_z_score": float(latest['loss_z']),
        "trading_status": latest['status'],
        "recommendation": "EXECUTE" if latest['status'] == 'GREEN' else "REDUCE/HALT"
    }
    
    with open("trading_status.json", "w") as f:
        json.dump(status_dict, f, indent=4)
        
    print(f"\nLast Update: {status_dict['date']}")
    print(f"Current Status: {status_dict['trading_status']}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Background coloring
    # We can't easily shade distinct regions without loop, so we plot colored points
    colors = {'GREEN': 'green', 'YELLOW': 'orange', 'RED (HALT)': 'red'}
    
    plt.plot(df['date'], df['loss'], 'k-', alpha=0.3, label='Physics Loss')
    
    for status, color in colors.items():
        subset = df[df['status'] == status]
        plt.scatter(subset['date'], subset['loss'], color=color, label=status, s=50)
        
    plt.axhline(np.exp(loss_mean + KILL_THRESH * loss_std), color='red', linestyle='--', label='Kill Threshold')
    plt.axhline(np.exp(loss_mean + WARN_THRESH * loss_std), color='orange', linestyle='--', label='Warning Threshold')
    
    plt.yscale('log')
    plt.title(f"Regime Monitor: Model Reliability\nCurrent: {latest['status']}")
    plt.ylabel("PINN Calibration Loss (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("regime_monitor.png")
    print("Saved regime_monitor.png")

if __name__ == "__main__":
    run_regime_monitor()
