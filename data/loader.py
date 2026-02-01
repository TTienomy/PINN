import pandas as pd
import numpy as np
import os

def generate_dummy_data(filepath="data/deribit_sample.csv"):
    """
    Generates a dummy Deribit-style CSV for testing.
    Columns: timestamp, strike, price, underlying_price, expiry
    """
    print(f"Generating dummy data at {filepath}...")
    
    strikes = np.linspace(8000, 12000, 20)
    S0 = 10000.0
    
    # Simple pricing (just for placeholder)
    # real prices will come from the market or our synthetic generator
    prices = np.maximum(S0 - strikes, 0) + 100 # erratic prices
    
    df = pd.DataFrame({
        'timestamp': pd.Timestamp.now(),
        'strike': strikes,
        'price': prices,
        'underlying_price': S0,
        'expiry': pd.Timestamp.now() + pd.Timedelta(days=30)
    })
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath

def load_deribit_data(filepath):
    """
    Loads Deribit options data from CSV.
    """
    if not os.path.exists(filepath):
        print(f"File {filepath} not found. Generating dummy data.")
        generate_dummy_data(filepath)
        
    df = pd.read_csv(filepath)
    return df
