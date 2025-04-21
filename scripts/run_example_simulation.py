import numpy as np
from matplotlib import pyplot as plt

from src.parameters import SimulationParams, MeanReversionParams, JumpParams
from src.processes import MeanRevertingJumpProcess
from src.seasonality import SeasonalityModel
from src.simulation import SimulationEngine

if __name__ == "__main__":
    # --- Configuration ---
    sim_params = SimulationParams(
        s0=25.0,  # Initial electricity price ($/MWh)
        t0=0.0,
        T=1.0,  # Simulate for 1 year
        dt=1 / 252,  # Daily steps (approx. trading days)
        num_paths=1000
    )

    mr_params = MeanReversionParams(
        alpha=5.0,  # Strong mean reversion typical for electricity
        mu=np.log(30.0),  # Base long-term mean log-price (around $30/MWh)
        sigma=0.8  # Base volatility (high for electricity)
    )

    jump_params = JumpParams(
        lambda_j=50.0,  # Avg 50 jumps per year (can be frequent in electricity)
        mu_j=0.0,  # Mean jump size (log %) - centered around 0
        sigma_j=0.4  # Volatility of jump size (log %) - significant jumps possible
    )

    # Define Seasonality (example: higher mean/vol in winter/summer)
    # Centered around t=0. Adjust phase if needed.
    seasonality = SeasonalityModel(
        base_mu=mr_params.mu,
        mu_amplitude=np.log(1.3),  # Mean can swing +/- ~30% seasonally
        mu_freq=1.0,  # Yearly cycle
        base_sigma=mr_params.sigma,
        sigma_amplitude=0.4,  # Volatility also swings
        sigma_freq=1.0  # Yearly cycle
    )

    # --- Strategy Selection ---
    # Create the concrete strategy instance
    # We can easily swap this for other strategies later
    price_process = MeanRevertingJumpProcess(mr_params, jump_params, seasonality)

    # --- Simulation ---
    # Create the context (engine) and inject the strategy
    engine = SimulationEngine(price_process, sim_params)
    spot_paths, time_points = engine.run_simulation()

    # --- Output/Plotting ---
    print("Simulation complete.")
    print(f"Final average price: {np.mean(spot_paths[-1, :]):.2f}")
    print(f"Standard deviation of final prices: {np.std(spot_paths[-1, :]):.2f}")

    plt.figure(figsize=(10, 6))
    # Plot a few paths
    num_paths_to_plot = 5
    plt.plot(time_points, spot_paths[:, :num_paths_to_plot])
    plt.title(f'Simulated Electricity Spot Price Paths (First {num_paths_to_plot}) - MRJP with Seasonality')
    plt.xlabel('Time (Years)')
    plt.ylabel('Spot Price ($/MWh)')
    plt.grid(True)

    # Add expected seasonal mean path for reference
    seasonal_mean_log = [seasonality.get_long_term_mean(t) for t in time_points]
    plt.plot(time_points, np.exp(seasonal_mean_log), 'k--', label='Seasonal Mean (exp(mu_t))', linewidth=2)
    plt.legend()
    plt.show()