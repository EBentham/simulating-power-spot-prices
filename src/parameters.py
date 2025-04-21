from dataclasses import dataclass


@dataclass
class MeanReversionParams:
    """Parameters for the mean-reverting component."""
    alpha: float  # Speed of mean reversion
    mu: float     # Long-term mean (log-price) - base level before seasonality
    sigma: float  # Volatility (base level before seasonality/stochastic vol)

@dataclass
class JumpParams:
    """Parameters for the jump component."""
    lambda_j: float # Jump intensity (average jumps per year)
    mu_j: float     # Mean jump size (log-percentage)
    sigma_j: float  # Jump size volatility (log-percentage)

@dataclass
class SimulationParams:
    """Parameters for the simulation execution."""
    s0: float       # Initial spot price
    t0: float       # Start time (years)
    T: float        # End time (years)
    dt: float       # Time step size (years)
    num_paths: int  # Number of simulation paths